"""Durability: checkpoints (resume/fork) and an effect journal (exactly-once side effects).

TensorSketch's durability rests on two layers, and both live behind one `Backend` interface:

1. **Checkpoints** — at every superstep barrier the runtime snapshots the channel values and the
   set of nodes about to run next. A checkpoint is a consistent point the run can **resume**
   from (or **fork** from). This is the "where are we" layer.

2. **The effect journal** — every side effect wrapped in `ctx.step(name, fn)` records its result
   the moment it completes, keyed deterministically. On resume, a recorded effect is **returned
   from the journal instead of re-run**. This is the "don't do it twice" layer — the difference
   between *checkpointing* (which re-runs the whole failed superstep, repeating side effects) and
   *durable execution* (which doesn't). Results are stored **as data**, so orchestration code
   carries no determinism constraints.

Backends are pluggable: `InMemoryBackend` (here) for dev/tests, `SqliteBackend` for local
persistence, and `PostgresBackend` / `RedisBackend` for production — all behind this one ABC. The
framework holds **no durable state of its own**; it all lives in whichever `Backend` you supply,
so an agent stays stateless and you bring your own database. See `tensorsketch.runtime.backends`.
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.channels import Channel
from .streaming import Event


@dataclass(frozen=True)
class Checkpoint:
    """A consistent snapshot of a run at one superstep boundary.

    Attributes:
        thread_id: The run/thread this checkpoint belongs to.
        id: Unique id of this checkpoint (the target for fork/resume).
        parent_id: The checkpoint this one descends from — checkpoints form a tree, so forks are
            first-class.
        step: The superstep index at which `active` will run when resumed.
        channel_values: `{channel: snapshot}` for every set channel (see `Channel.snapshot`).
        active: The node names scheduled to run at `step`. Empty means the run has halted.
        sends: Pending dynamic fan-out at `step` — `(node, payload)` pairs, one scheduled
            instance each (see `tensorsketch.Send`). Ordered, so instances resume with stable keys.
            The run has halted only when both `active` and `sends` are empty.
    """

    thread_id: str
    id: str
    parent_id: str | None
    step: int
    channel_values: dict[str, Any]
    active: frozenset[str]
    sends: tuple[tuple[str, dict[str, Any]], ...] = ()


class Backend(ABC):
    """Storage for checkpoints and the effect journal. One object serves both roles."""

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    def latest_checkpoint(self, thread_id: str) -> Checkpoint | None:
        """The most recent checkpoint for a thread, or None if it has never run."""

    @abstractmethod
    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None:
        """A specific checkpoint by id (used to fork)."""

    @abstractmethod
    def list_checkpoints(self, thread_id: str) -> list[Checkpoint]:
        """All checkpoints for a thread, oldest first."""

    @abstractmethod
    def record_effect(self, thread_id: str, key: str, result: Any) -> None:
        """Persist the result of a durable step under a deterministic key."""

    @abstractmethod
    def lookup_effect(self, thread_id: str, key: str) -> tuple[bool, Any]:
        """Return `(found, result)` for a previously recorded effect."""

    @abstractmethod
    def append_event(self, event: Event) -> None:
        """Persist a streamed event so the stream can be replayed from a cursor."""

    @abstractmethod
    def read_events(self, thread_id: str, since: int = 0) -> list[Event]:
        """Return persisted events for a thread with `seq >= since`, in order."""


# -- helpers shared by the runtime ---------------------------------------------------------


def capture(channels: Mapping[str, Channel[Any, Any]]) -> dict[str, Any]:
    """Snapshot every set channel into a serializable dict."""
    return {name: ch.snapshot() for name, ch in channels.items() if ch.is_set}


def apply(channels: MutableMapping[str, Channel[Any, Any]], values: Mapping[str, Any]) -> None:
    """Rehydrate channels from a captured snapshot (the inverse of `capture`)."""
    for name, value in values.items():
        if name in channels:
            channels[name].restore(value)


def save_checkpoint(
    backend: Backend,
    thread_id: str,
    parent_id: str | None,
    step: int,
    channels: Mapping[str, Channel[Any, Any]],
    active: frozenset[str] | set[str],
    sends: Sequence[tuple[str, dict[str, Any]]] = (),
) -> str:
    """Build and persist a checkpoint; return its new id (the parent of the next one)."""
    checkpoint = Checkpoint(
        thread_id=thread_id,
        id=uuid.uuid4().hex,
        parent_id=parent_id,
        step=step,
        channel_values=capture(channels),
        active=frozenset(active),
        sends=tuple(sends),
    )
    backend.save_checkpoint(checkpoint)
    return checkpoint.id


# -- in-memory backend ---------------------------------------------------------------------


def _copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    # Deep-copy the values so a stored checkpoint can never alias (and be mutated by) live
    # channel state; the same for any pending fan-out payloads.
    return dataclasses.replace(
        checkpoint,
        channel_values=copy.deepcopy(checkpoint.channel_values),
        sends=copy.deepcopy(checkpoint.sends),
    )


class InMemoryBackend(Backend):
    """Non-persistent backend for development and tests. Fast, and safe against aliasing."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, list[Checkpoint]] = defaultdict(list)
        self._effects: dict[tuple[str, str], Any] = {}
        self._events: dict[str, list[Event]] = defaultdict(list)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.thread_id].append(_copy_checkpoint(checkpoint))

    def latest_checkpoint(self, thread_id: str) -> Checkpoint | None:
        history = self._checkpoints.get(thread_id)
        return _copy_checkpoint(history[-1]) if history else None

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None:
        for checkpoint in self._checkpoints.get(thread_id, []):
            if checkpoint.id == checkpoint_id:
                return _copy_checkpoint(checkpoint)
        return None

    def list_checkpoints(self, thread_id: str) -> list[Checkpoint]:
        return [_copy_checkpoint(c) for c in self._checkpoints.get(thread_id, [])]

    def record_effect(self, thread_id: str, key: str, result: Any) -> None:
        self._effects[(thread_id, key)] = copy.deepcopy(result)

    def lookup_effect(self, thread_id: str, key: str) -> tuple[bool, Any]:
        if (thread_id, key) in self._effects:
            return True, copy.deepcopy(self._effects[(thread_id, key)])
        return False, None

    def append_event(self, event: Event) -> None:
        self._events[event.thread_id].append(event)

    def read_events(self, thread_id: str, since: int = 0) -> list[Event]:
        return [e for e in self._events.get(thread_id, []) if e.seq >= since]


# `SqliteBackend`, `PostgresBackend`, and `RedisBackend` live in `tensorsketch.runtime.backends`
# (the
# database connectors), so this module keeps no driver dependencies.
