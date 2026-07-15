"""Channels: the typed-state substrate.

A graph's state is not a plain dict — it is a set of **channels**. Each channel owns one
value and knows how to fold new updates into it via a **reducer**. This is what makes the BSP
runtime work: within a superstep every node writes *updates*, and at the barrier each channel
reduces the updates it received into its next value. The reducer is where "last write wins",
"append to a list", or "sum the numbers" is decided — per field, by type.

Channels are declared implicitly by the fields of a state `Schema`. A plain field becomes a
`LastValue`; annotate it with `Reducer(op)` or `Topic()` to pick a different fold::

    class State(Schema):
        answer: str                                  # LastValue: one writer per step
        scratch: Annotated[list[str], Reducer(add)]  # concurrent writers accumulate
        events: Annotated[list[str], Topic()]        # stream: written lists concatenated

This module intentionally has no dependency on nodes, graphs, or the runtime — it is the
lowest layer of the type spine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic.fields import FieldInfo

from .errors import EmptyChannelError, InvalidUpdateError

Value = TypeVar("Value")
Update = TypeVar("Update")


class Channel(ABC, Generic[Value, Update]):
    """One typed cell of graph state with a reduction rule.

    `Value` is what readers see; `Update` is what writers contribute in a superstep. For most
    channels they are the same type, but they can differ (a `Topic[str]` reads as `list[str]`).
    """

    __slots__ = ()

    @abstractmethod
    def get(self) -> Value:
        """Return the current value, or raise `EmptyChannelError` if never written."""

    @abstractmethod
    def update(self, updates: Sequence[Update]) -> bool:
        """Fold the updates from one superstep into the value.

        Returns True iff the value changed (used by the scheduler to decide what to notify).
        Called exactly once per channel per barrier, with every update written that step.
        """

    @property
    @abstractmethod
    def is_set(self) -> bool:
        """Whether this channel has a value yet."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Return a serializable copy of the value (called only when `is_set`).

        Used to checkpoint state at a barrier. Must round-trip through `restore`.
        """

    @abstractmethod
    def restore(self, value: Any) -> None:
        """Rehydrate the channel's value from a `snapshot` (inverse of `snapshot`)."""


class LastValue(Channel[Value, Value]):
    """Keep the most recent write. The default channel.

    Rejects two writes in a single superstep — that would make the result depend on scheduling
    order. If concurrent writers are intended, use a reducer channel instead.
    """

    __slots__ = ("_set", "_value")

    def __init__(self) -> None:
        self._value: Value = None  # type: ignore[assignment]
        self._set = False

    def get(self) -> Value:
        if not self._set:
            raise EmptyChannelError("channel has no value yet")
        return self._value

    def update(self, updates: Sequence[Value]) -> bool:
        if not updates:
            return False
        if len(updates) > 1:
            raise InvalidUpdateError(
                f"LastValue channel received {len(updates)} writes in one superstep; "
                f"give this field a reducer (e.g. Reducer(add)) to combine concurrent writes"
            )
        self._value = updates[-1]
        self._set = True
        return True

    @property
    def is_set(self) -> bool:
        return self._set

    def snapshot(self) -> Any:
        return self._value

    def restore(self, value: Any) -> None:
        self._value = value
        self._set = True


class BinaryOperatorAggregate(Channel[Value, Value]):
    """Reduce writes with an associative binary operator, e.g. `operator.add`.

    The first write seeds the value; each subsequent write (this step or later) is folded in as
    `value = op(value, update)`. This is how `Annotated[list, Reducer(add)]` accumulates across
    both concurrent writers and successive supersteps.
    """

    __slots__ = ("_op", "_set", "_value")

    def __init__(self, op: Callable[[Value, Value], Value]) -> None:
        self._value: Value = None  # type: ignore[assignment]
        self._set = False
        self._op = op

    def get(self) -> Value:
        if not self._set:
            raise EmptyChannelError("channel has no value yet")
        return self._value

    def update(self, updates: Sequence[Value]) -> bool:
        if not updates:
            return False
        for u in updates:
            if not self._set:
                self._value = u
                self._set = True
            else:
                self._value = self._op(self._value, u)
        return True

    @property
    def is_set(self) -> bool:
        return self._set

    def snapshot(self) -> Any:
        return self._value

    def restore(self, value: Any) -> None:
        self._value = value
        self._set = True


class Topic(Channel[list[Value], list[Value]]):
    """A stream channel: each write is a *list* of items, concatenated into one growing list.

    Value and update are both `list[Value]`, so a node whose `Out` field is `list[str]` writes a
    (possibly singleton) list and it is appended to the channel. Reads return the full list. With
    `accumulate=True` (default) the list grows across supersteps; with `accumulate=False` it holds
    only the most recent step's writes (useful for one-shot fan-out payloads). Always "set" — an
    unwritten Topic reads as an empty list.
    """

    __slots__ = ("_accumulate", "_values")

    def __init__(self, accumulate: bool = True) -> None:
        self._values: list[Value] = []
        self._accumulate = accumulate

    def get(self) -> list[Value]:
        return list(self._values)

    def update(self, updates: Sequence[list[Value]]) -> bool:
        changed = False
        if not self._accumulate:
            self._values = []
            changed = True
        for batch in updates:
            if batch:
                self._values.extend(batch)
                changed = True
        return changed

    @property
    def is_set(self) -> bool:
        return True

    def snapshot(self) -> Any:
        return list(self._values)

    def restore(self, value: Any) -> None:
        self._values = list(value)


@dataclass(frozen=True)
class Reducer:
    """Field annotation selecting a `BinaryOperatorAggregate` for a state field.

    Use inside `Annotated` on a state Schema field::

        from operator import add
        scratch: Annotated[list[str], Reducer(add)]
    """

    op: Callable[[Any, Any], Any]


def channel_for_field(field: FieldInfo) -> Channel[Any, Any]:
    """Build the channel a state field maps to, honoring its `Annotated` reducer marker.

    `Reducer(op)` → `BinaryOperatorAggregate(op)`; a `Topic()` marker → that Topic; anything
    else → `LastValue`. The marker is read from Pydantic's collected field metadata.
    """
    for meta in field.metadata:
        if isinstance(meta, Reducer):
            return BinaryOperatorAggregate(meta.op)
        if isinstance(meta, Topic):
            # The annotation carries a Topic used only as a marker; hand back a *fresh*
            # channel so state never leaks between runs.
            return Topic(accumulate=meta._accumulate)
    return LastValue()
