"""`Send` — graph-level dynamic fan-out: schedule a node with an explicit payload, N times over.

A normal edge activates a node **once**, reading shared state. A `Send` instead schedules *one
instance* of a node with its *own* input. Return a list of them from a router and the engine spawns
a task per `Send` — each its own superstep unit — all merging at the next barrier. This is the map
half of a graph-level map/reduce: fan a collection out to workers, then converge on a reducer
channel a downstream node reads.

    from tensorsketch import Send

    def fan(state: State) -> list[Send]:
        return [Send("worker", {"item": x}) for x in state.items]

    g.router("split", fan)          # "split" fans out to one "worker" per item
    g.edge("worker", "collect")     # every worker flows into a single "collect" (deduped)

The payload provides the worker's `In` fields; any `In` field the payload omits still reads from
shared state. Workers should write an **aggregating** channel (a `Topic` or a reducer) so their
results merge at the barrier instead of overwriting one another.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .schema import Schema


@dataclass(frozen=True)
class Send:
    """Schedule one instance of `node`, fed by `input` (a `Schema` or a field mapping)."""

    node: str
    input: Schema | Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        """The input as a plain, serializable dict — used for scheduling and checkpoints."""
        return dict(self.input.model_dump() if isinstance(self.input, Schema) else self.input)
