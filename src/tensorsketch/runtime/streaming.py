"""Streaming: a run emits a live, namespaced, resumable event stream.

Everything interesting a run does becomes an `Event`: a node starting, a node finishing with
its writes, the merged state after a barrier, and whatever a node body chooses to emit itself
(`ctx.emit(...)` — the seam through which LLM token deltas will flow in a later phase). Every
event carries a **namespace** (`run_id`, `thread_id`, `node`) so a consumer can separate lanes
in a multi-agent run, and a monotonic **`seq`** cursor so a stream can be replayed from a point.

The transport is deliberately tiny: `event_stream` bridges an async producer (the superstep
loop) to an async consumer via a bounded queue. The bound is the backpressure — if the consumer
is slow, `emit` awaits, which pauses the producing node.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

# Framework event types. Node bodies may emit any other (custom) type via `ctx.emit`.
RUN_START = "run_start"
NODE_START = "node_start"
NODE_END = "node_end"
VALUES = "values"
RUN_END = "run_end"


@dataclass(frozen=True)
class Event:
    """One thing that happened during a run.

    Attributes:
        seq: Monotonic per-run cursor (0-based). Replay resumes after a given `seq`.
        run_id: The `invoke`/`stream` call this event came from.
        thread_id: The durable run key (empty when durability is off).
        superstep: The BSP superstep the event belongs to.
        node: The node the event is about, or None for run-level events.
        type: One of the framework constants above, or a custom name from `ctx.emit`.
        data: Event payload — e.g. `{"writes": ...}` for NODE_END, `{"state": ...}` for VALUES,
            or whatever a node passed to `ctx.emit`.
    """

    seq: int
    run_id: str
    thread_id: str
    superstep: int
    node: str | None
    type: str
    data: dict[str, Any] = field(default_factory=dict)


# The runtime calls this to emit an event: (type, node, superstep, data). Identity (run_id/
# thread_id) and the `seq` cursor are added by the stream that owns the emitter.
EmitFn = Callable[[str, str | None, int, Mapping[str, Any]], Awaitable[None]]

# A run driver: given an emitter, perform the run (drive the superstep loop) to completion.
Runner = Callable[[EmitFn], Awaitable[None]]


@dataclass
class _Done:
    """Sentinel queued when the producer has finished."""


async def event_stream(
    runner: Runner,
    *,
    run_id: str,
    thread_id: str,
    append: Callable[[Event], None] | None = None,
    buffer: int = 256,
) -> AsyncIterator[Event]:
    """Run `runner`, yielding each `Event` it emits, live.

    Args:
        runner: Drives the actual work; receives the `emit` function to call.
        run_id: Stamped onto every event.
        thread_id: Stamped onto every event.
        append: If given, each event is also handed here (to persist for resumable replay).
        buffer: Max in-flight events before `emit` applies backpressure.

    The producer runs as a background task; if the consumer stops early, the task is cancelled
    cleanly. An exception raised by the run surfaces to the consumer after its events drain.
    """
    queue: asyncio.Queue[Event | _Done] = asyncio.Queue(maxsize=buffer)
    counter = itertools.count()

    async def emit(type: str, node: str | None, superstep: int, data: Mapping[str, Any]) -> None:
        event = Event(
            seq=next(counter),
            run_id=run_id,
            thread_id=thread_id,
            superstep=superstep,
            node=node,
            type=type,
            data=dict(data),
        )
        if append is not None:
            append(event)
        await queue.put(event)

    async def drive() -> None:
        try:
            await emit(RUN_START, None, 0, {})
            await runner(emit)
            await emit(RUN_END, None, 0, {})
        finally:
            await queue.put(_Done())

    task = asyncio.create_task(drive())
    try:
        while True:
            item = await queue.get()
            if isinstance(item, _Done):
                break
            yield item
        await task  # re-raise any error from the run, once its events have been delivered
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
