"""The BSP / Pregel superstep engine.

This is the beating heart of TensorSketch's runtime, and it is deliberately tiny and untyped: it
schedules **processes** (opaque units of work) over **channels** (typed state cells) in
lockstep supersteps. Everything higher up — typed nodes, agents, routers, loops — compiles
into processes and channels, so the scheduling semantics live in exactly one place.

## Why BSP (Bulk Synchronous Parallel)

Each superstep has three phases:

1. **Plan** — the set of *active* processes for this step is known (initially the entry
   process; thereafter, whoever the previous step's processes named as successors).
2. **Execute** — every active process runs in parallel, each reading the *same* immutable
   snapshot of channel state taken at the start of the step. A process cannot see another
   process's writes mid-step, so there are no read/write races by construction.
3. **Barrier** — all writes collected this step are folded into channels via their reducers,
   atomically. Then successors are computed and become the next step's active set.

From this single model we get, for free: **cycles** (a process may name a predecessor as a
successor), **deterministic parallel fan-out** (siblings run on the same snapshot and merge at
the barrier), and a **clean checkpoint boundary** (the barrier is exactly where durable state
is consistent — so when a `Backend` is supplied, the engine snapshots there).

The engine mutates the `channels` mapping in place and returns the id of the last checkpoint it
wrote (or None). Durability is optional: without a `Backend` the loop is a plain in-memory run.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from ..core.channels import Channel
from ..core.context import Context
from ..core.errors import GraphError, GraphRecursionError
from ..core.send import Send
from ..observability.tracing import (
    NODE,
    NODE_KIND,
    NOOP,
    RUN,
    RUN_ID,
    SUPERSTEP,
    THREAD_ID,
    Tracer,
)
from .checkpoint import Backend, save_checkpoint
from .streaming import NODE_END, NODE_START, VALUES, EmitFn

# A process reads a snapshot of the channels it subscribes to and returns the channel updates
# it wants applied at the barrier (a mapping of channel-name -> update value).
RunFn = Callable[[Context, Mapping[str, Any]], Coroutine[Any, Any, Mapping[str, Any]]]

# Given the post-barrier state, a process names what to activate next superstep: process names
# (sequential/conditional edges) and/or `Send`s (dynamic fan-out with per-instance payloads).
SuccessorsFn = Callable[[Mapping[str, Any]], Iterable["str | Send"]]

# One scheduled unit of a superstep: a process, the inputs it runs on (channel reads, or a Send's
# merged payload), and an instance tag (empty for a normal node; set to disambiguate fan-out).
_Send = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class Process:
    """One schedulable unit in the superstep loop.

    Attributes:
        name: Unique identifier within a run.
        reads: Channel names to pull from the step snapshot and pass to `run`.
        run: Async work. Receives the context and `{channel: value}` for the readable subset of
            `reads` that is set; returns `{channel: update}` to apply at the barrier.
        successors: Given post-barrier state, returns the process names to activate next step.
            This is where sequential edges, conditional routing, and loop-backs are encoded.
    """

    name: str
    reads: tuple[str, ...]
    run: RunFn
    successors: SuccessorsFn


def _snapshot(channels: Mapping[str, Channel[Any, Any]]) -> dict[str, Any]:
    """A consistent read of every set channel — the immutable view a superstep executes on."""
    return {name: ch.get() for name, ch in channels.items() if ch.is_set}


@dataclass(frozen=True)
class _Unit:
    """One scheduled execution this superstep: a process, its inputs, and an instance tag."""

    proc: Process
    inputs: Mapping[str, Any]
    instance: str  # "" for a normally-scheduled node; an index for a fan-out instance


def _plan(
    processes: Mapping[str, Process],
    active: set[str],
    sends: list[_Send],
    snapshot: Mapping[str, Any],
) -> list[_Unit]:
    """Turn the active names and pending sends into the concrete units to run this step.

    Plain names run once, reading their channel subset of the snapshot (sorted for determinism so
    fan-out instance tags are stable across resume). Each `Send` runs its target once with the
    snapshot overlaid by its payload — so a worker sees shared state plus its per-instance input.
    """
    units: list[_Unit] = []
    for name in sorted(active):
        proc = processes[name]
        inputs = {r: snapshot[r] for r in proc.reads if r in snapshot}
        units.append(_Unit(proc, inputs, ""))
    for index, (node, payload) in enumerate(sends):
        target = processes.get(node)
        if target is None:
            raise GraphError(f"Send target {node!r} is not a node in the graph")
        merged = {r: snapshot[r] for r in target.reads if r in snapshot}
        merged.update({k: v for k, v in payload.items() if k in target.reads})
        units.append(_Unit(target, merged, str(index)))
    return units


async def _run_process(
    proc: Process, ctx: Context, inputs: Mapping[str, Any], emit: EmitFn | None
) -> Mapping[str, Any]:
    """Run one process in a node span, bracketing it with node_start/node_end events."""
    with ctx.span(proc.name, kind=NODE_KIND, **{NODE: proc.name, SUPERSTEP: ctx.superstep}):
        if emit is not None:
            await emit(NODE_START, proc.name, ctx.superstep, {})
        result = await proc.run(ctx, inputs)
        if emit is not None:
            await emit(NODE_END, proc.name, ctx.superstep, {"writes": dict(result)})
        return result


async def execute(
    processes: Mapping[str, Process],
    channels: MutableMapping[str, Channel[Any, Any]],
    active: set[str],
    *,
    run_id: str,
    thread_id: str = "",
    backend: Backend | None = None,
    start_step: int = 0,
    parent_id: str | None = None,
    sends: list[_Send] | None = None,
    max_steps: int = 25,
    emit: EmitFn | None = None,
    tracer: Tracer | None = None,
) -> str | None:
    """Run the superstep loop until quiescence, mutating `channels` in place.

    Args:
        processes: All processes, keyed by name.
        channels: The live state; updated at each barrier.
        active: Process names to run in the first superstep (typically the entry, or the active
            set restored from a checkpoint when resuming).
        run_id: Identifier threaded into each `Context`.
        thread_id: Durable run key; passed to each `Context` for effect journaling.
        backend: If given, a checkpoint is written after every barrier and each `ctx.step` is
            journaled through it. If None, the run is purely in-memory.
        start_step: The superstep index to start counting from (nonzero when resuming).
        parent_id: The checkpoint the run continues from (parent of the next checkpoint written).
        max_steps: Recursion limit. Exceeding it raises `GraphRecursionError` (an unguarded
            loop, almost always).

    Returns:
        The id of the last checkpoint written, or None if durability was off.

    Raises:
        GraphRecursionError: The loop did not settle within `max_steps`.
    """
    tracer = tracer or NOOP
    step = start_step
    last_checkpoint_id = parent_id
    pending_sends: list[_Send] = list(sends or [])
    # A run span covers the whole loop; node spans (opened in each task) nest under it via the
    # tracer's ContextVar — so the trace tree mirrors the execution without threading anything.
    with tracer.span("run", kind=RUN, **{RUN_ID: run_id, THREAD_ID: thread_id}):
        while active or pending_sends:
            if step >= max_steps:
                raise GraphRecursionError(
                    f"exceeded {max_steps} supersteps without halting (still active: "
                    f"{sorted(active)}). If this loop is intended to run longer, raise max_steps; "
                    f"otherwise check that its exit condition can be met."
                )

            snapshot = _snapshot(channels)
            units = _plan(processes, active, pending_sends, snapshot)

            # Phase 2 — execute every scheduled unit in parallel on the shared snapshot. Each gets
            # its own Context (so `ctx.step` keys effects by node + instance + call order — fan-out
            # instances of one node stay distinct). A TaskGroup gives structured concurrency (a
            # failing sibling cancels the rest) but reports an ExceptionGroup; for the common
            # single-failure case we unwrap so callers see the real error, not a wrapper.
            tasks: list[asyncio.Task[Mapping[str, Any]]] = []
            try:
                async with asyncio.TaskGroup() as tg:
                    for unit in units:
                        ctx = Context(
                            run_id=run_id,
                            thread_id=thread_id,
                            superstep=step,
                            node=unit.proc.name,
                            backend=backend,
                            emit=emit,
                            tracer=tracer,
                            instance=unit.instance,
                        )
                        coro = _run_process(unit.proc, ctx, unit.inputs, emit)
                        tasks.append(tg.create_task(coro))
            except BaseExceptionGroup as group:
                if len(group.exceptions) == 1:
                    raise group.exceptions[0] from None
                raise

            # Phase 3 — barrier: collect writes from every unit, then fold them into channels via
            # reducers (this is where fan-out instances merge into an aggregating channel).
            writes: dict[str, list[Any]] = defaultdict(list)
            for task in tasks:
                for channel_name, update in task.result().items():
                    writes[channel_name].append(update)
            for channel_name, updates in writes.items():
                channels[channel_name].update(updates)

            # Compute what runs next from the now-consistent post-barrier state: plain node names
            # (deduped) and/or fresh `Send`s (one instance each, ordered).
            post = _snapshot(channels)
            if emit is not None:
                await emit(VALUES, None, step, {"state": post})
            next_active: set[str] = set()
            next_sends: list[_Send] = []
            for unit in units:
                for succ in unit.proc.successors(post):
                    if isinstance(succ, Send):
                        next_sends.append((succ.node, succ.payload()))
                    else:  # `successors` already dropped END; a plain name activates once
                        next_active.add(succ)
            step += 1

            # The barrier is the checkpoint boundary: durable state is consistent here, and
            # (next_active, next_sends) record exactly what to resume with.
            if backend is not None:
                last_checkpoint_id = save_checkpoint(
                    backend, thread_id, last_checkpoint_id, step, channels, next_active, next_sends
                )
            active = next_active
            pending_sends = next_sends

    return last_checkpoint_id
