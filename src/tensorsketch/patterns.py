"""Composition patterns for node and agent bodies: map, parallel, and subgraph.

These are the classic control-flow shapes — fan a collection out and reduce it, run several
things at once, call one graph from inside another — expressed as helpers you call inside a
node's `run`. Each wraps its work in `ctx.step`, so they inherit TensorSketch's durability for free:
on
resume, work that already finished is replayed from the journal instead of re-run. Because a
concurrent map can't rely on call order, each unit gets an **explicit, deterministic key**, so
exactly-once holds even though the pieces run in parallel.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

from .core.context import Context
from .core.graph import CompiledGraph
from .core.schema import Schema

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S", bound=Schema)


def _key(ctx: Context, label: str, index: int) -> str:
    # Stable across replays and unique within a run, independent of completion order.
    return f"{ctx.node}:{ctx.superstep}:{label}:{index}"


async def gather_map(
    ctx: Context,
    items: Iterable[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    label: str = "map",
    max_concurrency: int | None = None,
) -> list[R]:
    """Run `fn` over every item concurrently and return the results in order.

    Durable: each item's call is journaled under a deterministic key, so a crash mid-map resumes
    without re-running the items that already completed. `max_concurrency` caps how many run at
    once (default: unbounded).
    """
    work = list(items)
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def run_one(index: int, item: T) -> R:
        async def call() -> R:
            return await fn(item)

        key = _key(ctx, label, index)
        if semaphore is not None:
            async with semaphore:
                return await ctx.step(f"{label}:{index}", call, idempotency_key=key)
        return await ctx.step(f"{label}:{index}", call, idempotency_key=key)

    return list(await asyncio.gather(*(run_one(i, item) for i, item in enumerate(work))))


async def parallel(
    ctx: Context,
    *fns: Callable[[], Awaitable[Any]],
    label: str = "parallel",
) -> tuple[Any, ...]:
    """Run several independent async calls at once and return their results, in argument order.

    Durable, like `gather_map`. Handy for firing off independent tool/model calls together.
    """

    async def run_one(index: int, fn: Callable[[], Awaitable[Any]]) -> Any:
        return await ctx.step(f"{label}:{index}", fn, idempotency_key=_key(ctx, label, index))

    return tuple(await asyncio.gather(*(run_one(i, fn) for i, fn in enumerate(fns))))


async def run_subgraph(
    graph: CompiledGraph[S],
    input: Schema | dict[str, Any],
    *,
    ctx: Context | None = None,
    label: str = "subgraph",
) -> S:
    """Run a compiled graph from inside a node and return its final state.

    Pass `ctx` to journal the whole subgraph call as one durable step (its result is replayed on
    resume rather than re-executed). This is the composition primitive: build small graphs and
    call them from larger ones.
    """
    if ctx is not None:

        async def call() -> S:
            return await graph.invoke(input)

        return await ctx.step(label, call)
    return await graph.invoke(input)
