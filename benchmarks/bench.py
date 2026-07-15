"""Indicative micro-benchmarks for the TensorSketch runtime.

These are *not* rigorous benchmarks — they measure the pure-Python reference runtime's overhead
so we can watch it as the framework grows and catch regressions. Node bodies here are trivial, so
the numbers reflect scheduler + channel + (optional) checkpoint cost, not real agent work.

Run:  uv run python benchmarks/bench.py
"""

from __future__ import annotations

import asyncio
import itertools
import statistics
import time
from collections.abc import Awaitable, Callable
from operator import add
from typing import Annotated

from tensorsketch import (
    END,
    START,
    CompiledGraph,
    Context,
    Graph,
    InMemoryBackend,
    Node,
    Reducer,
    Schema,
    SqliteBackend,
)


class Chain(Schema):
    n: int = 0


class Incr(Node):
    class In(Schema):
        n: int

    class Out(Schema):
        n: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(n=inp.n + 1)


def build_chain(length: int) -> CompiledGraph[Chain]:
    g: Graph[Chain] = Graph(Chain)
    for i in range(length):
        g.add(Incr, name=f"s{i}")
    g.edge(START, "s0")
    for i in range(length - 1):
        g.edge(f"s{i}", f"s{i + 1}")
    g.edge(f"s{length - 1}", END)
    return g.compile()


class Fan(Schema):
    seed: int = 0
    parts: Annotated[list[int], Reducer(add)] = []
    total: int = 0


class Source(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        seed: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(seed=inp.seed)


class Worker(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        parts: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(parts=[inp.seed + 1])


class Collect(Node):
    class In(Schema):
        parts: list[int]

    class Out(Schema):
        total: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(total=sum(inp.parts))


def build_fanout(width: int) -> CompiledGraph[Fan]:
    g: Graph[Fan] = Graph(Fan)
    g.add(Source).add(Collect)
    g.edge(START, "Source")
    for i in range(width):
        g.add(Worker, name=f"w{i}")
        g.edge("Source", f"w{i}")
        g.edge(f"w{i}", "Collect")
    g.edge("Collect", END)
    return g.compile()


async def _median_ms(run_once: Callable[[], Awaitable[object]], iters: int = 25) -> float:
    await run_once()  # warmup
    samples = [0.0] * iters
    for i in range(iters):
        start = time.perf_counter()
        await run_once()
        samples[i] = (time.perf_counter() - start) * 1000
    return statistics.median(samples)


async def bench_chain_row(length: int) -> tuple[float, float, float]:
    """Median ms for an N-superstep chain: no backend, in-memory, and SQLite."""
    compiled = build_chain(length)
    mem = InMemoryBackend()
    sql = SqliteBackend(":memory:")
    ids = itertools.count()
    budget = length + 5

    async def plain() -> object:
        return await compiled.invoke({"n": 0}, max_steps=budget)

    async def with_mem() -> object:
        return await compiled.invoke(
            {"n": 0}, thread_id=f"m{next(ids)}", backend=mem, max_steps=budget
        )

    async def with_sql() -> object:
        return await compiled.invoke(
            {"n": 0}, thread_id=f"s{next(ids)}", backend=sql, max_steps=budget
        )

    return await _median_ms(plain), await _median_ms(with_mem), await _median_ms(with_sql)


async def bench_fanout_row(width: int) -> float:
    """Median ms for a width-W parallel fan-out merged through a reducer."""
    compiled = build_fanout(width)

    async def run() -> object:
        return await compiled.invoke({"seed": 1})

    return await _median_ms(run)


async def main() -> None:
    print("TensorSketch runtime micro-benchmarks (median of 25 runs)\n")

    print("Sequential chain (N supersteps):")
    print(f"  {'N':>5}  {'no backend':>12}  {'in-memory':>12}  {'sqlite':>12}   µs/superstep")
    for length in (10, 50, 200):
        plain, mem, sql = await bench_chain_row(length)
        per = plain / length * 1000
        print(f"  {length:>5}  {plain:>10.3f}ms  {mem:>10.3f}ms  {sql:>10.3f}ms   {per:>10.1f}")

    print("\nParallel fan-out (width W in one superstep):")
    print(f"  {'W':>5}  {'time':>12}")
    for width in (10, 50, 200):
        print(f"  {width:>5}  {await bench_fanout_row(width):>10.3f}ms")


if __name__ == "__main__":
    asyncio.run(main())
