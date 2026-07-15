"""Composition patterns: map, parallel, subgraph, and structured-output repair."""

from __future__ import annotations

import pytest

from tensorsketch import (
    END,
    START,
    Context,
    FakeProvider,
    Graph,
    InMemoryBackend,
    Node,
    Schema,
    gather_map,
    generate_structured,
    parallel,
    run_subgraph,
)
from tensorsketch.messages import assistant

# --------------------------------------------------------------------------------------------
# gather_map
# --------------------------------------------------------------------------------------------


class MapState(Schema):
    items: list[int]
    doubled: list[int] = []


class Doubler(Node):
    class In(Schema):
        items: list[int]

    class Out(Schema):
        doubled: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        async def double(x: int) -> int:
            return x * 2

        return self.Out(doubled=await gather_map(ctx, inp.items, double))


async def test_gather_map_preserves_order() -> None:
    app = Graph(MapState).add(Doubler).edge(START, "Doubler").edge("Doubler", END).compile()
    out = await app.invoke({"items": [1, 2, 3, 4]})
    assert out.doubled == [2, 4, 6, 8]


async def test_gather_map_is_durable_exactly_once() -> None:
    calls = {"n": 0}
    crashed = {"yet": False}

    class S(Schema):
        items: list[int]
        total: int = 0

    class Mapper(Node):
        class In(Schema):
            items: list[int]

        class Out(Schema):
            total: int

        async def run(self, ctx: Context, inp: In) -> Out:
            async def effect(x: int) -> int:
                calls["n"] += 1
                return x * 10

            results = await gather_map(ctx, inp.items, effect)
            if not crashed["yet"]:
                crashed["yet"] = True
                raise RuntimeError("crash after the map completed")
            return self.Out(total=sum(results))

    backend = InMemoryBackend()
    app = Graph(S).add(Mapper).edge(START, "Mapper").edge("Mapper", END).compile()

    with pytest.raises(RuntimeError):
        await app.invoke({"items": [1, 2, 3]}, thread_id="m", backend=backend)
    assert calls["n"] == 3  # each item ran once, before the crash

    out = await app.invoke(thread_id="m", backend=backend)  # resume
    assert out.total == 60
    assert calls["n"] == 3  # items replayed from the journal, not re-run


# --------------------------------------------------------------------------------------------
# parallel
# --------------------------------------------------------------------------------------------


async def test_parallel_runs_and_orders_results() -> None:
    class S(Schema):
        x: int
        a: int = 0
        b: int = 0

    class Par(Node):
        class In(Schema):
            x: int

        class Out(Schema):
            a: int
            b: int

        async def run(self, ctx: Context, inp: In) -> Out:
            async def plus_one() -> int:
                return inp.x + 1

            async def plus_two() -> int:
                return inp.x + 2

            first, second = await parallel(ctx, plus_one, plus_two)
            return self.Out(a=first, b=second)

    app = Graph(S).add(Par).edge(START, "Par").edge("Par", END).compile()
    out = await app.invoke({"x": 10})
    assert out.a == 11
    assert out.b == 12


# --------------------------------------------------------------------------------------------
# run_subgraph
# --------------------------------------------------------------------------------------------


class InnerState(Schema):
    text: str
    up: str = ""


class Up(Node):
    class In(Schema):
        text: str

    class Out(Schema):
        up: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(up=inp.text.upper())


async def test_run_subgraph_composes_graphs() -> None:
    inner = Graph(InnerState).add(Up).edge(START, "Up").edge("Up", END).compile()

    class OuterState(Schema):
        word: str
        shout: str = ""

    class Caller(Node):
        class In(Schema):
            word: str

        class Out(Schema):
            shout: str

        async def run(self, ctx: Context, inp: In) -> Out:
            result = await run_subgraph(inner, {"text": inp.word}, ctx=ctx)
            return self.Out(shout=result.up + "!")

    outer = Graph(OuterState).add(Caller).edge(START, "Caller").edge("Caller", END).compile()
    out = await outer.invoke({"word": "hi"})
    assert out.shout == "HI!"


# --------------------------------------------------------------------------------------------
# validate-and-repair
# --------------------------------------------------------------------------------------------


class Box(Schema):
    x: int


async def test_structured_output_repairs_on_bad_reply() -> None:
    provider = FakeProvider([assistant("not json at all"), assistant('{"x": 7}')])
    box = await generate_structured(provider, Box, "give me x")
    assert box.x == 7
    assert len(provider.calls) == 2  # one bad attempt, one repair


async def test_structured_output_gives_up_after_max_repairs() -> None:
    provider = FakeProvider([assistant("nope"), assistant("still not json")])
    with pytest.raises(ValueError, match="failed after"):
        await generate_structured(provider, Box, "x", max_repairs=1)
