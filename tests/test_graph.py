"""End-to-end graph execution: pipelines, parallel fan-out/fan-in, cycles, and validation."""

from __future__ import annotations

from operator import add
from typing import Annotated

import pytest

from tensorsketch import END, START, Context, Graph, Hole, Node, Reducer, Schema
from tensorsketch.core.errors import GraphError, GraphRecursionError

# --------------------------------------------------------------------------------------------
# Linear pipeline
# --------------------------------------------------------------------------------------------


class PipeState(Schema):
    text: str
    upper: str = ""
    excited: str = ""


class Upper(Node):
    class In(Schema):
        text: str

    class Out(Schema):
        upper: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(upper=inp.text.upper())


class Excite(Node):
    class In(Schema):
        upper: str

    class Out(Schema):
        excited: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(excited=inp.upper + "!")


async def test_linear_pipeline() -> None:
    app = (
        Graph(PipeState)
        .add(Upper)
        .add(Excite)
        .edge(START, "Upper")
        .edge("Upper", "Excite")
        .edge("Excite", END)
    ).compile()
    out = await app.invoke({"text": "hi"})
    assert out.upper == "HI"
    assert out.excited == "HI!"


# --------------------------------------------------------------------------------------------
# Parallel fan-out / fan-in through a reducer channel
# --------------------------------------------------------------------------------------------


class FanState(Schema):
    seed: int
    parts: Annotated[list[int], Reducer(add)] = []
    total: int = 0


class Seeded(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        seed: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(seed=inp.seed)


class PlusOne(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        parts: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(parts=[inp.seed + 1])


class PlusTwo(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        parts: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(parts=[inp.seed + 2])


class Sum(Node):
    class In(Schema):
        parts: list[int]

    class Out(Schema):
        total: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(total=sum(inp.parts))


async def test_parallel_fanout_fanin() -> None:
    app = (
        Graph(FanState)
        .add(Seeded)
        .add(PlusOne)
        .add(PlusTwo)
        .add(Sum)
        .edge(START, "Seeded")
        .edge("Seeded", "PlusOne")
        .edge("Seeded", "PlusTwo")
        .edge("PlusOne", "Sum")
        .edge("PlusTwo", "Sum")
        .edge("Sum", END)
    ).compile()
    out = await app.invoke({"seed": 10})
    assert sorted(out.parts) == [11, 12]  # both branches ran on the same snapshot
    assert out.total == 23  # merged by the reducer before Sum ran


# --------------------------------------------------------------------------------------------
# Guarded loop (a cycle) via a conditional edge
# --------------------------------------------------------------------------------------------


class LoopState(Schema):
    count: int = 0
    limit: int = 3
    log: Annotated[list[str], Reducer(add)] = []


class Tick(Node):
    class In(Schema):
        count: int

    class Out(Schema):
        count: int
        log: list[str]

    async def run(self, ctx: Context, inp: In) -> Out:
        nxt = inp.count + 1
        return self.Out(count=nxt, log=[f"tick@{ctx.superstep}->{nxt}"])


def _loop_route(state: LoopState) -> str:
    return END if state.count >= state.limit else "Tick"


async def test_guarded_loop_cycles_until_condition() -> None:
    app = (
        Graph(LoopState).add(Tick).edge(START, "Tick").conditional("Tick", _loop_route)
    ).compile()
    out = await app.invoke({"limit": 3})
    assert out.count == 3
    assert out.log == ["tick@0->1", "tick@1->2", "tick@2->3"]


async def test_recursion_limit_trips_on_unguarded_loop() -> None:
    app = (
        Graph(LoopState)
        .add(Tick)
        .edge(START, "Tick")
        .conditional("Tick", lambda _state: "Tick")  # never exits
    ).compile()
    with pytest.raises(GraphRecursionError):
        await app.invoke({}, max_steps=5)


# --------------------------------------------------------------------------------------------
# Holes and error surfacing
# --------------------------------------------------------------------------------------------


class Needy(Node):
    class In(Schema):
        text: str

    class Out(Schema):
        upper: str

    async def run(self, ctx: Context, inp: In) -> Out:
        raise Hole("uppercase the text using str.upper")


async def test_hole_propagates_when_run() -> None:
    app = Graph(PipeState).add(Needy).edge(START, "Needy").compile()
    with pytest.raises(Hole):
        await app.invoke({"text": "x"})


# --------------------------------------------------------------------------------------------
# Structural validation (caught at compile)
# --------------------------------------------------------------------------------------------


def test_missing_entry_is_rejected() -> None:
    with pytest.raises(GraphError, match="no entry"):
        Graph(PipeState).add(Upper).compile()


def test_edge_to_unknown_node_is_rejected() -> None:
    with pytest.raises(GraphError, match="not a node"):
        Graph(PipeState).add(Upper).edge(START, "Upper").edge("Upper", "Ghost").compile()


def test_port_not_in_state_is_rejected() -> None:
    class Rogue(Node):
        class In(Schema):
            missing: str

        class Out(Schema):
            upper: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(upper="")

    with pytest.raises(GraphError, match="does not exist"):
        Graph(PipeState).add(Rogue).edge(START, "Rogue").compile()


def test_port_type_mismatch_is_rejected() -> None:
    class Mismatch(Node):
        class In(Schema):
            upper: int  # state 'upper' is str

        class Out(Schema):
            excited: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(excited="")

    with pytest.raises(GraphError, match="holds"):
        Graph(PipeState).add(Mismatch).edge(START, "Mismatch").compile()


def test_both_static_and_conditional_edges_is_rejected() -> None:
    with pytest.raises(GraphError, match="both static edges and a conditional"):
        (
            Graph(LoopState)
            .add(Tick)
            .edge(START, "Tick")
            .edge("Tick", END)
            .conditional("Tick", _loop_route)
            .compile()
        )


async def test_duplicate_node_name_is_rejected() -> None:
    with pytest.raises(GraphError, match="already in the graph"):
        Graph(PipeState).add(Upper).add(Upper)


async def test_unknown_input_field_is_rejected() -> None:
    app = Graph(PipeState).add(Upper).edge(START, "Upper").compile()
    with pytest.raises(GraphError, match="not a field"):
        await app.invoke({"nope": 1})
