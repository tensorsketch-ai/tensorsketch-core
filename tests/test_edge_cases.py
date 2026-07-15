"""Edge cases and hardening: routing shapes, reducers, races, errors, and resume semantics."""

from __future__ import annotations

from operator import add
from typing import Annotated

import pytest
from pydantic import ValidationError

from tensorsketch import (
    END,
    START,
    Context,
    Graph,
    InMemoryBackend,
    InvalidUpdateError,
    Node,
    NodeError,
    Reducer,
    Schema,
    Topic,
)
from tensorsketch.core.errors import GraphRecursionError

# --------------------------------------------------------------------------------------------
# Routing shapes
# --------------------------------------------------------------------------------------------


class FanState(Schema):
    seed: int
    parts: Annotated[list[int], Reducer(add)] = []
    total: int = 0


class Seed(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        seed: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(seed=inp.seed)


class Plus(Node):
    """A configurable adder — different instances add different amounts to `parts`."""

    def __init__(self, amount: int) -> None:
        self.amount = amount

    class In(Schema):
        seed: int

    class Out(Schema):
        parts: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(parts=[inp.seed + self.amount])


class SumParts(Node):
    class In(Schema):
        parts: list[int]

    class Out(Schema):
        total: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(total=sum(inp.parts))


async def test_router_returning_list_fans_out() -> None:
    app = (
        Graph(FanState)
        .add(Seed)
        .add(Plus(1), name="A")
        .add(Plus(2), name="B")
        .add(Plus(3), name="C")
        .add(SumParts)
        .edge(START, "Seed")
        .conditional("Seed", lambda _s: ["A", "B", "C"])  # dynamic parallel fan-out
        .edge("A", "SumParts")
        .edge("B", "SumParts")
        .edge("C", "SumParts")
        .edge("SumParts", END)
    ).compile()
    out = await app.invoke({"seed": 10})
    assert sorted(out.parts) == [11, 12, 13]  # three-way fan-in through the reducer
    assert out.total == 36


async def test_conditional_mapping_resolves_targets() -> None:
    class S(Schema):
        n: int
        label: str = ""

    class Pick(Node):
        class In(Schema):
            n: int

        class Out(Schema):
            n: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(n=inp.n)

    class Even(Node):
        class In(Schema):
            n: int

        class Out(Schema):
            label: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(label="even")

    class Odd(Node):
        class In(Schema):
            n: int

        class Out(Schema):
            label: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(label="odd")

    app = (
        Graph(S)
        .add(Pick)
        .add(Even)
        .add(Odd)
        .edge(START, "Pick")
        .conditional(
            "Pick",
            lambda s: "e" if s.n % 2 == 0 else "o",
            {"e": "Even", "o": "Odd"},
        )
        .edge("Even", END)
        .edge("Odd", END)
    ).compile()
    assert (await app.invoke({"n": 4})).label == "even"
    assert (await app.invoke({"n": 7})).label == "odd"


async def test_conditional_can_route_straight_to_end() -> None:
    class S(Schema):
        n: int
        seen: bool = False

    class Gate(Node):
        class In(Schema):
            n: int

        class Out(Schema):
            seen: bool

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(seen=True)

    app = (
        Graph(S)
        .add(Gate)
        .edge(START, "Gate")
        .conditional("Gate", lambda _s: END)  # stop immediately after Gate
    ).compile()
    out = await app.invoke({"n": 1})
    assert out.seen is True


# --------------------------------------------------------------------------------------------
# Reducers end-to-end
# --------------------------------------------------------------------------------------------


async def test_topic_channel_accumulates_across_nodes() -> None:
    class S(Schema):
        events: Annotated[list[str], Topic()] = []
        done: bool = False

    class First(Node):
        class In(Schema):
            events: list[str]

        class Out(Schema):
            events: list[str]

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(events=["first"])

    class Second(Node):
        class In(Schema):
            events: list[str]

        class Out(Schema):
            events: list[str]
            done: bool

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(events=["second"], done=True)

    app = (
        Graph(S)
        .add(First)
        .add(Second)
        .edge(START, "First")
        .edge("First", "Second")
        .edge("Second", END)
    ).compile()
    out = await app.invoke({})
    assert out.events == ["first", "second"]
    assert out.done is True


async def test_reducer_sums_across_supersteps() -> None:
    class S(Schema):
        count: int = 0
        limit: int = 4
        total: Annotated[int, Reducer(add)] = 0

    class Tick(Node):
        class In(Schema):
            count: int

        class Out(Schema):
            count: int
            total: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(count=inp.count + 1, total=inp.count + 1)

    def route(s: S) -> str:
        return END if s.count >= s.limit else "Tick"

    app = Graph(S).add(Tick).edge(START, "Tick").conditional("Tick", route).compile()
    out = await app.invoke({"limit": 4})
    assert out.count == 4
    assert out.total == 1 + 2 + 3 + 4  # accumulated by the reducer, not overwritten


# --------------------------------------------------------------------------------------------
# Races and errors
# --------------------------------------------------------------------------------------------


async def test_concurrent_write_to_lastvalue_is_rejected() -> None:
    class S(Schema):
        seed: int = 0
        result: str = ""  # a plain LastValue — two writers in one step is illegal

    class Fork(Node):
        class In(Schema):
            seed: int

        class Out(Schema):
            seed: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(seed=inp.seed)

    class WriteA(Node):
        class In(Schema):
            seed: int

        class Out(Schema):
            result: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(result="a")

    class WriteB(Node):
        class In(Schema):
            seed: int

        class Out(Schema):
            result: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(result="b")

    app = (
        Graph(S)
        .add(Fork)
        .add(WriteA)
        .add(WriteB)
        .edge(START, "Fork")
        .edge("Fork", "WriteA")
        .edge("Fork", "WriteB")
    ).compile()
    with pytest.raises(InvalidUpdateError):
        await app.invoke({"seed": 1})


async def test_missing_required_input_raises_node_error() -> None:
    class S(Schema):
        x: int  # required, no default
        y: int = 0

    class Double(Node):
        class In(Schema):
            x: int

        class Out(Schema):
            y: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(y=inp.x * 2)

    app = Graph(S).add(Double).edge(START, "Double").edge("Double", END).compile()
    with pytest.raises(NodeError, match="could not build its input"):
        await app.invoke({})  # x never provided → the node can't form its In


def test_schema_rejects_unknown_fields() -> None:
    class S(Schema):
        a: int

    with pytest.raises(ValidationError):
        S(a=1, bogus=2)  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------------
# Resume semantics
# --------------------------------------------------------------------------------------------


async def test_idempotency_key_dedupes_across_nodes() -> None:
    calls = {"n": 0}

    class S(Schema):
        first: int = 0
        second: int = 0

    async def effect() -> int:
        calls["n"] += 1
        return 7

    class One(Node):
        class In(Schema):
            first: int

        class Out(Schema):
            first: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(first=await ctx.step("e", effect, idempotency_key="shared"))

    class Two(Node):
        class In(Schema):
            first: int

        class Out(Schema):
            second: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(second=await ctx.step("e", effect, idempotency_key="shared"))

    app = (
        Graph(S).add(One).add(Two).edge(START, "One").edge("One", "Two").edge("Two", END)
    ).compile()
    out = await app.invoke({}, thread_id="t", backend=InMemoryBackend())
    assert out.first == 7
    assert out.second == 7
    assert calls["n"] == 1  # the shared idempotency key ran the effect once for both nodes


async def test_input_injected_on_resume_changes_behavior() -> None:
    class S(Schema):
        count: int = 0
        limit: int = 100

    class Tick(Node):
        class In(Schema):
            count: int

        class Out(Schema):
            count: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(count=inp.count + 1)

    def route(s: S) -> str:
        return END if s.count >= s.limit else "Tick"

    backend = InMemoryBackend()
    app = Graph(S).add(Tick).edge(START, "Tick").conditional("Tick", route).compile()

    with pytest.raises(GraphRecursionError):
        await app.invoke({"limit": 100}, thread_id="t", backend=backend, max_steps=2)

    # Resume with a lowered limit — the injected input makes the loop terminate promptly.
    finished = await app.invoke({"limit": 2}, thread_id="t", backend=backend, max_steps=50)
    assert finished.count < 10


def test_get_state_and_history_on_unknown_thread() -> None:
    class S(Schema):
        x: int = 0

    class Noop(Node):
        class In(Schema):
            x: int

        class Out(Schema):
            x: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(x=inp.x)

    app = Graph(S).add(Noop).edge(START, "Noop").edge("Noop", END).compile()
    backend = InMemoryBackend()
    assert app.get_state("nope", backend) is None
    assert app.get_history("nope", backend) == []
