"""Graph-level dynamic fan-out (`Send`) and the `loop` / `router` builder sugar.

A router returns `Send`s; the engine spawns one task per Send (each its own superstep unit, with
its own payload) and they merge at the barrier via a reducer channel — the map/reduce shape at the
*graph* level. Fan-out is durable: each instance journals its effects under a distinct key.
"""

from __future__ import annotations

from operator import add
from typing import Annotated

import pytest

from tensorsketch import (
    END,
    START,
    Context,
    Graph,
    InMemoryBackend,
    Node,
    Reducer,
    Schema,
    Send,
)

# --------------------------------------------------------------------------------------------
# Map / reduce: a router fans out to one worker per item; workers merge into a reducer channel
# --------------------------------------------------------------------------------------------


class MapState(Schema):
    items: list[int] = []
    item: int = 0  # per-worker input slot — a Send payload overrides it per instance
    doubled: Annotated[list[int], Reducer(add)] = []
    total: int = 0


class Split(Node):
    class In(Schema):
        items: list[int]

    class Out(Schema):
        items: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(items=inp.items)


class Work(Node):
    class In(Schema):
        item: int

    class Out(Schema):
        doubled: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(doubled=[inp.item * 2])


class Collect(Node):
    class In(Schema):
        doubled: list[int]

    class Out(Schema):
        total: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(total=sum(inp.doubled))


def _map_reduce() -> Graph[MapState]:
    return (
        Graph(MapState)
        .add(Split)
        .add(Work)
        .add(Collect)
        .edge(START, "Split")
        .router("Split", lambda s: [Send("Work", {"item": i}) for i in s.items])
        .edge("Work", "Collect")
        .edge("Collect", END)
    )


async def test_dynamic_fanout_map_reduce() -> None:
    out = await _map_reduce().compile().invoke({"items": [1, 2, 3, 4]})
    assert sorted(out.doubled) == [2, 4, 6, 8]  # one worker instance per item, merged at barrier
    assert out.total == 20  # the collector read the merged reducer channel


async def test_fanout_scales_to_one_and_zero_items() -> None:
    one = await _map_reduce().compile().invoke({"items": [7]})
    assert one.total == 14
    none = await _map_reduce().compile().invoke({"items": []})
    assert none.total == 0  # no workers scheduled; the graph still settles


async def test_send_accepts_a_schema_payload() -> None:
    graph = (
        Graph(MapState)
        .add(Split)
        .add(Work)
        .add(Collect)
        .edge(START, "Split")
        .router("Split", lambda s: [Send("Work", Work.In(item=i)) for i in s.items])
        .edge("Work", "Collect")
        .edge("Collect", END)
    )
    out = await graph.compile().invoke({"items": [5, 6]})
    assert out.total == 22  # a Schema payload works just like a dict


async def test_send_to_unknown_node_raises() -> None:
    graph = (
        Graph(MapState)
        .add(Split)
        .add(Work)
        .edge(START, "Split")
        .router("Split", lambda s: [Send("ghost", {"item": 1})])
        .edge("Work", END)
    )
    with pytest.raises(Exception, match="Send target 'ghost'"):
        await graph.compile().invoke({"items": [1]})


# --------------------------------------------------------------------------------------------
# Durability of fan-out
# --------------------------------------------------------------------------------------------


async def test_fanout_instances_get_distinct_journal_keys() -> None:
    """The core per-instance guarantee: two instances of one node in one superstep journal under
    *distinct* keys, so each replays its own result — not a sibling's."""
    backend = InMemoryBackend()

    async def double(value: int) -> int:
        return value * 2

    async def step_for(instance: str, value: int) -> int:
        ctx = Context(
            run_id="r", thread_id="t", superstep=1, node="Work", backend=backend, instance=instance
        )
        return await ctx.step("double", lambda: double(value))

    assert (await step_for("0", 10), await step_for("1", 20)) == (10 * 2, 20 * 2)
    # replay: the recorded value comes back per instance (the 999s are never computed)
    assert (await step_for("0", 999), await step_for("1", 999)) == (20, 40)


async def test_fanout_worker_survives_crash_and_replays() -> None:
    """A Send instance that crashes after committing its effect replays it on resume."""
    calls = {"effect": 0, "attempts": 0}

    class Flaky(Node):
        class In(Schema):
            item: int

        class Out(Schema):
            total: int

        async def run(self, ctx: Context, inp: In) -> Out:
            async def effect() -> int:
                calls["effect"] += 1
                return inp.item * 10

            value = await ctx.step("compute", effect)
            calls["attempts"] += 1
            if calls["attempts"] == 1:
                raise RuntimeError("simulated crash after the fan-out effect committed")
            return self.Out(total=value)

    graph = (
        Graph(MapState)
        .add(Split)
        .add(Flaky)
        .edge(START, "Split")
        .router("Split", lambda s: [Send("Flaky", {"item": s.items[0]})])
        .edge("Flaky", END)
    )
    app = graph.compile()
    backend = InMemoryBackend()

    with pytest.raises(RuntimeError):
        await app.invoke({"items": [3]}, thread_id="t", backend=backend)
    assert calls["effect"] == 1  # the worker's effect ran once, journaled before the crash

    out = await app.invoke(thread_id="t", backend=backend)  # resume re-schedules the same Send
    assert out.total == 30
    assert calls["effect"] == 1  # exactly once — replayed from the journal under its instance key


# --------------------------------------------------------------------------------------------
# loop / router builder sugar
# --------------------------------------------------------------------------------------------


class LoopState(Schema):
    count: int = 0
    limit: int = 3


class Tick(Node):
    class In(Schema):
        count: int

    class Out(Schema):
        count: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(count=inp.count + 1)


async def test_loop_repeats_until_predicate() -> None:
    app = (
        Graph(LoopState)
        .add(Tick)
        .edge(START, "Tick")
        .loop("Tick", until=lambda s: s.count >= s.limit)
    ).compile()
    out = await app.invoke({"count": 0, "limit": 3})
    assert out.count == 3  # Tick looped back to itself until the predicate held, then stopped


class RouteState(Schema):
    n: int = 0
    tag: str = ""


class Origin(Node):
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
        tag: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(tag="even")


class Odd(Node):
    class In(Schema):
        n: int

    class Out(Schema):
        tag: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(tag="odd")


async def test_router_with_mapping_picks_a_branch() -> None:
    app = (
        Graph(RouteState)
        .add(Origin)
        .add(Even)
        .add(Odd)
        .edge(START, "Origin")
        .router("Origin", lambda s: "e" if s.n % 2 == 0 else "o", {"e": "Even", "o": "Odd"})
        .edge("Even", END)
        .edge("Odd", END)
    ).compile()
    assert (await app.invoke({"n": 4})).tag == "even"
    assert (await app.invoke({"n": 7})).tag == "odd"
