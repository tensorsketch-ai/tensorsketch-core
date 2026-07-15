"""Streaming: live namespaced events, custom emits, resumable replay, and error propagation."""

from __future__ import annotations

import pytest

from tensorsketch import END, START, Context, Event, Graph, InMemoryBackend, Node, Schema


class S(Schema):
    text: str
    out: str = ""


class Work(Node):
    class In(Schema):
        text: str

    class Out(Schema):
        out: str

    async def run(self, ctx: Context, inp: In) -> Out:
        await ctx.emit("progress", {"msg": "starting"})
        await ctx.emit("progress", {"msg": "done"})
        return self.Out(out=inp.text.upper())


async def test_stream_emits_ordered_namespaced_events() -> None:
    app = Graph(S).add(Work).edge(START, "Work").edge("Work", END).compile()
    events: list[Event] = [ev async for ev in app.stream({"text": "hi"})]

    types = [e.type for e in events]
    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    assert "node_start" in types
    assert "node_end" in types
    assert "values" in types

    # Cursor is monotonic and contiguous from 0.
    assert [e.seq for e in events] == list(range(len(events)))

    # node_end carries the node's writes; values carries merged state.
    node_end = next(e for e in events if e.type == "node_end")
    assert node_end.node == "Work"
    assert node_end.data["writes"] == {"out": "HI"}
    values = next(e for e in events if e.type == "values")
    assert values.data["state"]["out"] == "HI"


async def test_custom_emit_is_namespaced() -> None:
    app = Graph(S).add(Work).edge(START, "Work").edge("Work", END).compile()
    events = [ev async for ev in app.stream({"text": "hi"})]
    progress = [e for e in events if e.type == "progress"]
    assert [e.data["msg"] for e in progress] == ["starting", "done"]
    assert all(e.node == "Work" for e in progress)  # tagged with the emitting node


async def test_stream_persists_events_for_replay() -> None:
    backend = InMemoryBackend()
    app = Graph(S).add(Work).edge(START, "Work").edge("Work", END).compile()

    live = [ev async for ev in app.stream({"text": "hi"}, thread_id="t", backend=backend)]

    # Full replay reproduces the whole stream.
    replayed = [ev async for ev in app.replay("t", backend)]
    assert [e.seq for e in replayed] == [e.seq for e in live]

    # Replay from a cursor returns only the tail.
    tail = [ev async for ev in app.replay("t", backend, since=4)]
    assert tail and all(e.seq >= 4 for e in tail)
    assert [e.seq for e in tail] == [e.seq for e in live if e.seq >= 4]


async def test_stream_surfaces_run_errors() -> None:
    class Boom(Node):
        class In(Schema):
            text: str

        class Out(Schema):
            out: str

        async def run(self, ctx: Context, inp: In) -> Out:
            raise RuntimeError("boom")

    app = Graph(S).add(Boom).edge(START, "Boom").edge("Boom", END).compile()

    seen: list[Event] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for ev in app.stream({"text": "hi"}):
            seen.append(ev)

    # Events emitted before the failure are still delivered.
    assert any(e.type == "run_start" for e in seen)
    assert any(e.type == "node_start" for e in seen)
    assert not any(e.type == "run_end" for e in seen)
