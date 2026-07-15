"""Durability: checkpoints (resume/fork) and the effect journal (exactly-once side effects).

Each test runs against both backends to keep them behavior-identical.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tensorsketch import (
    END,
    START,
    Backend,
    Context,
    Graph,
    InMemoryBackend,
    Node,
    Schema,
    Send,
    SqliteBackend,
)
from tensorsketch.core.errors import GraphRecursionError

BACKENDS: list[Callable[[], Backend]] = [InMemoryBackend, lambda: SqliteBackend(":memory:")]

# Run the whole durability suite against RedisBackend too (via fakeredis, so no server needed) —
# proof the database connectors resume and journal exactly-once through the real engine.
try:
    import fakeredis

    from tensorsketch.runtime.backends import RedisBackend

    BACKENDS.append(lambda: RedisBackend(client=fakeredis.FakeStrictRedis()))
except ImportError:  # pragma: no cover - fakeredis is a dev dependency
    pass


@pytest.mark.parametrize("make_backend", BACKENDS)
async def test_effect_runs_once_then_replays_on_resume(make_backend: Callable[[], Backend]) -> None:
    calls = {"n": 0}

    class S(Schema):
        x: int = 0
        y: int = 0

    class Work(Node):
        class In(Schema):
            x: int

        class Out(Schema):
            y: int

        async def run(self, ctx: Context, inp: In) -> Out:
            async def effect() -> int:
                calls["n"] += 1
                return inp.x * 10

            return self.Out(y=await ctx.step("compute", effect))

    backend = make_backend()
    app = Graph(S).add(Work).edge(START, "Work").edge("Work", END).compile()

    out = await app.invoke({"x": 4}, thread_id="t", backend=backend)
    assert out.y == 40
    assert calls["n"] == 1

    # Resuming a completed run returns the same state without re-running the effect.
    again = await app.invoke(thread_id="t", backend=backend)
    assert again.y == 40
    assert calls["n"] == 1


@pytest.mark.parametrize("make_backend", BACKENDS)
async def test_crash_harness_effect_not_reexecuted(make_backend: Callable[[], Backend]) -> None:
    """The core durability guarantee: a mid-run crash + resume never repeats a side effect."""
    calls = {"effect": 0, "attempts": 0}

    class S(Schema):
        x: int = 0
        y: int = 0

    class Flaky(Node):
        class In(Schema):
            x: int

        class Out(Schema):
            y: int

        async def run(self, ctx: Context, inp: In) -> Out:
            async def effect() -> int:
                calls["effect"] += 1
                return inp.x * 10

            value = await ctx.step("compute", effect)  # effect is journaled the moment it runs
            calls["attempts"] += 1
            if calls["attempts"] == 1:
                raise RuntimeError("simulated crash after the effect committed")
            return self.Out(y=value)

    backend = make_backend()
    app = Graph(S).add(Flaky).edge(START, "Flaky").edge("Flaky", END).compile()

    with pytest.raises(RuntimeError):
        await app.invoke({"x": 3}, thread_id="t", backend=backend)
    assert calls["effect"] == 1  # the effect ran once, before the crash

    out = await app.invoke(thread_id="t", backend=backend)  # resume
    assert out.y == 30
    assert calls["effect"] == 1  # exactly once — the resume replayed it from the journal


@pytest.mark.parametrize("make_backend", BACKENDS)
async def test_pending_fanout_resumes_across_backends(make_backend: Callable[[], Backend]) -> None:
    """A crash mid-fan-out resumes: the pending `Send` (payload and all) survives the checkpoint
    and the worker's effect replays exactly once — so `sends` round-trip through each backend."""
    calls = {"effect": 0, "attempts": 0}

    class S(Schema):
        seed: int = 0
        item: int = 0
        total: int = 0

    class Src(Node):
        class In(Schema):
            seed: int

        class Out(Schema):
            seed: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(seed=inp.seed)

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
                raise RuntimeError("crash while a fan-out instance was in flight")
            return self.Out(total=value)

    backend = make_backend()
    app = (
        Graph(S)
        .add(Src)
        .add(Flaky)
        .edge(START, "Src")
        .router("Src", lambda s: [Send("Flaky", {"item": s.seed})])
        .edge("Flaky", END)
    ).compile()

    with pytest.raises(RuntimeError):
        await app.invoke({"seed": 4}, thread_id="t", backend=backend)
    assert calls["effect"] == 1

    out = await app.invoke(thread_id="t", backend=backend)  # resume re-schedules the pending Send
    assert out.total == 40  # the payload (item=4) came back from the checkpoint
    assert calls["effect"] == 1  # exactly once


@pytest.mark.parametrize("make_backend", BACKENDS)
async def test_resume_continues_after_recursion_limit(make_backend: Callable[[], Backend]) -> None:
    """A run stopped by the step budget resumes from its last checkpoint and finishes."""

    class S(Schema):
        count: int = 0
        limit: int = 4

    class Tick(Node):
        class In(Schema):
            count: int

        class Out(Schema):
            count: int

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(count=inp.count + 1)

    def route(state: S) -> str:
        return END if state.count >= state.limit else "Tick"

    backend = make_backend()
    app = Graph(S).add(Tick).edge(START, "Tick").conditional("Tick", route).compile()

    with pytest.raises(GraphRecursionError):
        await app.invoke({"limit": 4}, thread_id="t", backend=backend, max_steps=2)

    midway = app.get_state("t", backend)
    assert midway is not None
    assert midway.count < 4  # stopped partway through the loop

    finished = await app.invoke(thread_id="t", backend=backend, max_steps=10)
    assert finished.count == 4


@pytest.mark.parametrize("make_backend", BACKENDS)
async def test_history_and_fork(make_backend: Callable[[], Backend]) -> None:
    class S(Schema):
        text: str = ""
        out: str = ""

    class Echo(Node):
        class In(Schema):
            text: str

        class Out(Schema):
            out: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(out=inp.text.upper())

    backend = make_backend()
    app = Graph(S).add(Echo).edge(START, "Echo").edge("Echo", END).compile()

    out = await app.invoke({"text": "hi"}, thread_id="t", backend=backend)
    assert out.out == "HI"

    history = app.get_history("t", backend)
    assert len(history) >= 2  # an initial checkpoint plus one per barrier

    # Fork from the initial checkpoint with a different input — a new branch, fresh journal.
    initial = history[0]
    forked = await app.fork(backend, "t", initial.id, "t-fork", {"text": "bye"})
    assert forked.out == "BYE"

    # The original thread is untouched by the fork.
    original = app.get_state("t", backend)
    assert original is not None
    assert original.out == "HI"


async def test_backend_requires_thread_id() -> None:
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
    with pytest.raises(Exception, match="thread_id"):
        await app.invoke({"x": 1}, backend=InMemoryBackend())
