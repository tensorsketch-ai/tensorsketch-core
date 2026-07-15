"""Middleware — the extensibility seam around an agent's model and tool calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from tensorsketch import (
    FakeProvider,
    InMemoryBackend,
    Middleware,
    ObservabilityMiddleware,
    RetryMiddleware,
    create_agent,
    tool,
)
from tensorsketch.core.schema import Schema
from tensorsketch.messages import Message, ToolCall, assistant, system
from tensorsketch.providers.base import ChatProvider, Completion
from tensorsketch.tools import Tool


class FlakyProvider(ChatProvider):
    """Raises for the first `fails` calls, then returns `answer`. Counts every call."""

    def __init__(self, fails: int, answer: str = "ok") -> None:
        self.fails = fails
        self.answer = answer
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError("transient model error")
        return Completion(message=assistant(content=self.answer))


class Recorder(ChatProvider):
    """Records the messages it last saw, then answers `done`."""

    def __init__(self) -> None:
        self.seen: list[Message] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        self.seen = list(messages)
        return Completion(message=assistant(content="done"))


async def test_retry_recovers_from_a_flaky_model() -> None:
    provider = FlakyProvider(fails=2)
    agent = create_agent(provider, middleware=[RetryMiddleware(attempts=3)])
    out = await agent.invoke({"query": "hi"})
    assert out.output == "ok"
    assert provider.calls == 3  # two failures, then success


async def test_retry_reraises_when_exhausted() -> None:
    provider = FlakyProvider(fails=5)
    agent = create_agent(provider, middleware=[RetryMiddleware(attempts=2)])
    with pytest.raises(RuntimeError, match="transient"):
        await agent.invoke({"query": "hi"})
    assert provider.calls == 2


async def test_middleware_can_modify_the_request() -> None:
    class InjectSystem(Middleware):
        async def wrap_model(self, request: Any, call_next: Any) -> Any:
            request.messages.append(system("be terse"))
            return await call_next(request)

    provider = Recorder()
    agent = create_agent(provider, middleware=[InjectSystem()])
    await agent.invoke({"query": "hi"})
    assert any(m.role == "system" and "be terse" in m.content for m in provider.seen)


async def test_middleware_runs_outermost_first() -> None:
    order: list[str] = []

    class Tagging(Middleware):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        async def wrap_model(self, request: Any, call_next: Any) -> Any:
            order.append(f"{self.tag}:before")
            result = await call_next(request)
            order.append(f"{self.tag}:after")
            return result

    agent = create_agent(Recorder(), middleware=[Tagging("a"), Tagging("b")])
    await agent.invoke({"query": "hi"})
    assert order == ["a:before", "b:before", "b:after", "a:after"]


async def test_retry_wraps_tool_calls_too() -> None:
    calls = {"n": 0}

    @tool
    def flaky_double(x: int) -> int:
        """Double a number, but fail the first time."""
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient tool error")
        return x * 2

    provider = FakeProvider(
        [
            assistant(tool_calls=[ToolCall(id="c1", name="flaky_double", args={"x": 5})]),
            assistant(content="ten"),
        ]
    )
    agent = create_agent(provider, tools=[flaky_double], middleware=[RetryMiddleware(attempts=3)])
    out = await agent.invoke({"query": "double 5"})
    assert out.output == "ten"
    assert calls["n"] == 2  # failed once, retried, succeeded


async def test_retry_is_journaled_not_rerun_on_resume() -> None:
    """A retried call is inside ctx.step, so a resume replays the result — no new model calls."""
    provider = FlakyProvider(fails=1)
    backend = InMemoryBackend()
    agent = create_agent(provider, middleware=[RetryMiddleware(attempts=3)])

    out = await agent.invoke({"query": "hi"}, thread_id="t", backend=backend)
    assert out.output == "ok"
    assert provider.calls == 2  # one failure + one success, journaled

    again = await agent.invoke(thread_id="t", backend=backend)
    assert again.output == "ok"
    assert provider.calls == 2  # resume replayed the journal; the model was not called again


async def test_observability_middleware_emits_stream_events() -> None:
    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    provider = FakeProvider(
        [
            assistant(tool_calls=[ToolCall(id="c1", name="echo", args={"text": "hi"})]),
            assistant(content="done"),
        ]
    )
    agent = create_agent(provider, tools=[echo], middleware=[ObservabilityMiddleware()])

    seen: set[tuple[str, Any]] = set()
    async for event in agent.stream({"query": "x"}):
        if event.type in ("model_call", "tool_call"):
            seen.add((event.type, event.data.get("phase")))

    assert ("model_call", "start") in seen
    assert ("model_call", "end") in seen
    assert ("tool_call", "start") in seen
    assert ("tool_call", "end") in seen
