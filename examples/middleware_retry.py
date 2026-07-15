"""Middleware: retries, a guardrail, and observability around an agent's model/tool calls.

One uniform seam wraps every model and tool call. Here a flaky provider fails its first call;
`RetryMiddleware` transparently recovers, `ObservabilityMiddleware` emits timing events into the
stream, and a tiny custom middleware injects a guardrail instruction. All offline.

Run:  uv run python examples/middleware_retry.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from tensorsketch import (
    Middleware,
    ObservabilityMiddleware,
    RetryMiddleware,
    Schema,
    create_agent,
    tool,
)
from tensorsketch.messages import Message, ToolCall, assistant, system
from tensorsketch.providers.base import ChatProvider, Completion
from tensorsketch.tools import Tool


@tool
def weather(city: str) -> dict[str, object]:
    """Look up the (pretend) weather for a city."""
    return {"city": city, "tempC": 21, "sky": "clear"}


class FlakyProvider(ChatProvider):
    """Fails the very first call (as a rate limit would), then drives a tool call and answers."""

    def __init__(self) -> None:
        self.calls = 0
        self._script = [
            assistant(tool_calls=[ToolCall(id="c1", name="weather", args={"city": "Paris"})]),
            assistant(content="It's clear and about 21°C in Paris."),
        ]

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
        if self.calls == 1:
            raise RuntimeError("rate limited (503)")
        return Completion(message=self._script.pop(0))


class Guardrail(Middleware):
    """A custom middleware — inject a house rule before every model call."""

    async def wrap_model(self, request: Any, call_next: Any) -> Completion:
        request.messages.append(system("Answer in one short sentence."))
        return await call_next(request)


async def main() -> None:
    agent = create_agent(
        FlakyProvider(),
        tools=[weather],
        middleware=[RetryMiddleware(attempts=3), ObservabilityMiddleware(), Guardrail()],
    )

    print("event stream:")
    answer = ""
    async for event in agent.stream({"query": "what's the weather in Paris?"}):
        if event.type in ("retry", "model_call", "tool_call"):
            print(f"  {event.type:<11} {event.data}")
        elif event.type == "values":
            answer = event.data["state"].get("output") or answer

    print(f"\nanswer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
