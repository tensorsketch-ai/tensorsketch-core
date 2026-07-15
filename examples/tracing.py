"""Native tracing: run an agent, then read the span tree — timing, tokens, and cost.

TensorSketch traces itself through its own `Tracer` (no OpenTelemetry required). Pass an
`InMemoryTracer`
to `invoke`, and you get a tree of spans — run → node → model / tool calls — with duration, token
usage, and estimated cost on each. That's the raw material for evaluating latency and spend. This
example uses an offline provider that reports token usage so the cost column is populated.

Run:  uv run python examples/tracing.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from tensorsketch import InMemoryTracer, Schema, Usage, create_agent, tool
from tensorsketch.messages import Message, ToolCall, assistant
from tensorsketch.providers.base import ChatProvider, Completion
from tensorsketch.tools import Tool


@tool
def search(query: str) -> str:
    """Search the web (pretend)."""
    return f"top result for {query!r}"


class DemoProvider(ChatProvider):
    """Offline provider that reports a model id and token usage, so cost can be estimated."""

    _model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self._script = [
            assistant(
                tool_calls=[ToolCall(id="c1", name="search", args={"query": "tensorsketch"})]
            ),
            assistant(content="TensorSketch is a code-first, durable agentic framework."),
        ]
        self._i = 0

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        message = self._script[self._i]
        self._i += 1
        return Completion(
            message=message,
            usage=Usage(input_tokens=800 + 200 * self._i, output_tokens=120),
            model=self._model,
        )


async def main() -> None:
    tracer = InMemoryTracer()
    agent = create_agent(DemoProvider(), tools=[search])

    out = await agent.invoke({"query": "what is tensorsketch?"}, tracer=tracer)
    print(f"answer: {out.output}\n")

    print("trace:")
    print(tracer.trace.render())
    print(f"\nsummary: {tracer.trace.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
