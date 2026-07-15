"""A tool-using agent, runnable offline with a scripted provider.

Shows the whole Phase 2 surface: a `@tool` with an auto-generated schema, `create_agent`, and
the model→tool→model loop. It uses `FakeProvider` so it runs with no API key; the comment shows
the one-line swap to a real model.

Run:  uv run python examples/calculator_agent.py
"""

from __future__ import annotations

import asyncio

from tensorsketch import FakeProvider, create_agent, tool
from tensorsketch.messages import ToolCall, assistant


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main() -> None:
    # Offline, deterministic provider: first it asks for the tool, then it answers.
    provider = FakeProvider(
        [
            assistant(tool_calls=[ToolCall(id="c1", name="multiply", args={"a": 6, "b": 7})]),
            assistant(content="6 times 7 is 42."),
        ]
    )
    # For a real model instead:
    #   from tensorsketch.providers.anthropic import AnthropicProvider
    #   provider = AnthropicProvider(model="claude-sonnet-4-6")   # needs ANTHROPIC_API_KEY

    agent = create_agent(
        provider,
        tools=[multiply],
        system="You are a helpful calculator. Use tools for arithmetic.",
    )

    result = await agent.invoke({"query": "what is 6 * 7?"})

    print(f"answer: {result.output}\n")
    print("transcript:")
    for msg in result.messages:
        detail = msg.content
        if msg.tool_calls:
            detail = ", ".join(f"{c.name}({c.args})" for c in msg.tool_calls)
        print(f"  {msg.role:<9} {detail}")


if __name__ == "__main__":
    asyncio.run(main())
