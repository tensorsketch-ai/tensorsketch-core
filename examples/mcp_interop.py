"""MCP interop: expose TensorSketch tools as an MCP server, then use them from a TensorSketch agent.

Both directions run over the **real** Model Context Protocol, connected in-memory (no subprocess,
no network), so the example runs offline. In practice the two sides are different processes —
you'd `serve_stdio(tools)` on one and `stdio_session("python", "server.py")` on the other.

Requires the mcp extra:  uv run --extra mcp python examples/mcp_interop.py
"""

from __future__ import annotations

import asyncio

from mcp.shared.memory import create_connected_server_and_client_session as connected

from tensorsketch import FakeProvider, create_agent, tool
from tensorsketch.interop.mcp import build_mcp_server, mcp_tools
from tensorsketch.messages import ToolCall, assistant


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def weather(city: str) -> dict[str, object]:
    """Look up the (pretend) weather for a city."""
    return {"city": city, "tempC": 21, "sky": "clear"}


async def main() -> None:
    # Expose the TensorSketch tools as an MCP server.
    server = build_mcp_server([multiply, weather], name="demo-tools")

    async with connected(server) as session:
        # Client side: discover the server's tools and wrap them as TensorSketch tools.
        tools = await mcp_tools(session)
        print("discovered over MCP:", [t.name for t in tools])

        remote_weather = next(t for t in tools if t.name == "weather")
        print("weather('Paris') ->", await remote_weather.run({"city": "Paris"}))

        # Drop the remote tools straight into an agent — the model calls them over MCP.
        provider = FakeProvider(
            [
                assistant(tool_calls=[ToolCall(id="c1", name="multiply", args={"a": 6, "b": 7})]),
                assistant(content="6 times 7 is 42."),
            ]
        )
        agent = create_agent(provider, tools=tools, system="Use tools for arithmetic.")
        result = await agent.invoke({"query": "what is 6 * 7?"})
        print("agent answer:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
