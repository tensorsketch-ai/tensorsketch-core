"""MCP interop — TensorSketch tools out to a server, remote tools back in, over the real protocol.

Uses the MCP SDK's in-memory transport (`create_connected_server_and_client_session`), so this
exercises the actual client/server handshake and tool calls without a subprocess or a network.
"""

from __future__ import annotations

import sys

import pytest

from tensorsketch import tool

pytest.importorskip("mcp")

from mcp.shared.memory import (
    create_connected_server_and_client_session as connected,
)

from tensorsketch.interop.mcp import build_mcp_server, mcp_tools


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def profile(name: str) -> dict[str, str]:
    """Return a small structured record."""
    return {"greeting": f"hello {name}", "name": name}


async def test_loom_tools_are_exposed_and_callable_over_mcp() -> None:
    server = build_mcp_server([add, profile], name="test")
    async with connected(server) as session:
        # the server advertises both tools, with their schemas
        listed = await session.list_tools()
        assert {t.name for t in listed.tools} == {"add", "profile"}
        add_spec = next(t for t in listed.tools if t.name == "add")
        assert add_spec.inputSchema["properties"]["a"]["type"] == "integer"
        assert "Add two integers." in (add_spec.description or "")

        # a scalar result comes back as text; a dict comes back as structured content
        scalar = await session.call_tool("add", {"a": 2, "b": 3})
        assert scalar.content[0].text == "5"
        structured = await session.call_tool("profile", {"name": "ada"})
        assert structured.structuredContent == {"greeting": "hello ada", "name": "ada"}


async def test_remote_tools_wrap_as_loom_tools() -> None:
    server = build_mcp_server([add, profile], name="test")
    async with connected(server) as session:
        tools = await mcp_tools(session)
        by_name = {t.name: t for t in tools}
        assert set(by_name) == {"add", "profile"}

        # the wrapped tool advertises the remote JSON schema to the model
        assert by_name["add"].json_schema()["properties"]["b"]["type"] == "integer"

        # calling the TensorSketch tool forwards over MCP and returns a Python value
        assert await by_name["add"].run({"a": 4, "b": 5}) == "9"
        assert await by_name["profile"].run({"name": "grace"}) == {
            "greeting": "hello grace",
            "name": "grace",
        }


async def test_unknown_tool_errors() -> None:
    from tensorsketch.interop.mcp import MCPError

    server = build_mcp_server([add], name="test")
    async with connected(server) as session:
        result = await session.call_tool("nope", {})
        assert result.isError  # the server reports the error rather than crashing

        tools = await mcp_tools(session)
        with pytest.raises(MCPError):
            # a wrapped tool surfaces a remote failure as MCPError
            await tools[0].run({"a": "not-an-int", "b": 1})


def test_importing_loom_does_not_import_mcp() -> None:
    # The core must stay dependency-free: in a fresh interpreter, `import tensorsketch` pulls in no
    # MCP SDK.
    import subprocess

    code = "import tensorsketch, sys; assert 'mcp' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
