"""MCP (Model Context Protocol) interop — consume external tool servers and expose TensorSketch's.

`pip install tensorsketch-core[mcp]`. Two directions, both over the official `mcp` SDK (imported
here, so the
core never depends on it — importing this module is how you opt in):

**Client — use someone else's tools.** `mcp_tools(session)` lists a connected server's tools and
wraps each as a TensorSketch `Tool`, ready to drop into `create_agent(tools=...)`:

    from tensorsketch.interop.mcp import mcp_tools, stdio_session

    async with stdio_session("python", "weather_server.py") as session:
        tools = await mcp_tools(session)
        agent = create_agent(provider, tools=tools)
        ...   # the model can now call the remote tools; each call goes over MCP

**Server — expose your tools.** `build_mcp_server(tools)` / `serve_stdio(tools)` turn TensorSketch
tools
into an MCP server any client (Claude Desktop, another agent) can call:

    from tensorsketch.interop.mcp import serve_stdio

    await serve_stdio([add, search], name="my-tools")

The bridge is thin: a remote tool's JSON Schema becomes the TensorSketch tool's advertised schema,
and a
call is forwarded to `session.call_tool`; in reverse, a TensorSketch tool's schema/description
become an
MCP tool and a call runs `tool.run(...)`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

from ..tools import Tool


class MCPError(RuntimeError):
    """A remote MCP tool call failed."""


# -- client: wrap a server's tools as TensorSketch tools -------------------------------------------


async def mcp_tools(session: ClientSession) -> list[Tool]:
    """List a connected MCP server's tools and wrap each as a TensorSketch `Tool`.

    The `session` must be initialized (see `stdio_session`, or use the SDK's transports
    directly). Each returned tool forwards its call to the server over MCP.
    """
    listed = await session.list_tools()
    return [_wrap_remote(session, spec) for spec in listed.tools]


def _wrap_remote(session: ClientSession, spec: mcp_types.Tool) -> Tool:
    name = spec.name
    schema = spec.inputSchema or {"type": "object", "properties": {}}

    async def call(**kwargs: Any) -> Any:
        result = await session.call_tool(name, kwargs)
        return _from_result(name, result)

    return Tool(call, name=name, description=spec.description or "", json_schema=schema)


def _from_result(name: str, result: mcp_types.CallToolResult) -> Any:
    """Turn an MCP tool result into a plain Python value (structured content, else joined text)."""
    if result.isError:
        raise MCPError(f"MCP tool {name!r} failed: {_text(result)}")
    if result.structuredContent is not None:
        return result.structuredContent
    return _text(result)


def _text(result: mcp_types.CallToolResult) -> str:
    return "\n".join(
        block.text for block in result.content if isinstance(block, mcp_types.TextContent)
    )


@asynccontextmanager
async def stdio_session(
    command: str, *args: str, env: dict[str, str] | None = None
) -> AsyncIterator[ClientSession]:
    """Launch an MCP server as a subprocess over stdio and yield an initialized `ClientSession`.

    async with stdio_session("python", "server.py") as session:
        tools = await mcp_tools(session)
    """
    params = StdioServerParameters(command=command, args=list(args), env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


# -- server: expose TensorSketch tools as an MCP server --------------------------------------------


def build_mcp_server(tools: Sequence[Tool], *, name: str = "tensorsketch") -> Server[Any]:
    """Build a low-level MCP `Server` that exposes `tools`. Run it with any MCP transport."""
    server: Server[Any] = Server(name)
    by_name = {t.name: t for t in tools}

    @server.list_tools()
    async def _list() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(name=t.name, description=t.description, inputSchema=t.json_schema())
            for t in tools
        ]

    @server.call_tool()
    async def _call(tool_name: str, arguments: dict[str, Any]) -> Any:
        tool = by_name.get(tool_name)
        if tool is None:
            raise MCPError(f"unknown tool: {tool_name!r}")
        result = await tool.run(arguments or {})
        return _to_content(result)

    return server


def _to_content(result: Any) -> Any:
    """Render a TensorSketch tool result as MCP content — a dict becomes structured content, else
    text."""
    if isinstance(result, dict):
        return result  # structured content (the SDK also serializes it into text)
    if isinstance(result, str):
        return [mcp_types.TextContent(type="text", text=result)]
    if isinstance(result, bool | int | float):
        return [mcp_types.TextContent(type="text", text=str(result))]
    return [mcp_types.TextContent(type="text", text=json.dumps(result, default=str))]


async def serve_stdio(tools: Sequence[Tool], *, name: str = "tensorsketch") -> None:
    """Serve `tools` as an MCP server over stdio (the common local transport). Runs until closed."""
    server = build_mcp_server(tools, name=name)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
