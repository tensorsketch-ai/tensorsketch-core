# MCP interop

The [Model Context Protocol](https://modelcontextprotocol.io) is the emerging standard for
connecting agents to tools. TensorSketch speaks it **both ways**: use anyone's MCP tools inside a TensorSketch
agent, and expose your TensorSketch tools to any MCP client (Claude Desktop, another agent, an IDE).

It's an optional install — the MCP SDK is imported only when you use this module, so the core
stays dependency-free:

```bash
pip install tensorsketch-core[mcp]
```

```python
from tensorsketch.interop.mcp import mcp_tools, stdio_session, build_mcp_server, serve_stdio
```

## Consume external tools (client)

Connect to a server, wrap its tools as TensorSketch `Tool`s, and hand them to an agent. Each call is
forwarded over MCP; the remote tool's JSON Schema becomes the tool's advertised schema, so the
model sees exactly the right arguments.

```python
from tensorsketch import create_agent
from tensorsketch.interop.mcp import mcp_tools, stdio_session

async with stdio_session("python", "weather_server.py") as session:
    tools = await mcp_tools(session)              # remote tools → TensorSketch tools
    agent = create_agent(provider, tools=tools)
    result = await agent.invoke({"query": "what's the weather in Paris?"})
```

`stdio_session(command, *args)` launches a server as a subprocess and yields an initialized
session. Already have a session (SSE / streamable-HTTP transport)? Pass it straight to
`mcp_tools(session)` — the wrapping is transport-agnostic.

A tool result comes back as a Python value: **structured content** (a dict) when the server
returns one, otherwise the joined text. A server-side error surfaces as `MCPError`.

## Expose your tools (server)

Turn TensorSketch tools into an MCP server any client can call. A TensorSketch tool's schema and description
become the MCP tool; a call runs `tool.run(...)`.

```python
from tensorsketch import tool
from tensorsketch.interop.mcp import serve_stdio

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

await serve_stdio([add], name="my-tools")   # runs over stdio until closed
```

Need to embed the server (a custom transport, or tests)? `build_mcp_server(tools, name=...)`
returns the configured low-level MCP `Server` for you to run.

## How the bridge works

There's no magic layer — the adapter is thin and symmetric:

| Direction | Mapping |
|---|---|
| remote → TensorSketch | MCP tool's `inputSchema` → the TensorSketch tool's advertised JSON Schema; a call → `session.call_tool` |
| TensorSketch → remote | TensorSketch tool's `json_schema()` + `description` → an MCP tool; a call → `tool.run(...)` |

This works because a TensorSketch `Tool` separates its **advertised schema** from its **invocation**: a
local `@tool` derives the schema from the function signature, while a remote tool carries the raw
JSON Schema the server provided. Same `Tool` type either way, so remote and local tools mix
freely in one agent.

## Try it

[`examples/mcp_interop.py`](../../examples/mcp_interop.py) exposes TensorSketch tools as an MCP server and
then uses them from a TensorSketch agent — over the real protocol, connected in-memory, so it runs
offline with no subprocess.
