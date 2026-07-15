# Tools

A tool is a function the model can call. Write an ordinary function, annotate its parameters,
add a docstring, and decorate it with `@tool` — TensorSketch derives the JSON schema the model needs
from the signature, so there's nothing to hand-write.

```python
from tensorsketch import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

`add` is now a `Tool`:

- **`add.name`** → `"add"` (the function name; override with `@tool(name=...)`).
- **`add.description`** → the docstring (override with `@tool(description=...)`).
- **`add.json_schema()`** → the argument schema advertised to the model, inferred from the
  annotations (`{"a": integer, "b": integer}`, both required).

## Calling a tool

`Tool.run(args)` validates the arguments against the schema, then invokes the function:

```python
await add.run({"a": 2, "b": 3})        # 5
await add.run({"a": "oops", "b": 3})   # raises ValidationError
```

Because arguments are validated first, a model that returns the wrong shape fails loudly with a
clear error instead of blowing up inside your function.

## Sync or async

Tools can be either — an `async def` tool is awaited automatically:

```python
@tool
async def fetch(url: str) -> str:
    """Fetch a URL."""
    return await http_get(url)
```

## Using tools

Pass tools to an [agent](agents.md); the agent advertises them to the model, runs the ones the
model asks for, and feeds the results back:

```python
from tensorsketch import create_agent
agent = create_agent(provider, tools=[add, fetch])
```

Inside an agent, every tool call is wrapped in `ctx.step`, so tool side effects are
[durable](durability.md) — run exactly once, even across a crash and resume.

## Context-aware tools

A tool function may declare a `ctx` parameter. When it does, TensorSketch injects the run
[`Context`](nodes-and-graphs.md) into it — and never advertises `ctx` to the model:

```python
@tool
def remember(ctx: Context, note: str) -> str:
    """Persist a note."""
    ...  # the model only sees `note`; `ctx` is injected at call time
```

This lets a tool journal its own durable steps, emit stream events, or run a sub-graph under the
same trace. It's the seam that [`as_tool`](coordination.md) uses to run one agent from inside
another.

## Roadmap

Hosted tools, MCP tool servers, and per-parameter descriptions parsed from the docstring build
on this. See the [architecture plan](../design/framework-design.md).
