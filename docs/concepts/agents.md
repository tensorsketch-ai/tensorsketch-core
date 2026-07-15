# Agents

An **agent** is the model+tools loop: call the model, and if it asks for tools, run them and
feed the results back, until the model answers or a budget is reached. In TensorSketch an agent is a
single, **durable** `Node` — every model and tool call inside the loop is journaled, so a crash
mid-loop resumes without repeating a single API call.

## The quick path: `create_agent`

```python
from tensorsketch import create_agent, tool
from tensorsketch.providers.anthropic import AnthropicProvider

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = create_agent(
    AnthropicProvider(model="claude-sonnet-4-6"),
    tools=[add],
    system="You are a careful calculator.",
    max_iterations=8,
)

result = await agent.invoke({"query": "what is 2 + 3?"})
print(result.output)     # the final answer
print(result.messages)   # the full transcript (system, user, assistant, tool, ...)
```

`create_agent` returns a normal compiled graph, so everything you know applies: run it with
`invoke`, watch it with [`stream`](streaming.md), make it durable with a `thread_id` + backend.

## Durability comes for free

The agent wraps each model call and each tool call in `ctx.step`. Run it with a backend:

```python
result = await agent.invoke({"query": "..."}, thread_id="chat-1", backend=SqliteBackend("a.db"))
```

If the process dies at iteration 5, resuming the same thread replays iterations 0–4 **from the
journal** — no repeated LLM calls, no repeated tool side effects, no reasoning drift — and
continues from where it stopped. (This is proven in the test suite's crash-harness.)

## The budget

`max_iterations` caps the loop so a misbehaving model can't spin forever. When the budget is
hit, the agent returns the best answer it has rather than erroring.

## Composing agents into graphs

`Agent` is just a `Node`. Drop it into any [graph](nodes-and-graphs.md) — route to different
agents, run several in parallel, put a human-review node after one. `create_agent` is the
convenience wrapper; the primitive composes.

## Single calls and structured output

For a one-shot call, use the `Llm` node:

```python
from tensorsketch import Llm
graph.add(Llm(provider, system="Summarize the input."))
```

### Structured output

`generate_structured` asks the model for a specific `Schema` and returns a validated instance —
call it inside any node body (pass `ctx` to journal it):

```python
from tensorsketch import Schema, generate_structured

class Sentiment(Schema):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

result = await generate_structured(provider, Sentiment, "I loved it!", ctx=ctx)
# result is a validated Sentiment
```

If the model's reply doesn't match the schema, `generate_structured` feeds the validation error
back and asks again (up to `max_repairs` times) — the validate-and-repair loop — before giving
up. Each attempt is journaled when you pass `ctx`.

## Composing agents with patterns

Inside an agent or node, the [composition patterns](patterns.md) — `gather_map` (map/reduce over
data), `parallel` (independent calls at once), and `run_subgraph` (call one graph from another) —
let you build map-reduce and multi-step flows that stay durable end to end.

## Roadmap

Sub-agent handoff, supervisor/team patterns, and structured *agent* output build on this loop.
Agent memory is intentionally **not** built into the framework — TensorSketch stays stateless and you
bring your own store; see the [decisions log](../design/decisions.md). See also the
[architecture plan](../design/framework-design.md).
