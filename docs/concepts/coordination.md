# Multi-agent coordination

Real systems are rarely one agent. A **supervisor** triages and delegates to specialists; a
research agent hands off to a writer; a planner farms sub-tasks out to workers. TensorSketch expresses all
of these with one small primitive — `as_tool` — and **nothing new in the runtime**.

## Agents as tools

An [agent](agents.md) is a compiled graph you invoke with a query and read an answer from. `as_tool`
wraps one as a [`Tool`](tools.md), so another agent can *call* it exactly like any other tool:

```python
from tensorsketch import as_tool, create_agent

billing = create_agent(provider, tools=[lookup_invoice], system="You handle billing.")
tech = create_agent(provider, tools=[search_kb], system="You debug technical issues.")

supervisor = create_agent(
    provider,
    tools=[
        as_tool(billing, name="billing", description="Answer billing questions."),
        as_tool(tech, name="tech", description="Debug technical problems."),
    ],
    system="Route each request to the right specialist, then relay the answer.",
)

result = await supervisor.invoke({"query": "I want a refund for order #7"})
```

The supervisor runs the ordinary [agent loop](agents.md): the model sees `billing` and `tech` as
callable tools, picks one, and the tool call *runs that specialist* and returns its answer as the
tool result. That's the whole supervisor / handoff pattern — the same ReAct loop, one level up.

## Why this needs no new machinery

Because a delegation is just a tool call, it inherits everything the agent loop already guarantees:

- **Durability.** Each tool call is wrapped in `ctx.step`, so a specialist's whole run is journaled.
  If the supervisor crashes after delegating but before finishing, resuming **replays** the
  specialist's answer from the journal instead of re-running it — no duplicated model calls, no
  drift.
- **One trace for the whole team.** The specialist runs under the caller's tracer, so its model and
  tool [spans nest](tracing.md) under the delegating tool call. A single trace tree shows the
  supervisor, each specialist it called, and the per-specialist cost.
- **Composability.** Specialists are ordinary agents, so they can have their own tools, their own
  sub-agents (supervisors of supervisors), or a different provider per specialist — a cheap model
  for triage, a strong one for the hard specialist.

## Shaping the call

`as_tool` defaults match `create_agent` — a `query` in, an `output` out — and exposes a single
string argument the supervisor fills:

```python
as_tool(
    graph,
    name="research",
    description="Research a topic and return a summary.",
    input_key="query",        # the state field the sub-agent reads
    output_key="output",      # the state field to return as text
    arg="request",            # the tool argument the caller fills
    arg_description="What to research.",
)
```

To wrap a graph whose state is shaped differently, point `input_key` / `output_key` at its fields:

```python
# a writer graph with state {topic, draft}
as_tool(writer, name="writer", description="Draft a section.",
        input_key="topic", output_key="draft")
```

## Context-aware tools

`as_tool` is built on a general capability: **a tool function may declare a `ctx` parameter**, and
TensorSketch injects the run [`Context`](nodes-and-graphs.md) into it (it's never shown to the model). That
lets a tool journal its own durable steps, emit stream events, or — as `as_tool` does — run a
sub-graph under the same trace:

```python
@tool
def remember(ctx: Context, note: str) -> str:
    """Persist a note durably."""
    ...  # ctx is injected; the model only sees `note`
```

See the runnable `examples/multi_agent.py` for a supervisor routing to two specialists, offline.
