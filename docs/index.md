# TensorSketch

**A code-first, visually-editable, durable agentic framework.**

TensorSketch is a framework for building AI agents and agentic workflows where:

- **Code is the single ground truth.** A visual canvas — [TensorSketch Studio](guides/studio.md) — is a
  *losslessly-synced projection* of your code, never a second, competing source of truth. Edits on
  the canvas write straight back into the source.
- **One type abstraction runs through everything.** The same `Schema` describes tool inputs,
  structured LLM output, your graph's state, and every node's ports.
- **Execution is durable and parallel by construction.** A BSP (bulk-synchronous parallel)
  runtime gives you cycles, deterministic fan-out, dynamic fan-out (`Send`), and clean checkpoint
  boundaries — so a run resumes exactly where it left off.
- **The core knows interfaces, never implementations.** Every provider, tool, database backend,
  and protocol is chosen by name or passed in, so TensorSketch absorbs new research without a rewrite.

> **Status: Phases 0, 2, and 3 complete; Phase 1 (code⇄canvas) in progress.** Built and tested:
> the type spine and BSP runtime; durable execution (checkpoints, resume/fork, exactly-once
> effects); streaming; the full agent layer (tools, three providers, the durable agent loop,
> structured output, dynamic fan-out); interop and observability (MCP, middleware, tracing +
> exporters, a name registry, OpenAI/A2A/AG-UI serving, an eval harness with drift detection); and
> **TensorSketch Studio** — the visual canvas with a live trace overlay. The public API is pre-1.0 and may
> still change. See the [roadmap](design/roadmap.md) and [build status](design/status.md).

---

## Install (development)

```bash
cd tensorsketch
uv sync
uv run pytest        # the test suite (green on 3.11 + 3.12)
```

See [Installation](guides/installation.md) for details.

## Hello, graph

A graph is **typed nodes** wired over **typed state**. Here's a two-step pipeline:

```python
import asyncio
from tensorsketch import Schema, Node, Graph, Context, START, END


class State(Schema):        # the shared, typed state — each field is a channel
    text: str
    shout: str = ""


class Shout(Node):
    class In(Schema):       # this node reads state.text ...
        text: str
    class Out(Schema):      # ... and writes state.shout
        shout: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(shout=inp.text.upper() + "!")


app = (
    Graph(State)
    .add(Shout)
    .edge(START, "Shout")
    .edge("Shout", END)
).compile()

out = asyncio.run(app.invoke({"text": "hello"}))
print(out.shout)   # HELLO!
```

`out` is a fully-typed `State`, so `out.shout` is known to be a `str`.

Next: **[Getting started](guides/getting-started.md)** builds a routing agent step by step.

## Learn the model

- **[State & channels](concepts/state-and-channels.md)** — how state is stored and reduced.
- **[Nodes & graphs](concepts/nodes-and-graphs.md)** — typed ports, edges, routing, holes.
- **[Execution model](concepts/execution-model.md)** — how the BSP superstep runtime runs.
- **[Durability](concepts/durability.md)** — checkpoints, resume/fork, and exactly-once effects.
- **[Streaming](concepts/streaming.md)** — live namespaced events, `ctx.emit`, resumable replay.
- **[Tools](concepts/tools.md)** · **[Providers](concepts/providers.md)** ·
  **[Agents](concepts/agents.md)** — build a durable model+tools agent.
- **[Composition patterns](concepts/patterns.md)** — map/reduce, parallel, and subgraphs.
- **[Tracing](concepts/tracing.md)** · **[Evaluation](concepts/evaluation.md)** ·
  **[Serving](concepts/serving.md)** — observe, grade, and expose an agent.
- **[Code ⇄ canvas](concepts/code-and-canvas.md)** and **[Studio](guides/studio.md)** — the
  visual projection of your code.

## Design

- **[Architecture plan](design/framework-design.md)** — the full "TensorSketch" design.
- **[Roadmap](design/roadmap.md)** — phases and what's built so far.
