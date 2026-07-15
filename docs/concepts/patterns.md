# Composition patterns

The classic control-flow shapes — fan a collection out and reduce it, run several things at
once, call one graph from inside another — are helpers you call inside a node's `run`. Each
wraps its work in `ctx.step`, so they inherit TensorSketch's [durability](durability.md): on resume,
work that already finished is replayed from the journal instead of re-run.

## `gather_map` — map/reduce over a collection

Run an async function over every item **concurrently**, in order, durably:

```python
from tensorsketch import gather_map

class Summarize(Node):
    class In(Schema):  docs: list[str]
    class Out(Schema): summaries: list[str]

    async def run(self, ctx, inp):
        async def summarize(doc: str) -> str:
            return await llm_summarize(doc)     # a real model/tool call
        summaries = await gather_map(ctx, inp.docs, summarize, max_concurrency=5)
        return self.Out(summaries=summaries)
```

Because a concurrent map can't depend on completion order, each item is journaled under an
explicit, deterministic key — so if the process dies halfway through 100 documents, resuming
only processes the ones that didn't finish. `max_concurrency` caps how many run at once.

## `parallel` — run independent calls at once

```python
from tensorsketch import parallel

async def run(self, ctx, inp):
    weather, news = await parallel(
        ctx,
        lambda: get_weather(inp.city),
        lambda: get_news(inp.city),
    )
    ...
```

Results come back in argument order, and each call is durable.

## `run_subgraph` — compose graphs

Build small graphs and call them from larger ones. `run_subgraph` runs a compiled graph and
returns its final (typed) state; pass `ctx` to journal the whole call as one durable step.

```python
from tensorsketch import run_subgraph

async def run(self, ctx, inp):
    result = await run_subgraph(research_graph, {"topic": inp.topic}, ctx=ctx)
    return self.Out(report=result.summary)
```

This is the composition primitive: an agent can call a sub-workflow, which can call another,
each a normal graph you can test in isolation.

## Why body helpers (for now)

These are functions you call inside a node, not new graph-builder syntax. That keeps them fully
type-safe and lets them compose freely inside agent loops. A graph-level dynamic fan-out
(spawning parallel node instances that each appear as their own superstep) is a planned runtime
addition; see the [architecture plan](../design/framework-design.md).
