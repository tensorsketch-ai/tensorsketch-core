# Getting started

This guide builds a small **support router**: it classifies a user's query and routes it to a
specialist. Along the way you'll meet the core of TensorSketch's authoring API — typed state, typed
nodes, sequential and conditional edges, and running a graph.

The finished example lives at [`examples/support_router.py`](../../examples/support_router.py).

## 1. Define the state

A graph has one **state Schema**. Every field is a *channel* the runtime stores and updates.

```python
from typing import Literal
from tensorsketch import Schema

Intent = Literal["billing", "tech", "other"]

class Support(Schema):
    query: str
    intent: Intent = "other"
    answer: str = ""
```

`query` is required (the caller provides it); `intent` and `answer` have defaults, so they
start populated and get overwritten as the graph runs.

## 2. Write typed nodes

A `Node` declares an **`In`** Schema (the state fields it reads) and an **`Out`** Schema (the
fields it writes). The body is ordinary async code — TensorSketch never looks inside it.

```python
from tensorsketch import Node, Context

class Classify(Node):
    class In(Schema):
        query: str
    class Out(Schema):
        intent: Intent

    async def run(self, ctx: Context, inp: In) -> Out:
        q = inp.query.lower()
        if any(w in q for w in ("refund", "charge", "invoice")):
            return self.Out(intent="billing")
        if any(w in q for w in ("error", "crash", "bug")):
            return self.Out(intent="tech")
        return self.Out(intent="other")
```

Add specialist nodes the same way — `Billing`, `Tech`, and `Fallback`, each reading `query`
and writing `answer`. (In a real agent, these bodies would call an LLM or a tool; the graph
would look identical.)

### Don't have the body yet? Leave a typed hole

You can declare a node's interface and defer its body:

```python
from tensorsketch import Hole

class Billing(Node):
    class In(Schema):
        query: str
    class Out(Schema):
        answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        raise Hole("Answer billing questions using the KB tool")
```

The graph still compiles and type-checks; running it will stop at the hole. `Hole` is a
greppable, type-checked marker for "this node needs code" — later phases turn that description
into a real body.

## 3. Wire the graph

Add nodes, set the entry with `START`, and connect edges. Use `conditional` to route
dynamically based on state:

```python
from tensorsketch import Graph, START, END

def route(state: Support) -> str:
    return {"billing": "Billing", "tech": "Tech"}.get(state.intent, "Fallback")

app = (
    Graph(Support)
    .add(Classify).add(Billing).add(Tech).add(Fallback)
    .edge(START, "Classify")
    .conditional("Classify", route)
    .edge("Billing", END).edge("Tech", END).edge("Fallback", END)
).compile()
```

`compile()` validates the whole graph: every port maps to a real state field of a compatible
type, and every edge points at a real node. Mistakes are caught here, before anything runs.

## 4. Run it

```python
import asyncio

out = asyncio.run(app.invoke({"query": "I'd like a refund on my invoice"}))
print(out.intent)   # billing
print(out.answer)   # [Billing] Looking into your billing question: ...
```

`invoke` seeds the state with your input, runs the graph to completion, and returns the final
`Support` state — fully typed, so your editor knows `out.intent` and `out.answer`.

## Where to go next

- **[State & channels](../concepts/state-and-channels.md)** — accumulate values with reducers.
- **[Nodes & graphs](../concepts/nodes-and-graphs.md)** — fan-out, joins, and holes in depth.
- **[Execution model](../concepts/execution-model.md)** — how supersteps actually run.
