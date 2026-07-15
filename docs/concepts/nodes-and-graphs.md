# Nodes & graphs

A TensorSketch program is **typed nodes** wired over **typed state**. This page covers the authoring
API in depth: ports, edges, routing, fan-out/fan-in, and holes.

## Nodes: typed ports, opaque bodies

A `Node` declares two nested Schemas and one method:

```python
from tensorsketch import Node, Schema, Context

class Classify(Node):
    class In(Schema):        # input ports  = state fields this node READS
        query: str
    class Out(Schema):       # output ports = state fields this node WRITES
        intent: str

    async def run(self, ctx: Context, inp: In) -> Out:
        ...                  # opaque body: LLM calls, tools, parsing — anything async
        return self.Out(intent="billing")
```

The split is deliberate and is the heart of TensorSketch's design:

- **The interface (`In`/`Out`) is transparent.** It's what the compiler type-checks, what the
  canvas will draw, and what natural-language generation will target.
- **The body (`run`) is opaque.** TensorSketch never introspects it. You get the full power of the host
  language inside a node without giving up a statically-inspectable graph.

A node's ports are **slices of the graph's state**: an `In` field named `query` reads
`state.query`; an `Out` field named `intent` writes `state.intent`. The field types must be
compatible with the state's — checked at `compile()`.

### Node names

A node's default name is its class name. Override it when adding:

```python
g.add(Classify)                 # name = "Classify"
g.add(Classify, name="Triage")  # name = "Triage"
```

## Graphs: wiring nodes over state

`Graph(StateSchema)` is a fluent builder. Every method returns the graph, so wiring chains:

```python
from tensorsketch import Graph, START, END

app = (
    Graph(State)
    .add(Classify).add(Billing).add(Tech)
    .edge(START, "Classify")                # entry
    .conditional("Classify", route)         # dynamic routing out of Classify
    .edge("Billing", END).edge("Tech", END) # terminals
).compile()
```

### Edges

- **`edge(src, dst)`** — a sequential edge: after `src` runs, `dst` runs.
- **`edge(START, x)`** — sets the entry node. (`entry("x")` is a synonym.)
- **`edge(x, END)`** — marks `x` as terminal along that path.
- **Fan-out:** add several edges from one node (`edge("A", "B")`, `edge("A", "C")`) and both
  run — in parallel, in the next superstep.
- **Fan-in / join:** point several nodes at one (`edge("B", "D")`, `edge("C", "D")`). When B and
  C finish together, D runs once, reading their merged writes. Give the joined field a reducer
  so both writes combine instead of colliding.

### Conditional edges (routing)

`conditional(src, path, mapping=None)` routes dynamically. `path` receives the current state
and returns the next node name — or a list of names for dynamic fan-out, or `END` to stop:

```python
def route(state: State) -> str:
    return "Billing" if state.intent == "billing" else "Tech"

g.conditional("Classify", route)
```

With a `mapping`, `path` can return short keys that are looked up:

```python
g.conditional("Classify", lambda s: s.intent, {"billing": "Billing", "tech": "Tech"})
```

A node may have **either** static edges **or** a conditional edge, not both — `compile()`
enforces this so a node's successors have one clear source.

`router(src, path, mapping=None)` is the same thing under an intent-revealing name — reach for it
when the point is "pick where to go next," and especially for the dynamic fan-out below.

### Dynamic fan-out with `Send`

A conditional returning a *list of node names* fans out to those (distinct) nodes. To instead run
**the same node many times, each on its own input**, return a list of **`Send`s**. The engine
schedules one instance per `Send` — each its own superstep task with its own payload — and they all
merge at the next barrier. This is a graph-level **map/reduce**:

```python
from tensorsketch import Send

class State(Schema):
    numbers: list[int] = []
    n: int = 0                                    # per-worker input slot
    squares: Annotated[list[int], Reducer(add)] = []   # workers merge here
    total: int = 0

g.router("Split", lambda s: [Send("Square", {"n": x}) for x in s.numbers])
g.edge("Square", "Total")                         # every worker converges on one Total (deduped)
```

- The **payload** provides the worker's `In` fields; any field it omits still reads from shared
  state. (So the payload keys are ordinary state fields — a `Send` just overrides them for *that*
  instance.) A `Schema` works too: `Send("Square", State.In(...))`.
- Workers must write an **aggregating channel** — a `Reducer` or `Topic` — so their results merge
  at the barrier instead of overwriting one another. A downstream node then reads the merged value.
- Fan-out is **durable**: each instance journals its `ctx.step` effects under a distinct key, so a
  crash mid-fan-out resumes and replays completed workers exactly once.

This is the *graph-level* counterpart to [`gather_map`](patterns.md) (which fans out inside one
node's body). Reach for `Send` when each unit should be its own node/superstep — visible in the
trace, checkpointed, and individually durable.

### Loops

`loop(node, until, *, exit=END)` repeats a node until a predicate holds, then continues — sugar
over a self-conditional (`node → node` while `not until(state)`, else `→ exit`):

```python
g.edge(START, "Refine").loop("Refine", until=lambda s: s.score >= 0.9)
```

Wire the entry separately; `loop` adds only the repeat/exit branch. The `invoke(max_steps=...)`
recursion limit still bounds a runaway loop.

## The `>>` wiring surface

For wiring that reads like the diagram it describes, `Graph.nodes(...)` hands back a **handle**
per node, and the handles overload `>>`:

```python
from tensorsketch import Graph, Router, START, END

g = Graph(Support)
classify, billing, tech = g.nodes(Classify, Billing, Tech)

START >> classify                                       # entry
classify >> Router(route, billing=billing, tech=tech)   # conditional fan-out
billing >> END                                          # terminate a branch
tech >> END

app = g.compile()
```

The operators just call `.add`/`.edge`/`.conditional` underneath, so this is **pure sugar** —
the compiled graph is identical to the fluent form, and it round-trips through the code⇄canvas
engine the same way. The shapes:

| Expression | Meaning |
| --- | --- |
| `a >> b` | sequential edge `a → b` |
| `a >> [b, c]` | fan-out (two sequential edges) |
| `START >> a` | set the entry node |
| `a >> END` | terminate the branch out of `a` |
| `a >> Router(fn)` | dynamic conditional (targets decided at runtime) |
| `a >> Router(fn, {"k": b})` / `Router(fn, k=b)` | mapped conditional |

`a >> b >> c` chains left to right (each `>>` returns its right operand). `g["Name"]` gives a
handle for a node you already added, so you can mix styles. Pick whichever reads best — the
[fluent builder](#graphs-wiring-nodes-over-state), statement-by-statement calls, or `>>`; all
three compile to the same graph and extract to the same canvas.

## Holes: declare the interface, defer the body

Raise `Hole` to mark a node as "needs code" while keeping its typed interface:

```python
from tensorsketch import Hole

class BillingAgent(Node):
    class In(Schema):  query: str
    class Out(Schema): answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        raise Hole("Answer billing questions using the KB tool")
```

The graph compiles and type-checks; running reaches the hole and raises it. Because holes are
greppable and typed, tooling can report "3 nodes need code", and a later phase can compile the
description into a real body against the same `In`/`Out` contract.

## Compile-time validation

`compile()` rejects structural mistakes before any run, with messages that teach:

- no entry node set;
- an edge (or conditional target) pointing at a node that doesn't exist;
- a port that reads/writes a state field that doesn't exist;
- a port whose type is incompatible with its state channel;
- a node given both static and conditional successors;
- a duplicate node name.

## Running a graph

```python
out = await app.invoke(
    {"query": "..."},   # seeds matching state channels (dict or a State instance)
    max_steps=25,       # recursion limit — the safety net for loops
)
```

`invoke` returns the final state, typed as your state Schema.

## Upcoming

The [code⇄canvas engine](code-and-canvas.md) builds on top of these authoring surfaces — the
fluent builder, statement-style calls, and `>>` all extract to the same graph, which the visual
canvas renders and edits. See the [roadmap](../design/roadmap.md).
