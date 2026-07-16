# Code ⇄ canvas

TensorSketch's defining idea: **your code is the single source of truth, and the visual canvas is a
losslessly-synced projection of it** — never a second, competing source. You sketch on the
canvas *or* write code, switch freely, and the two stay in sync.

This page covers the engine that makes that possible: **extraction** (code → graph),
**write-back** (graph → code), and the **round-trip invariant** that keeps them honest. The
[Studio](../guides/studio.md) is the visual canvas built on top of it — launch it with
`python -m tensorsketch.canvas <file>`.

## What round-trips, and what doesn't

Only two things round-trip between code and canvas:

- **The wiring** — which nodes connect to which (the `Graph(...).add(...).edge(...)` structure).
- **The typed interfaces** — each node's `In`/`Out` ports.

**Node bodies do not.** A node's `run` method is opaque — the canvas draws the node as a box with
its ports and never looks inside. This isn't a limitation to fix later; it's a computability
necessity. "What are this node's outgoing edges?" is a semantic property of a Turing-complete
program, which is undecidable (Rice's theorem). So TensorSketch keeps *wiring* a declarative, syntactic
surface that can round-trip, and treats *bodies* as opaque. This is exactly why it works where
"turn my arbitrary code into a diagram" tools don't.

## Extraction

```python
from tensorsketch.canvas import extract

ir = extract(open("support_agent.py").read())
ir.to_dict()   # JSON a canvas can render
```

`extract` parses the source with a **CST** (concrete syntax tree) and reads:

- every `class X(Node)` — its name, its `In`/`Out` port fields (name + type), and whether its
  body is an **unfilled hole** (`raise Hole(...)`);
- the graph's wiring — the nodes, entry, and edges (conditional edges carry their routing
  function; a mapping expands to one edge per target). `router(...)` reads as a `conditional`
  (its intent-named alias), and `loop(node, until, *, exit=END)` reads as a two-branch conditional
  — one edge back to the node (a self-loop the canvas draws as an arc), one to `exit`.

**Dynamic targets can't round-trip — by design.** When a route is decided inside an opaque callable
— `router("split", lambda s: [Send("worker", …) for … ])`, or a `loop`/`conditional` whose predicate
is a lambda — the *targets* aren't statically knowable (Rice's theorem again). Extraction shows
these as a **dynamic-route stub** (a `?` off the node) rather than inventing edges. The graph still
runs; the canvas is just honest that the destination is computed at runtime.

Wiring is read no matter which authoring style the source uses — they all fold to the same
`GraphIR`:

- a **fluent chain** — `app = Graph(S).add(A).edge(x, y).conditional(...)`;
- **statement style** — `g = Graph(S)`, then `g.add(A)`, `g.edge(x, y)` on their own lines;
- the **`>>` surface** — `a, b = g.nodes(A, B)`, then `START >> a`, `a >> Router(fn, ...)`.

It produces a **`GraphIR`** — plain data (`GraphIR` / `NodeIR` / `EdgeIR` / `Port`) with a
`to_dict()` for JSON.

### It works on incomplete code

Extraction is **purely syntactic** — it never imports or runs your module. So it works on code
that isn't finished: undefined names, missing imports, and unfilled holes are all fine. That's
essential — you need to see the graph *while* you're building it, holes and all. (A node whose
body is `raise Hole(...)` shows up with `has_hole=True`, so tooling can surface "3 nodes need
code".)

### Surfacing holes across a project

The same syntactic reading powers a *what's left to implement?* view over a whole codebase, not
just one file:

```python
from tensorsketch.canvas import find_holes

for hole in find_holes("src/"):
    print(f"{hole.file}:{hole.line}  {hole.node}  — {hole.spec}")
```

`find_holes(*paths)` walks files/directories for node classes still stubbed with `raise Hole(...)`,
returning a `HoleRef` (file, node, the `Hole` message, line) for each. Unreadable or unparseable
files are skipped, so a broken file elsewhere never hides the rest. The CLI wraps it —
`python -m tensorsketch.canvas --holes src/` — and the [Studio](../guides/studio.md) counts them in its
toolbar (click to list them across the project).

## Write-back

A canvas edit changes the *wiring* — add an edge, reroute, add a node — never a node body. So
write-back edits the `GraphIR` and calls `reconstruct`:

```python
from tensorsketch.canvas import extract, reconstruct
from tensorsketch.canvas.ir import EdgeIR

ir = extract(source)
ir.edges.append(EdgeIR(source="Billing", target="Review", kind="sequential"))  # a canvas edit
new_source = reconstruct(source, ir)
```

`reconstruct` regenerates the graph definition and drops any now-redundant wiring statements it
folded in. Everything else — node classes and their bodies, imports, comments, unrelated code —
is preserved **byte-for-byte**, because nothing else is touched.

Write-back is **style-preserving**: it detects how the source authored its wiring and re-emits in
that same style, so the file reads the way you wrote it after a canvas edit:

| Authoring style | What write-back emits |
| --- | --- |
| **fluent** — `app = Graph(S).add(A).edge(x, y)` (one chained expression) | the same chain, re-indented for wherever it sits (module level or nested in a function) |
| **statement** — `g = Graph(S)` then separate `g.add(A)` / `g.edge(x, y)` lines | a bare `g = Graph(S)` plus one statement per wiring op |
| **arrow** — `a, b = g.nodes(A, B)` then `START >> a >> Router(...)` | `g.nodes(...)` handles and `>>` statements (linear runs merge into one `a >> b >> c` spine) |

All three styles render from the same ordered wiring walk, so the edges come out in identical
order whichever style is chosen — which is exactly what keeps the round-trip a list equality.

### Creating a node

Adding an edge is a pure-wiring change, but *creating* a node isn't — the new node needs a
`class X(Node)` to exist. So when the IR names a node the source never defined (the canvas palette
just made one), `reconstruct` **synthesizes an idiomatic stub** and inserts it above the graph
builder:

```python
ir = extract(source)
ir.nodes.append(NodeIR(name="Escalate",
                       inputs=[Port("query", "str")],
                       outputs=[Port("ticket", "str")],
                       has_hole=True))
ir.added.append("Escalate")
new_source = reconstruct(source, ir)
```

yields, written straight into your file:

```python
class Escalate(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        ticket: str

    async def run(self, ctx: Context, inp: In) -> Out:
        raise Hole("Escalate needs code")
```

The stub is born a [hole](../guides/getting-started.md) — its typed interface is declared, its body
is left for you to fill in code. A `from tensorsketch import Hole` is added if it isn't already imported.
Crucially, the stub re-extracts to the *exact* `NodeIR` the canvas sent (`has_hole=True` and all),
so the round-trip invariant still holds after a node is born on the canvas. Existing node classes
are never touched — only genuinely new names are generated.

The safety property is the **round-trip invariant**, enforced in CI:

```
extract(reconstruct(extract(code))) == extract(code)
```

Re-extracting reconstructed code yields the identical graph. Within a style, reconstruct still
*tidies* wiring without changing what the graph *is* — `.entry(x)` becomes `.edge(START, "x")`,
conditional mappings are normalized, and a fluent chain is cleanly re-indented. (Trade-off: a
comment sitting on a folded wiring statement moves with it; comments on nodes, imports, and
unrelated code are untouched.)

## Install

The engine is authoring-time tooling, so it's an optional extra (it isn't pulled into the
runtime):

```bash
pip install tensorsketch-core[canvas]
```

## The Studio

The [Studio](../guides/studio.md) is the visual canvas on top of this engine. A stdlib bridge
(`tensorsketch.canvas.server`, `python -m tensorsketch.canvas <file>`) serves `extract(file)` to a hand-drawn,
[Excalidraw](../design/decisions.md)-aesthetic frontend and applies `reconstruct` on every edit —
so wiring on the canvas writes straight into your code, bodies untouched.

### Layout lives in a sidecar, not the code

Where you *place* a node is presentation, not part of the graph — so it must never touch the code
(the code is the source of truth about the graph, not its picture). When you drag a node in the
Studio, its position is saved to a sidecar `‹file›.py.layout.json` next to the source. Positions
are optional and forgiving: a node with no saved position falls back to the automatic layered
layout, a stale entry for a deleted node is ignored, and losing the sidecar loses only the
arrangement — never the graph. This keeps the code clean while letting you arrange the canvas by
hand.

See the [architecture plan](../design/framework-design.md) for the full design.
