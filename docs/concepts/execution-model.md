# Execution model

TensorSketch runs graphs with a **BSP (bulk-synchronous parallel)** scheduler — the same model behind
Google's Pregel and LangGraph's runtime. It's a small idea with big payoffs: cycles,
deterministic parallelism, and clean checkpoint boundaries all fall out of it for free.

![Superstep diagram placeholder](../images/supersteps.png)
<!-- TODO: diagram — three phases (plan → execute → barrier) with the active set flowing between steps -->

## Supersteps

Execution proceeds in discrete **supersteps**. Each superstep has three phases:

1. **Plan.** The scheduler knows the *active set* — the nodes to run this step. Initially that's
   the entry node; thereafter it's whoever the previous step's nodes named as successors.
2. **Execute.** Every active node runs **in parallel**, and each one reads the *same immutable
   snapshot* of state taken at the start of the step. A node cannot observe another node's
   writes mid-step, so there are no read/write races by construction.
3. **Barrier.** All writes collected this step are folded into their channels via reducers,
   atomically. Then each node's successors are computed from the now-consistent state, and they
   become the next step's active set.

The loop ends when the active set is empty — no node named a successor, so the graph has
settled.

## Why this model

- **Cycles are natural.** A node (or a conditional edge) can name a predecessor as a successor.
  The loop just runs another superstep. No special "loop node" needed — see
  [`counting_loop.py`](../../examples/counting_loop.py).
- **Parallelism is deterministic.** Sibling nodes in a fan-out all read the *same* snapshot and
  merge at the barrier through reducers. The result doesn't depend on which sibling finished
  first. (This is also why a `LastValue` channel refuses two writes in one step — that *would*
  be order-dependent.)
- **The barrier is a checkpoint boundary.** At each barrier the state is consistent. That's
  exactly where the [durable journal](durability.md) snapshots channels, so a run can resume or
  fork from any superstep.

## Dynamic fan-out (`Send`)

Static fan-out runs a fixed set of nodes. To run **one node many times** — a worker per item,
each with its own input — a router returns a list of [`Send`](nodes-and-graphs.md#dynamic-fan-out-with-send)s.
Each `Send` becomes its own unit in the next superstep (its payload overlaid on the shared
snapshot); they all merge at the barrier through a reducer channel. It's the same three-phase
model — the *plan* phase simply schedules N instances of one node instead of one. Pending sends
ride the checkpoint, and each instance journals its effects under a distinct key, so a crash
mid-fan-out resumes and replays completed workers exactly once.

## Parallel execution details

Active nodes run under an `asyncio.TaskGroup` — **structured concurrency**. If one node raises,
its siblings are cancelled cleanly. A single failure propagates as the real exception (a
`Hole`, a `NodeError`, ...), not an opaque wrapper.

Because a superstep is `await`ed as a whole, independent LLM/tool calls in a fan-out overlap in
wall-clock time — parallelism you get from the graph's shape, without threading anything
yourself.

## The recursion limit

A guarded loop is expected to terminate. If it doesn't — a routing function that never returns
`END`, say — the scheduler would run forever. The **recursion limit** is the safety net:

```python
await app.invoke(input, max_steps=25)   # default 25
# GraphRecursionError: exceeded 25 supersteps without halting (still active: ['Spin']) ...
```

Raise `max_steps` only once you're confident the loop's exit condition can be met.

## A worked trace

For the counting loop (`limit=3`, starting `count=0`):

| Superstep | Active | Reads `count` | Writes | Successor |
|-----------|--------|---------------|--------|-----------|
| 0 | `Tick` | 0 | `count=1`, append log | `Tick` (1 < 3) |
| 1 | `Tick` | 1 | `count=2`, append log | `Tick` (2 < 3) |
| 2 | `Tick` | 2 | `count=3`, append log | `END` (3 ≥ 3) |
| — | ∅ | | | halt |

Three supersteps, then the active set is empty and `invoke` returns the final state. The `log`
channel (a `Reducer(add)`) accumulated one line per step.

## Roadmap

The engine deliberately schedules *opaque processes* over channels, decoupled from the typed node
layer. The [durable journal](durability.md) and [namespaced streaming](streaming.md) already build
on that seam. Still ahead: a swappable message transport so the *same* graph runs single-process or
distributed across a gRPC host/workers, and a Rust hot-path core behind the same interface. See the
[architecture plan](../design/framework-design.md).
