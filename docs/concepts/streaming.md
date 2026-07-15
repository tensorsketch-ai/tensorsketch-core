# Streaming

`invoke` gives you the final state. `stream` gives you the run *as it happens* — a live
sequence of typed events: nodes starting and finishing, the merged state after each superstep,
and whatever a node chooses to surface itself. It's how you drive a progress UI, show tokens as
they arrive, or feed a live canvas trace.

```python
async for event in app.stream({"query": "..."}):
    print(event.seq, event.type, event.node, event.data)
```

## The event

Every event is an `Event`:

| field | meaning |
|---|---|
| `seq` | monotonic per-run cursor (0-based) — the handle for [replay](#resumable-replay) |
| `run_id` | the `stream`/`invoke` call it came from |
| `thread_id` | the durable run key (empty when durability is off) |
| `superstep` | the BSP superstep the event belongs to |
| `node` | the node it's about, or `None` for run-level events |
| `type` | the event type (below) |
| `data` | the payload |

### Event types

| `type` | when | `data` |
|---|---|---|
| `run_start` | the run begins | `{}` |
| `node_start` | a node begins executing | `{}` |
| `node_end` | a node finishes | `{"writes": {...}}` — what it wrote to state |
| `values` | after a barrier | `{"state": {...}}` — the merged state |
| `run_end` | the run settles | `{}` |
| *(custom)* | a node called `ctx.emit(...)` | whatever the node passed |

Because every event carries `run_id`, `thread_id`, and `node`, a consumer can separate **lanes**
in a multi-agent run — e.g. render each agent's tokens in its own column from a single stream.

## Emitting from a node

Call `ctx.emit(type, data)` inside a node body to push a custom event. It's a no-op if nobody is
streaming, so it's safe to leave in.

```python
class Search(Node):
    class In(Schema):  query: str
    class Out(Schema): hits: list[str]

    async def run(self, ctx, inp):
        await ctx.emit("status", {"stage": "searching"})
        results = await do_search(inp.query)
        await ctx.emit("status", {"stage": "done", "count": len(results)})
        return self.Out(hits=results)
```

This is the seam through which **LLM token deltas** will flow once provider nodes land — a
streaming model call emits a `token` event per chunk.

## Backpressure

The stream is bounded (`stream(..., buffer=256)`). If the consumer falls behind and the buffer
fills, `emit` waits — which pauses the producing node. So a slow consumer naturally throttles the
run instead of blowing up memory. If the consumer stops iterating early, the run is cancelled
cleanly (structured concurrency).

## Errors

If the run raises, the `async for` delivers every event emitted up to that point, then raises the
original exception. `run_end` is only emitted on success.

## Resumable replay

When you stream a **durable** run (with a `thread_id` and `backend`), every event is also
persisted. A consumer that dropped can catch up from where it left off using the `seq` cursor:

```python
# stream and remember the last seq you saw ...
async for event in app.stream(inp, thread_id="t", backend=backend):
    last = event.seq

# ... later, replay everything after that point
async for event in app.replay("t", backend, since=last + 1):
    handle(event)
```

`replay` reads the persisted event log for a thread and yields events with `seq >= since`, in
order — for a completed or in-progress run.

## Roadmap

Live-tailing an in-progress run from another process (merging `replay` catch-up with the live
`stream`), and token-level streaming from provider nodes, build on this foundation. See the
[architecture plan](../design/framework-design.md).
