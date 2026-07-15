# Tracing & observability

TensorSketch traces itself through its **own** tracing abstraction — not a third-party SDK. That's a
deliberate choice: you shouldn't have to adopt OpenTelemetry (or anything) to see what your agent
did, how long it took, and what it cost. The core ships a real, built-in tracer; OpenTelemetry, a
file/JSON exporter, or a live Studio overlay are then just **adapters** over the same spans —
optional, never required.

## The shape of it

A **span** is one timed unit of work — a run, a node, a model call, a tool call, or anything you
mark yourself — with a duration, a status, and free-form **attributes** (model, tokens, cost, …).
Spans nest into a **tree**. Tracing is always available and **zero-cost by default**: every run
carries a tracer (`NoopTracer` unless you pass one), so a trace appears the moment you supply a
real one.

```python
from tensorsketch import InMemoryTracer

tracer = InMemoryTracer()
out = await agent.invoke({"query": "what is tensorsketch?"}, tracer=tracer)

print(tracer.trace.render())
print(tracer.trace.summary())
```

```
  run (12.4ms)
    agent (11.9ms)
      model_call (7.1ms)  [claude-sonnet-4-6, 120 out-tok, $0.004800]
      tool_call (0.3ms)   [search]
      model_call (4.0ms)  [claude-sonnet-4-6, 120 out-tok, $0.005400]

{'duration_ms': 12.4, 'spans': 5, 'model_calls': 2, 'tool_calls': 1,
 'input_tokens': 2200, 'output_tokens': 240, 'cost_usd': 0.0102, 'errors': 0}
```

The engine opens the **run** and **node** spans; agents open **model** and **tool** spans and
record the model id, token usage, and estimated cost. A model call that's replayed from the
[durable journal](durability.md) on resume does no work, so it produces no span — the trace always
reflects what actually ran.

## What you get for evaluation

`trace.summary()` gives the headline numbers; the `Trace` also exposes them individually, which is
exactly what a cost/latency/correctness eval consumes:

| | |
|---|---|
| `trace.duration_ms` | wall time of the run |
| `trace.input_tokens` / `output_tokens` | token totals across model calls |
| `trace.cost_usd` | summed estimated cost |
| `trace.errors` | spans that failed (status `error`, with the exception) |
| `trace.of_kind("model")` / `("tool")` | every model / tool span |
| `trace.render()` | the indented tree, for logs and the CLI |

### Cost

Cost is estimated from token usage and the model id via a small, **overridable** price table
(`estimate_cost`, `DEFAULT_PRICES` — USD per million tokens). Pricing changes and every deployment
differs, so pass your own table rather than trusting the default as gospel.

## Trace your own work

Inside any node body, `ctx.span(...)` marks a sub-step — it nests automatically under the node:

```python
async def run(self, ctx, inp):
    with ctx.span("parse-invoice"):
        data = parse(inp.document)
    with ctx.span("score", model="scorer-v2"):
        return self.Out(score=await score(data))
```

## Exporters

The sink is just another `Tracer`, so switching where spans go changes nothing else.

### File (JSON Lines) — built in, no dependencies

`FileTracer` streams one JSON object per span to a file as each span closes — a durable,
`grep`/`jq`-friendly trace log:

```python
from tensorsketch.observability.export import FileTracer

with FileTracer("run.jsonl") as tracer:
    await app.invoke({...}, tracer=tracer)
```

```json
{"name": "model_call", "kind": "model", "duration_ms": 7.1, "status": "ok",
 "attributes": {"gen_ai.request.model": "claude-sonnet-4-6", "gen_ai.usage.cost_usd": 0.0048}, ...}
```

### OpenTelemetry — optional adapter

If you *do* use OTel, it's one import away — never a requirement. Configure OTel however you like,
then hand TensorSketch an `OTelTracer`; every TensorSketch span becomes an OTel span (nesting and attributes
preserved, and TensorSketch's GenAI-style attribute names map onto OTel's GenAI conventions):

```bash
pip install tensorsketch-core[otel]
```

```python
from tensorsketch.observability.otel import OTelTracer

await app.invoke({...}, tracer=OTelTracer())   # uses your globally-configured OTel tracer
```

### Fan-out to several sinks — `MultiTracer`

You often want more than one destination for the same run: keep a file log *and* an in-memory
`Trace` to assert on *and* a live feed for a viewer. `MultiTracer` owns a single span lifecycle (one
`trace_id`, correct nesting and timing) and hands each finished span to every sink, so the tree is
identical everywhere — no double-counting, no drift between destinations:

```python
from tensorsketch import InMemoryTracer, MultiTracer, FileTracer

collector = InMemoryTracer()
with FileTracer("run.jsonl") as file_tracer:
    tracer = MultiTracer(collector, file_tracer, lambda span: feed.send(span.to_dict()))
    await app.invoke({...}, tracer=tracer)

print(collector.trace.render())   # exactly the spans that were written to run.jsonl
```

A sink is either another `RecordingTracer` (its `_record` consumes the span) or any
`Callable[[Span], None]` — the callable form is the seam a live overlay plugs into (below).
(`OTelTracer` drives OTel's *own* live span context, so it isn't a `RecordingTracer` sink; use the
OTel SDK's exporter pipeline to fan OTel out.)

### Write your own

`Tracer` has one method — `span(name, *, kind, **attributes)`, a context manager. For a collecting
or streaming sink, subclass `RecordingTracer` and override `_record(span)` — it owns the lifecycle
(timing, status, nesting) and hands you each finished `Span` (with `.to_dict()`). That's all
`FileTracer` is. For a bridge to a system with its own context (like OTel), implement `Tracer`
directly.

See [`examples/tracing.py`](../../examples/tracing.py) for the whole thing running offline.

## Relationship to streaming and middleware

Three complementary seams:

- **Tracing** (this page) — an always-on tree of timed spans for after-the-fact analysis.
- **[Streaming](streaming.md)** — live events as a run progresses (for UIs).
- **[Middleware](middleware.md)** — intercept model/tool calls (retries, guardrails);
  `ObservabilityMiddleware` bridges middleware into the live event stream.

## Roadmap

The [live trace overlay in Studio](../guides/studio.md#live-trace-overlay) is built on this same
span model — feed it with `MultiTracer(..., http_span_sink(url))`. Richer per-provider cost/latency
metrics are still to come. The [eval harness](evaluation.md) already consumes traces directly for
cost / latency / correctness.
