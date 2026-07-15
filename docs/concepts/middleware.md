# Middleware

Middleware is TensorSketch's **extensibility seam** for agents. Each piece wraps a call — a model call or
a tool call — like an onion: it runs code before, after, or *instead of* the next layer. One
uniform mechanism covers what production agents need — **retries, tracing, caching, guardrails,
cost accounting, error handling** — without touching the agent loop.

```python
from tensorsketch import create_agent, RetryMiddleware, ObservabilityMiddleware

agent = create_agent(
    provider,
    tools=tools,
    middleware=[RetryMiddleware(attempts=3), ObservabilityMiddleware()],
)
```

## Writing one

Subclass `Middleware` and override the hook you care about — `wrap_model`, `wrap_tool`, or both.
Each receives a request and a `call_next` to invoke the rest of the stack. The default passes
straight through, so you only implement what you need.

```python
from tensorsketch import Middleware
from tensorsketch.messages import system

class Guardrail(Middleware):
    async def wrap_model(self, request, call_next):
        request.messages.append(system("Answer in one short sentence."))
        return await call_next(request)   # ← the rest of the stack, then the real model call
```

A middleware can:

- **observe** — time the call, log it, count tokens (`request` / the returned `Completion`);
- **modify** — mutate `request.messages` / `request.tools` / `request.options` before `call_next`,
  or transform the result after;
- **short-circuit** — return a value without calling `call_next` (a cache hit, a blocked call);
- **handle errors** — wrap `call_next` in `try/except` to retry, fall back, or annotate.

`request` is a `ModelRequest` (`messages`, `tools`, `output_schema`, `options`, `ctx`, `node`) or
a `ToolRequest` (`call`, `tool`, `ctx`, `node`). `ctx.emit(...)` from either lets a middleware
push events into the run's [stream](streaming.md).

## Ordering and durability

The list wraps **outermost-first** — `[A, B]` means `A` sees the call before `B` and the result
after. The whole stack runs **inside** the agent's durable `ctx.step`, so a wrapped call is
journaled as one effect: on [resume](durability.md), the recorded result is replayed and the
middleware, model, and tools are **not re-run**. A retry that eventually succeeds is journaled as
that single success.

## Built-ins

### `RetryMiddleware`

Retries model and tool calls on error — the `on_model_error` / `on_tool_error` primitive.

```python
RetryMiddleware(attempts=3, backoff=0.5, retry_on=(TimeoutError, ConnectionError))
```

Retries up to `attempts` times on any exception in `retry_on` (default: any `Exception`), with
optional exponential `backoff` seconds. Each retry emits a `retry` event.

### `ObservabilityMiddleware`

Emits `model_call` / `tool_call` **start / end / error** events (with duration and token counts)
into the run's stream — a no-op when nobody is streaming. This is the seam a logging or
OpenTelemetry exporter plugs into; it never changes results.

```
model_call  {'phase': 'start', 'node': 'agent'}
model_call  {'phase': 'error', 'error': 'rate limited (503)'}
retry       {'target': 'model', 'attempt': 1, 'error': 'rate limited (503)'}
model_call  {'phase': 'start', 'node': 'agent'}
model_call  {'phase': 'end', 'node': 'agent', 'duration_ms': 0.8, 'output_tokens': 12}
tool_call   {'phase': 'start', 'tool': 'weather'}
tool_call   {'phase': 'end', 'tool': 'weather', 'duration_ms': 0.3}
```

See [`examples/middleware_retry.py`](../../examples/middleware_retry.py) for the whole thing
running offline.

## Roadmap

Node/run-level middleware, and packaging middleware + tools + providers as discoverable
**plugins** (entry-point registries), build on this same seam. See the [roadmap](../design/roadmap.md).
