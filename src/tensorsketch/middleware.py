"""Middleware: composable interceptors around an agent's model and tool calls.

Middleware is TensorSketch's core **extensibility seam**. Each piece wraps a call — model or tool —
like
an onion: it runs code before, after, or *instead of* the next layer, so one uniform mechanism
covers retries, logging/tracing, caching, guardrails, cost accounting, and error handling. A
middleware overrides only the method it cares about; the default passes straight through.

    class Guardrail(Middleware):
        async def wrap_model(self, request, call_next):
            request.messages.append(system("Answer only in English."))
            return await call_next(request)

    agent = create_agent(provider, tools=tools, middleware=[RetryMiddleware(), Guardrail()])

The stack wraps the call **inside** the agent's durable `ctx.step`, so the whole wrapped call —
retries included — is journaled as one effect: a resume replays the recorded result and never
re-runs the model, the tool, or the middleware. Middleware runs outermost-first (the first in the
list is the outermost layer).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .core.context import Context
from .core.schema import Schema
from .messages import ToolCall
from .providers.base import Completion
from .tools import Tool

_T = TypeVar("_T")


@dataclass
class ModelRequest:
    """A single model invocation, passed down the middleware stack (mutate it to intercept)."""

    messages: list[Any]
    tools: list[Tool]
    ctx: Context
    node: str
    output_schema: type[Schema] | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolRequest:
    """A single tool invocation, passed down the middleware stack."""

    call: ToolCall
    ctx: Context
    node: str
    tool: Tool | None = None


ModelNext = Callable[[ModelRequest], Awaitable[Completion]]
ToolNext = Callable[[ToolRequest], Awaitable[Any]]


class Middleware:
    """Base class for a model/tool interceptor. Override a hook you need; others pass through."""

    async def wrap_model(self, request: ModelRequest, call_next: ModelNext) -> Completion:
        return await call_next(request)

    async def wrap_tool(self, request: ToolRequest, call_next: ToolNext) -> Any:
        return await call_next(request)


def compose_model(middleware: Sequence[Middleware], base: ModelNext) -> ModelNext:
    """Fold `middleware` around `base` into one handler (first item = outermost layer)."""
    handler = base
    for mw in reversed(list(middleware)):
        handler = _model_layer(mw, handler)
    return handler


def compose_tool(middleware: Sequence[Middleware], base: ToolNext) -> ToolNext:
    """Fold `middleware` around `base` into one handler (first item = outermost layer)."""
    handler = base
    for mw in reversed(list(middleware)):
        handler = _tool_layer(mw, handler)
    return handler


def _model_layer(mw: Middleware, next_handler: ModelNext) -> ModelNext:
    async def layer(request: ModelRequest) -> Completion:
        return await mw.wrap_model(request, next_handler)

    return layer


def _tool_layer(mw: Middleware, next_handler: ToolNext) -> ToolNext:
    async def layer(request: ToolRequest) -> Any:
        return await mw.wrap_tool(request, next_handler)

    return layer


# -- built-in middleware --------------------------------------------------------------------


class RetryMiddleware(Middleware):
    """Retry model and tool calls on error — the `on_model_error` / `on_tool_error` primitive.

    Retries up to `attempts` times on any exception in `retry_on`, with optional exponential
    `backoff` seconds between tries. Because the stack runs inside the durable step, a retry that
    eventually succeeds is journaled as the single result — a later resume never retries again.
    """

    def __init__(
        self,
        attempts: int = 3,
        *,
        backoff: float = 0.0,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        self.attempts = attempts
        self.backoff = backoff
        self.retry_on = retry_on

    async def wrap_model(self, request: ModelRequest, call_next: ModelNext) -> Completion:
        return await self._retry(lambda: call_next(request), request.ctx, "model")

    async def wrap_tool(self, request: ToolRequest, call_next: ToolNext) -> Any:
        return await self._retry(lambda: call_next(request), request.ctx, request.call.name)

    async def _retry(self, attempt_fn: Callable[[], Awaitable[_T]], ctx: Context, label: str) -> _T:
        last: BaseException | None = None
        for attempt in range(self.attempts):
            try:
                return await attempt_fn()
            except self.retry_on as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    await ctx.emit(
                        "retry", {"target": label, "attempt": attempt + 1, "error": str(exc)}
                    )
                    if self.backoff:
                        await asyncio.sleep(self.backoff * (2**attempt))
        assert last is not None
        raise last


class ObservabilityMiddleware(Middleware):
    """Emit `model_call` / `tool_call` start/end events (with duration) into the run's stream.

    A no-op when nobody is streaming (events flow through `ctx.emit`). This is the seam an OTel or
    logging exporter plugs into; it does not change results.
    """

    async def wrap_model(self, request: ModelRequest, call_next: ModelNext) -> Completion:
        await request.ctx.emit("model_call", {"phase": "start", "node": request.node})
        start = time.perf_counter()
        try:
            completion = await call_next(request)
        except Exception as exc:
            await request.ctx.emit("model_call", {"phase": "error", "error": str(exc)})
            raise
        await request.ctx.emit(
            "model_call",
            {
                "phase": "end",
                "node": request.node,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                "output_tokens": completion.usage.output_tokens,
            },
        )
        return completion

    async def wrap_tool(self, request: ToolRequest, call_next: ToolNext) -> Any:
        name = request.call.name
        await request.ctx.emit("tool_call", {"phase": "start", "tool": name})
        start = time.perf_counter()
        try:
            result = await call_next(request)
        except Exception as exc:
            await request.ctx.emit("tool_call", {"phase": "error", "tool": name, "error": str(exc)})
            raise
        await request.ctx.emit(
            "tool_call",
            {
                "phase": "end",
                "tool": name,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        return result
