"""The tracing spine: `Span`, the `Tracer` interface, and a built-in collecting tracer.

A **span** is one timed unit of work — a run, a node, a model call, a tool call, or anything a
node body wants to mark — carrying a name, a duration, a status, and free-form **attributes**
(model, tokens, cost, …). Spans nest into a tree via a parent link tracked through a `ContextVar`,
so nesting works correctly across `await` and parallel tasks without threading anything by hand.

The `Tracer` interface has one method — `span(...)`, a context manager. `NoopTracer` (the default)
does nothing at zero cost; `InMemoryTracer` collects finished spans into a `Trace` you can inspect,
aggregate (tokens, cost, latency), pretty-print, or hand to an exporter. Other tracers (an OTel
bridge, a file exporter) are just more implementations of the same interface.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# -- attribute vocabulary (vendor-neutral; the GenAI-style names map cleanly onto OTel later) --

RUN_ID = "tensorsketch.run_id"
THREAD_ID = "tensorsketch.thread_id"
NODE = "tensorsketch.node"
SUPERSTEP = "tensorsketch.superstep"
TOOL_NAME = "tensorsketch.tool.name"
# the arguments a tool was called with — for grading the payload
TOOL_ARGS = "tensorsketch.tool.args"
# the tool's return value (stringified) — for grading the effect
TOOL_RESULT = "tensorsketch.tool.result"
MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
COST_USD = "gen_ai.usage.cost_usd"

# Span kinds — coarse categories so aggregation ("total model time", "tool count") is easy.
RUN = "run"
NODE_KIND = "node"
MODEL_KIND = "model"
TOOL_KIND = "tool"
INTERNAL = "internal"

_current: ContextVar[Span | None] = ContextVar("loom_current_span", default=None)


@dataclass
class Span:
    """One timed unit of work in a trace."""

    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float  # monotonic (perf_counter) — for duration
    end: float | None = None
    started_at: float = 0.0  # wall-clock epoch seconds — for exporters/correlation
    status: str = "unset"  # "unset" | "ok" | "error"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, **attributes: Any) -> Span:
        """Attach attributes to the span (chainable): `span.set(**{MODEL: "gpt-4o"})`."""
        self.attributes.update(attributes)
        return self

    @property
    def duration_ms(self) -> float:
        return 0.0 if self.end is None else round((self.end - self.start) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view of the span — the unit an exporter writes."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class Tracer:
    """Creates spans. Subclass and override `span`; the base is a no-op."""

    @contextmanager
    def span(self, name: str, *, kind: str = INTERNAL, **attributes: Any) -> Iterator[Span]:
        yield Span(
            name=name,
            kind=kind,
            trace_id="",
            span_id="",
            parent_id=None,
            start=time.perf_counter(),
            attributes=dict(attributes),
        )


class NoopTracer(Tracer):
    """The default tracer — records nothing, at zero cost."""


#: A shared no-op instance so `Context` needs no per-run allocation when tracing is off.
NOOP = NoopTracer()


class RecordingTracer(Tracer):
    """Base for real tracers: owns the span lifecycle (timing, status, nesting) and hands each
    finished span to `_record`. Subclass and override `_record` to send spans anywhere — a list,
    a file, OpenTelemetry — without re-implementing the plumbing.
    """

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex

    @contextmanager
    def span(self, name: str, *, kind: str = INTERNAL, **attributes: Any) -> Iterator[Span]:
        parent = _current.get()
        span = Span(
            name=name,
            kind=kind,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex,
            parent_id=parent.span_id if parent is not None else None,
            start=time.perf_counter(),
            started_at=time.time(),
            attributes=dict(attributes),
        )
        token = _current.set(span)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.perf_counter()
            if span.status == "unset":
                span.status = "ok"
            _current.reset(token)
            self._record(span)

    def _record(self, span: Span) -> None:
        """Handle a finished span. Override in a subclass; the base discards it."""


class InMemoryTracer(RecordingTracer):
    """Collects finished spans into a `Trace`. The built-in tracer for tests, eval, and the CLI."""

    def __init__(self) -> None:
        super().__init__()
        self._spans: list[Span] = []

    def _record(self, span: Span) -> None:
        self._spans.append(span)

    @property
    def trace(self) -> Trace:
        """A view over the collected spans with tree + aggregate helpers."""
        return Trace(self.trace_id, list(self._spans))


class MultiTracer(RecordingTracer):
    """Fan one trace out to several sinks — e.g. write a file *and* collect in memory *and* feed a
    live overlay, all from a single run. It owns one span lifecycle (one `trace_id`, correct nesting
    and timing) and hands each finished span to every sink, so the tree is identical everywhere.

    A sink is either another `RecordingTracer` (`FileTracer`, `InMemoryTracer`, … — its `_record`
    consumes the span) or any `Callable[[Span], None]` — the seam a live viewer plugs into:

        collector = InMemoryTracer()
        with FileTracer("run.jsonl") as file_tracer:
            tracer = MultiTracer(collector, file_tracer, lambda s: feed.send(s.to_dict()))
            await app.invoke({...}, tracer=tracer)
        print(collector.trace.render())   # the same spans that were written to run.jsonl

    (`OTelTracer` isn't a `RecordingTracer` — it drives OTel's own live span context — so route
    OTel's own fan-out inside the OTel SDK rather than nesting it here.)
    """

    def __init__(self, *sinks: RecordingTracer | Callable[[Span], None]) -> None:
        super().__init__()
        self._consumers: list[Callable[[Span], None]] = []
        for sink in sinks:
            if isinstance(sink, RecordingTracer):
                # Share the one trace id (so a collecting sink's `Trace` matches its spans) and
                # drive only its consumer half — this MultiTracer owns the single lifecycle.
                sink.trace_id = self.trace_id
                self._consumers.append(sink._record)
            else:
                self._consumers.append(sink)

    def _record(self, span: Span) -> None:
        for consume in self._consumers:
            consume(span)


class Trace:
    """A collected span tree with convenience aggregates for cost, timing, and correctness."""

    def __init__(self, trace_id: str, spans: list[Span]) -> None:
        self.trace_id = trace_id
        self.spans = spans

    def roots(self) -> list[Span]:
        return [s for s in self.spans if s.parent_id is None]

    def children(self, span: Span) -> list[Span]:
        kids = [s for s in self.spans if s.parent_id == span.span_id]
        return sorted(kids, key=lambda s: s.start)

    def of_kind(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    @property
    def duration_ms(self) -> float:
        return max((s.duration_ms for s in self.roots()), default=0.0)

    @property
    def input_tokens(self) -> int:
        return sum(int(s.attributes.get(INPUT_TOKENS, 0)) for s in self.spans)

    @property
    def output_tokens(self) -> int:
        return sum(int(s.attributes.get(OUTPUT_TOKENS, 0)) for s in self.spans)

    @property
    def cost_usd(self) -> float:
        return round(sum(float(s.attributes.get(COST_USD, 0.0)) for s in self.spans), 6)

    @property
    def errors(self) -> list[Span]:
        return [s for s in self.spans if s.status == "error"]

    def summary(self) -> dict[str, Any]:
        """The headline numbers — what an eval or a dashboard wants."""
        return {
            "duration_ms": self.duration_ms,
            "spans": len(self.spans),
            "model_calls": len(self.of_kind(MODEL_KIND)),
            "tool_calls": len(self.of_kind(TOOL_KIND)),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "errors": len(self.errors),
        }

    def render(self) -> str:
        """An indented tree of the trace, one line per span, for logs and the CLI."""
        lines: list[str] = []

        def walk(span: Span, depth: int) -> None:
            mark = "✗" if span.status == "error" else " "
            detail = _detail(span)
            lines.append(f"{mark} {'  ' * depth}{span.name} ({span.duration_ms:.2f}ms){detail}")
            for child in self.children(span):
                walk(child, depth + 1)

        for root in sorted(self.roots(), key=lambda s: s.start):
            walk(root, 0)
        return "\n".join(lines)


def _detail(span: Span) -> str:
    bits = []
    if MODEL in span.attributes:
        bits.append(str(span.attributes[MODEL]))
    if OUTPUT_TOKENS in span.attributes:
        bits.append(f"{span.attributes[OUTPUT_TOKENS]} out-tok")
    if span.attributes.get(COST_USD):
        bits.append(f"${span.attributes[COST_USD]:.6f}")
    if TOOL_NAME in span.attributes:
        bits.append(str(span.attributes[TOOL_NAME]))
    return f"  [{', '.join(bits)}]" if bits else ""
