"""Observability — a native, vendor-neutral tracing spine.

TensorSketch traces itself through its own `Tracer` abstraction, not a third-party SDK. The core
ships a
real `InMemoryTracer` that captures a tree of `Span`s with rich attributes — timing, token usage,
cost, status, errors — which is exactly the raw material for evaluating latency, spend, and
correctness. OpenTelemetry, a JSON/file exporter, or a live Studio overlay are then just *adapters*
over the same spans (optional, added later), never a dependency you're forced into.

Tracing is **always available and zero-cost by default**: every `Context` carries a tracer
(`NoopTracer` unless you pass one to `invoke`/`stream`), the engine opens run and node spans, and
agents open model and tool spans — so a trace appears the moment you supply a real tracer.
"""

from .cost import DEFAULT_PRICES, estimate_cost
from .export import FileTracer, http_span_sink
from .tracing import (
    COST_USD,
    INPUT_TOKENS,
    MODEL,
    NODE,
    OUTPUT_TOKENS,
    RUN_ID,
    THREAD_ID,
    TOOL_NAME,
    InMemoryTracer,
    MultiTracer,
    NoopTracer,
    RecordingTracer,
    Span,
    Trace,
    Tracer,
)

__all__ = [
    "COST_USD",
    "DEFAULT_PRICES",
    "INPUT_TOKENS",
    "MODEL",
    "NODE",
    "OUTPUT_TOKENS",
    "RUN_ID",
    "THREAD_ID",
    "TOOL_NAME",
    "FileTracer",
    "InMemoryTracer",
    "MultiTracer",
    "NoopTracer",
    "RecordingTracer",
    "Span",
    "Trace",
    "Tracer",
    "estimate_cost",
    "http_span_sink",
]
