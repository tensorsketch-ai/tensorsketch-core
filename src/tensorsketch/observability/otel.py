"""OpenTelemetry adapter — export TensorSketch's native traces to OTel, when you want it.

`pip install tensorsketch-core[otel]`. This is the whole point of the vendor-neutral design: OTel is
one
`Tracer` implementation, not a dependency baked into the core. Configure OTel however you like
(any exporter/collector), then hand TensorSketch an `OTelTracer` and every TensorSketch span becomes
an OTel span.

    from opentelemetry import trace
    from tensorsketch.observability.otel import OTelTracer

    tracer = OTelTracer()                      # uses the globally-configured OTel tracer
    await app.invoke({...}, tracer=tracer)     # run/node/model/tool spans → your OTel backend

Spans nest through OTel's own context, and each TensorSketch span's attributes (model, tokens, cost,
…)
are copied onto the OTel span. The GenAI-style attribute names TensorSketch already uses map
straight onto
OTel's GenAI semantic conventions.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .tracing import INTERNAL, Span, Tracer


class OTelTracer(Tracer):
    """Bridges TensorSketch spans to OpenTelemetry. Pass an OTel tracer, or use the global
    default."""

    def __init__(self, otel_tracer: Any = None) -> None:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        self._otel = otel_tracer or trace.get_tracer("tensorsketch")
        self._status = Status
        self._error_code = StatusCode.ERROR

    @contextmanager
    def span(self, name: str, *, kind: str = INTERNAL, **attributes: Any) -> Iterator[Span]:
        with self._otel.start_as_current_span(name) as otel_span:
            # A throwaway TensorSketch span is the handle callers `.set(...)` on; we flush its
            # attributes
            # onto the OTel span at close (OTel manages the real ids, timing, and parent context).
            carrier = Span(name=name, kind=kind, trace_id="", span_id="", parent_id=None, start=0.0)
            carrier.attributes.update(attributes)
            otel_span.set_attribute("tensorsketch.kind", kind)
            try:
                yield carrier
            except BaseException as exc:
                otel_span.record_exception(exc)
                otel_span.set_status(self._status(self._error_code))
                raise
            finally:
                for key, value in carrier.attributes.items():
                    otel_span.set_attribute(key, _otel_value(value))


def _otel_value(value: Any) -> Any:
    """Coerce an attribute to an OTel-acceptable type (primitives pass; everything else → str)."""
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)
