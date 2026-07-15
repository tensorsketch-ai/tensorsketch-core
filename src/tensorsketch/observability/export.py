"""Exporters — tracers that stream finished spans somewhere durable.

`FileTracer` writes one JSON object per span (JSON Lines) as each span closes — a dependency-free,
`grep`/`jq`-friendly trace log. It's the same `Tracer` interface as everything else, so it drops
into `invoke(..., tracer=...)` unchanged:

    from tensorsketch.observability.export import FileTracer

    with FileTracer("run.jsonl") as tracer:
        await app.invoke({...}, tracer=tracer)
    # each line of run.jsonl is one span: {"name": "model_call", "duration_ms": ..., ...}
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO, TYPE_CHECKING, Any

from .tracing import RecordingTracer, Span

if TYPE_CHECKING:
    import queue


class FileTracer(RecordingTracer):
    """A tracer that appends each finished span to a file (or stream) as a JSON line."""

    def __init__(self, destination: str | Path | IO[str]) -> None:
        super().__init__()
        if isinstance(destination, str | Path):
            self._stream: IO[str] = open(destination, "a", encoding="utf-8")  # noqa: SIM115
            self._owns = True
        else:
            self._stream = destination
            self._owns = False

    def _record(self, span: Span) -> None:
        self._stream.write(json.dumps(span.to_dict(), default=str) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._owns:
            self._stream.close()

    def __enter__(self) -> FileTracer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def http_span_sink(url: str, *, timeout: float = 2.0) -> Callable[[Span], None]:
    """A trace sink that POSTs each finished span (as JSON) to an HTTP endpoint, off the hot path.

    Returns a `Callable[[Span], None]` for `MultiTracer` — the seam a live viewer plugs into. To
    feed the TensorSketch Studio live overlay from your running agent:

        from tensorsketch import MultiTracer, InMemoryTracer
        from tensorsketch.observability.export import http_span_sink

        tracer = MultiTracer(InMemoryTracer(), http_span_sink("http://127.0.0.1:8765/api/trace"))
        await agent.invoke(inputs, tracer=tracer)

    Delivery runs on a background daemon thread and drops silently if the endpoint is unreachable —
    a live overlay is a *view*, not a system of record, so it must never slow or break the run.
    """
    import queue
    import threading
    import urllib.request

    pending: queue.Queue[dict[str, Any]] = queue.Queue()

    def worker() -> None:
        while True:
            record = pending.get()
            try:
                body = json.dumps(record, default=str).encode()
                request = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(request, timeout=timeout).close()
            except (OSError, ValueError, TypeError):
                pass  # endpoint down or span not serializable — never disturb the traced run

    threading.Thread(target=worker, name="tensorsketch-span-sink", daemon=True).start()

    def sink(span: Span) -> None:
        pending.put(span.to_dict())

    return sink
