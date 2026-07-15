"""Where results go — a sink you point at your store. TensorSketch emits; it never owns the
database.

Every eval result serializes to a plain JSON record (`Report.to_dict()`, `TrialResult.to_dict()`),
and a `Reporter` is anything that can take one such record and put it somewhere: a file, stdout,
an HTTP endpoint, a warehouse, any database. This mirrors the tracing exporters exactly — the
transcript side already has `FileTracer` / `OTelTracer` / a custom `RecordingTracer._record`; this
is the same seam for the *scores*.

Connecting a database is a few lines — implement `emit` (sync or async):

    class PostgresReporter:
        def __init__(self, pool):
            self.pool = pool

        async def emit(self, record):
            await self.pool.execute("insert into evals(doc) values($1)", record)

or use `CallbackReporter(lambda r: my_table.insert(r))`.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """A result sink. `emit` takes one JSON-able record; it may be sync or return an awaitable."""

    def emit(self, record: dict[str, Any]) -> Any: ...


class JsonlReporter:
    """Append each record as one JSON line — a durable, `jq`-friendly eval log (no dependencies)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def emit(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


class CallbackReporter:
    """Adapt any function into a `Reporter` — the quickest bridge to your own store or API."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def emit(self, record: dict[str, Any]) -> Any:
        return self._fn(record)


class MultiReporter:
    """Fan one record out to several sinks — e.g. persist to a store *and* feed a `DriftMonitor`
    from a single hand-off: `MultiReporter(JsonlReporter("online.jsonl"), drift_monitor)`."""

    def __init__(self, *reporters: Reporter) -> None:
        self._reporters = reporters

    async def emit(self, record: dict[str, Any]) -> None:
        for reporter in self._reporters:
            await deliver(reporter, record)


async def deliver(reporter: Reporter, record: dict[str, Any]) -> None:
    """Emit a record, awaiting the sink if it's async."""
    result = reporter.emit(record)
    if inspect.isawaitable(result):
        await result
