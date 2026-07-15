"""The per-run execution context handed to every node.

`Context` is how a node body talks to the runtime without importing it. It carries identity
(`run_id`, `thread_id`), the current `superstep`, and — most importantly — `step()`, the one
durability primitive an author needs: wrap a side effect in `ctx.step(...)` and the runtime
records its result so it is **never re-executed on resume**.

A fresh `Context` is created for each node in each superstep, so `step()` can key effects
deterministically by (superstep, node, call-order).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ..observability.tracing import INTERNAL, NOOP, Span, Tracer

if TYPE_CHECKING:
    from ..runtime.checkpoint import Backend
    from ..runtime.streaming import EmitFn

T = TypeVar("T")


class Context:
    """Read-mostly handle passed to `Node.run`.

    Attributes:
        run_id: Identifier for this `invoke` call — threads a run together across logs/traces.
        thread_id: The durable conversation/run key. Resuming with the same `thread_id` continues
            from the last checkpoint. Empty string when durability is off.
        superstep: The current BSP superstep index (0-based), incrementing once per barrier.
        node: The name of the node this context belongs to.
    """

    def __init__(
        self,
        *,
        run_id: str,
        thread_id: str,
        superstep: int,
        node: str,
        backend: Backend | None = None,
        emit: EmitFn | None = None,
        tracer: Tracer = NOOP,
        instance: str = "",
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.superstep = superstep
        self.node = node
        self.tracer = tracer
        self._backend = backend
        self._emit = emit
        self._calls = 0
        # Disambiguates parallel instances of the *same* node in one superstep (dynamic fan-out).
        # Empty for a normally-scheduled node, so its effect-journal keys are unchanged.
        self._instance = instance

    def span(
        self, name: str, *, kind: str = INTERNAL, **attributes: Any
    ) -> AbstractContextManager[Span]:
        """Open a trace span around a piece of work in a node body::

            with ctx.span("parse-invoice"):
                ...

        A no-op unless a real tracer was passed to `invoke`/`stream`. Spans nest automatically.
        """
        return self.tracer.span(name, kind=kind, **attributes)

    async def emit(self, type: str, data: Mapping[str, Any] | None = None) -> None:
        """Emit a custom event into the run's stream (a no-op when nobody is streaming).

        Use this to surface progress from inside a node body — for example, LLM token deltas or
        a "searching..." status. The event is namespaced with this node's identity automatically.
        """
        if self._emit is not None:
            await self._emit(type, self.node, self.superstep, data or {})

    async def step(
        self,
        name: str,
        fn: Callable[[], Awaitable[T]],
        *,
        idempotency_key: str | None = None,
    ) -> T:
        """Run a side effect **at most once**, memoizing its result in the durable journal.

        The first time this runs, `fn` is awaited and its result recorded. If the run later
        resumes and reaches this call again, the recorded result is returned **without calling
        `fn`** — so an LLM/tool/API call is never duplicated and its output never drifts.

        `fn` must be a zero-argument callable returning an awaitable (e.g. `lambda: client.chat(
        ...)`). Wrap synchronous work with `asyncio.to_thread`.

        Args:
            name: A short label for the step (part of its journal key; keep it stable).
            fn: The effect to run once.
            idempotency_key: Override the automatic key to dedupe an effect across the whole
                run (e.g. a payment id), rather than per (superstep, node, call-order).

        Returns:
            The effect's result — freshly computed, or replayed from the journal.
        """
        if idempotency_key is not None:
            key = f"idem:{idempotency_key}"
        else:
            index = self._calls
            self._calls += 1
            scope = f"{self.node}#{self._instance}" if self._instance else self.node
            key = f"{self.superstep}:{scope}:{index}:{name}"

        if self._backend is not None:
            found, recorded = self._backend.lookup_effect(self.thread_id, key)
            if found:
                return cast(T, recorded)

        result = await fn()

        if self._backend is not None:
            self._backend.record_effect(self.thread_id, key, result)
        return result
