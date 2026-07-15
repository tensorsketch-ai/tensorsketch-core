"""Where a trial *runs* — behind a seam, so isolation can grow without touching the harness.

The security-relevant invariant holds in every sandbox: the sandbox only *runs the agent and
returns artifacts*; the graders run afterward, in the harness, never inside the agent's execution.
So an agent can't tamper with its own grade the way agents have gamed benchmarks that graded
in-process.

`InProcessSandbox` (the default) runs the trial in this process under a fresh `InMemoryTracer` —
right for LLM / tool / trajectory evaluation, where the `Case` controls the tools and environment.
Heavier isolation (a subprocess, a container, a remote runner) is a future `Sandbox` implementation
behind this same `run(...)` seam.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..core.graph import CompiledGraph
from ..observability.tracing import InMemoryTracer
from .case import Case, Trial


class Sandbox(Protocol):
    """Runs one trial and returns its artifacts (output, transcript, error). No grading here."""

    async def run(self, target: CompiledGraph[Any], case: Case, env: Any) -> Trial: ...


class InProcessSandbox:
    """Run the trial in-process under a fresh tracer. The default sandbox."""

    async def run(self, target: CompiledGraph[Any], case: Case, env: Any) -> Trial:
        tracer = InMemoryTracer()
        output: Any = None
        error: str | None = None
        try:
            output = await target.invoke(case.inputs, tracer=tracer)
        except Exception as exc:  # a failed run is a datum, not a harness crash
            error = f"{type(exc).__name__}: {exc}"
        return Trial(case=case, output=output, trace=tracer.trace, env=env, error=error)
