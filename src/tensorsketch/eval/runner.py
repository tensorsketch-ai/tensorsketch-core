"""The harness: run every case x trial, grade each trial, aggregate into a `Report`.

`evaluate` is the offline entry point — point it at an agent and a suite of goldens and it returns
a `Report` with the metrics the essay calls for: task-completion rate, **pass@k** (succeeds at
least once) and **pass^k** (succeeds every time), plus cost/latency and a per-grader breakdown.
`Report.require(...)` turns it into a CI gate. Trajectories are captured per trial (fresh
environment each time) so trials never cross-contaminate.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from ..core.graph import CompiledGraph
from .case import Case, Grade, Grader, Suite, Trial
from .reporters import Reporter, deliver
from .sandbox import InProcessSandbox, Sandbox


@dataclass
class TrialResult:
    """One graded trial: its grades and operational footprint."""

    grades: list[Grade]
    cost_usd: float
    latency_ms: float
    error: str | None = None

    @property
    def passed(self) -> bool:
        """A trial succeeds only if it didn't error and every grader passed."""
        return self.error is None and all(g.passed for g in self.grades)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            "grades": [g.to_dict() for g in self.grades],
        }


@dataclass
class CaseResult:
    """All trials of one case, with the multi-run outcome metrics."""

    name: str
    trials: list[TrialResult]

    @property
    def k(self) -> int:
        return len(self.trials)

    @property
    def passes(self) -> int:
        return sum(t.passed for t in self.trials)

    @property
    def success_rate(self) -> float:
        return self.passes / self.k if self.k else 0.0

    @property
    def pass_at_k(self) -> float:
        """Succeeded at least once across k trials (retry-friendly view)."""
        return 1.0 if self.passes >= 1 else 0.0

    @property
    def pass_pow_k(self) -> float:
        """Succeeded on every one of k trials (consistency view — penalizes flakiness)."""
        return 1.0 if self.k and self.passes == self.k else 0.0

    @property
    def mean_cost(self) -> float:
        return fmean(t.cost_usd for t in self.trials) if self.trials else 0.0

    @property
    def mean_latency(self) -> float:
        return fmean(t.latency_ms for t in self.trials) if self.trials else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "passes": self.passes,
            "pass_at_k": self.pass_at_k,
            "pass_pow_k": self.pass_pow_k,
            "mean_cost_usd": round(self.mean_cost, 6),
            "mean_latency_ms": round(self.mean_latency, 1),
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass
class Report:
    """The result of an eval run — aggregate metrics, a rendered view, and a CI gate."""

    suite: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        """Fraction of all trials (across all cases) that succeeded."""
        total = sum(c.k for c in self.cases)
        passed = sum(c.passes for c in self.cases)
        return passed / total if total else 0.0

    @property
    def pass_at_k(self) -> float:
        return fmean(c.pass_at_k for c in self.cases) if self.cases else 0.0

    @property
    def pass_pow_k(self) -> float:
        return fmean(c.pass_pow_k for c in self.cases) if self.cases else 0.0

    @property
    def mean_cost(self) -> float:
        return fmean(c.mean_cost for c in self.cases) if self.cases else 0.0

    @property
    def mean_latency(self) -> float:
        return fmean(c.mean_latency for c in self.cases) if self.cases else 0.0

    def grader_breakdown(self) -> dict[str, float]:
        """Pass rate per grader name across every trial — which criterion fails most."""
        totals: dict[str, list[bool]] = {}
        for case in self.cases:
            for trial in case.trials:
                for grade in trial.grades:
                    totals.setdefault(grade.name, []).append(grade.passed)
        return {name: fmean(1.0 if p else 0.0 for p in flags) for name, flags in totals.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "cases": len(self.cases),
            "completion_rate": round(self.completion_rate, 3),
            "pass_at_k": round(self.pass_at_k, 3),
            "pass_pow_k": round(self.pass_pow_k, 3),
            "mean_cost_usd": round(self.mean_cost, 6),
            "mean_latency_ms": round(self.mean_latency, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        """A JSON-able record of the whole run — the unit a `Reporter` emits to your store."""
        return {
            "suite": self.suite,
            "metrics": self.summary(),
            "grader_breakdown": self.grader_breakdown(),
            "cases": [c.to_dict() for c in self.cases],
        }

    def render(self) -> str:
        lines = [f"Eval: {self.suite}  ({len(self.cases)} cases)"]
        for case in self.cases:
            mark = "✓" if case.pass_pow_k == 1.0 else "✗"
            lines.append(
                f"  {mark} {case.name}  pass@{case.k}={case.pass_at_k:.2f} "
                f"pass^{case.k}={case.pass_pow_k:.2f}  "
                f"${case.mean_cost:.6f}  {case.mean_latency:.1f}ms"
            )
            # Surface the failing graders (from the first failing trial) for a quick diagnosis.
            failing = next((t for t in case.trials if not t.passed), None)
            if failing is not None:
                if failing.error:
                    lines.append(f"      ! error: {failing.error}")
                for grade in failing.grades:
                    if not grade.passed:
                        lines.append(f"      ✗ {grade.name}: {grade.reason}")
        s = self.summary()
        lines.append(
            f"Summary: completion {s['completion_rate']:.0%} · "
            f"pass@k {s['pass_at_k']:.2f} · pass^k {s['pass_pow_k']:.2f} · "
            f"${s['mean_cost_usd']:.6f} · {s['mean_latency_ms']:.1f}ms"
        )
        return "\n".join(lines)

    def require(
        self,
        *,
        completion: float | None = None,
        pass_at_k: float | None = None,
        pass_pow_k: float | None = None,
    ) -> Report:
        """Raise `AssertionError` (with the report) if a threshold isn't met — a CI/CD gate.

        Returns self on success, so it chains: `(await evaluate(...)).require(completion=0.9)`.
        """
        problems: list[str] = []
        if completion is not None and self.completion_rate < completion:
            problems.append(f"completion {self.completion_rate:.2f} < {completion}")
        if pass_at_k is not None and self.pass_at_k < pass_at_k:
            problems.append(f"pass@k {self.pass_at_k:.2f} < {pass_at_k}")
        if pass_pow_k is not None and self.pass_pow_k < pass_pow_k:
            problems.append(f"pass^k {self.pass_pow_k:.2f} < {pass_pow_k}")
        if problems:
            raise AssertionError(
                "eval thresholds not met: " + "; ".join(problems) + "\n" + self.render()
            )
        return self


async def grade_trial(trial: Trial, graders: Sequence[Grader]) -> TrialResult:
    """Run every grader over one (already executed) trial and package the result.

    The single place grading happens — shared by the offline runner and online scoring, so both
    compute pass/fail and the operational footprint identically.
    """
    grades = [await g.grade(trial) for g in graders]
    return TrialResult(
        grades=grades,
        cost_usd=trial.trace.cost_usd,
        latency_ms=trial.trace.duration_ms,
        error=trial.error,
    )


async def evaluate(
    target: CompiledGraph[Any],
    suite: Suite | Sequence[Case],
    *,
    trials: int | None = None,
    sandbox: Sandbox | None = None,
    concurrency: int = 1,
    reporter: Reporter | None = None,
) -> Report:
    """Run `suite` against `target` and return a `Report`.

    Args:
        target: The agent/graph under test.
        suite: A `Suite` (named goldens) or a plain sequence of `Case`s.
        trials: Override each case's trial count (else `Case.trials`).
        sandbox: Where trials run (default `InProcessSandbox`).
        concurrency: Max trials in flight at once (default 1 = sequential).
        reporter: If given, the report's `to_dict()` is emitted to this sink (file, DB, dashboard).
    """
    cases = suite.cases if isinstance(suite, Suite) else list(suite)
    name = suite.name if isinstance(suite, Suite) else "eval"
    box = sandbox or InProcessSandbox()
    limit = asyncio.Semaphore(max(1, concurrency))

    async def run_one(case: Case) -> TrialResult:
        async with limit:
            env = None
            try:
                if case.setup is not None:
                    env = await _maybe(case.setup())
                trial = await box.run(target, case, env)
                return await grade_trial(trial, case.graders)
            finally:
                if case.teardown is not None:
                    await _maybe(case.teardown(env))

    results: list[CaseResult] = []
    for case in cases:
        k = trials if trials is not None else case.trials
        trial_results = await asyncio.gather(*(run_one(case) for _ in range(k)))
        results.append(CaseResult(name=case.name, trials=list(trial_results)))
    report = Report(suite=name, cases=results)
    if reporter is not None:
        await deliver(reporter, report.to_dict())
    return report


async def _maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
