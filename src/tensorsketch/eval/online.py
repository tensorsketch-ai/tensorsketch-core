"""Online evaluation — score *production* runs, not just offline goldens.

Once an agent is live there's no pre-defined ground truth for a novel query, so online eval scores
what a run actually did: reference-free checks (safety, tool-call failures, cost/latency budgets)
and `LlmJudge` criteria that don't need an expected answer. It reuses the exact same graders as the
offline suite — they already read a `Trace` — so a check you gate CI on can also watch production.

`score(output, trace, graders)` grades one captured run. `OnlineMonitor` wraps sampling + emitting
to a `Reporter`: build it once, then hand it each finished run (or a sampled fraction) and it scores
in the background and ships the result to your store.

    monitor = OnlineMonitor([LlmJudge(judge, "Answer is on-policy."), LatencyBudget(3000)],
                            reporter=JsonlReporter("online.jsonl"), sample=0.1)

    tracer = InMemoryTracer()
    state = await agent.invoke(inputs, tracer=tracer)
    await monitor.observe(state, tracer.trace)   # samples, scores, emits — off the response path
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ..observability.tracing import Trace
from .case import Grader, Trial
from .reporters import Reporter, deliver
from .runner import TrialResult, grade_trial


async def score(
    output: Any,
    trace: Trace,
    graders: Sequence[Grader],
    *,
    env: Any = None,
    error: str | None = None,
) -> TrialResult:
    """Grade a single captured run (no `Case` needed) — the online primitive."""
    trial = Trial(case=None, output=output, trace=trace, env=env, error=error)
    return await grade_trial(trial, graders)


class OnlineMonitor:
    """Sample finished production runs, score them, and emit each result to a `Reporter`."""

    def __init__(
        self,
        graders: Sequence[Grader],
        *,
        reporter: Reporter | None = None,
        sample: float = 1.0,
    ) -> None:
        self._graders = list(graders)
        self._reporter = reporter
        self._sample = sample

    async def observe(
        self, output: Any, trace: Trace, *, env: Any = None, error: str | None = None
    ) -> TrialResult | None:
        """Score this run (subject to the sample rate) and emit it; returns the result, or `None`
        if the run wasn't sampled."""
        if self._sample < 1.0 and random.random() > self._sample:
            return None
        result = await score(output, trace, self._graders, env=env, error=error)
        if self._reporter is not None:
            await deliver(self._reporter, result.to_dict())
        return result
