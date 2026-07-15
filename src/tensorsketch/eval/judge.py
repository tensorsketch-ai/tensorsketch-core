"""LLM-as-a-judge — for open-ended answers where exact checks are too brittle.

`LlmJudge` scores one **atomic** criterion with a binary pass/fail verdict (the "one criterion, one
failure mode" rubric style that makes judges converge and stay calibratable). Compose several
`LlmJudge`s for a multi-dimensional rubric. The judge runs on the harness's own provider — its call
is *not* part of the agent's trace, so it never counts against the agent's cost/latency budgets.

Calibration is your job: validate a judge against human labels (aim for a high rank correlation)
before trusting it, and keep each criterion narrow and specific rather than "is this good?".
"""

from __future__ import annotations

from ..core.schema import Schema
from ..messages import system, user
from ..providers.base import ChatProvider
from .case import Grade, Grader, Trial


class Verdict(Schema):
    """The judge's structured output: a binary decision and a short justification."""

    passed: bool
    reason: str = ""


_INSTRUCTIONS = (
    "You are a strict evaluator. Judge ONLY the single criterion below against the agent's output. "
    "Decide pass or fail — do not grade on a curve, do not invent other criteria. "
    "Return `passed` (true/false) and a one-sentence `reason`."
)


class LlmJudge(Grader):
    """Grade one criterion of the answer (optionally the trajectory) with an LLM."""

    def __init__(
        self,
        provider: ChatProvider,
        criterion: str,
        *,
        name: str = "judge",
        include_trajectory: bool = False,
    ) -> None:
        self._provider = provider
        self._criterion = criterion
        self.name = name
        self._include_trajectory = include_trajectory

    async def grade(self, trial: Trial) -> Grade:
        parts = [f"CRITERION:\n{self._criterion}", f"AGENT OUTPUT:\n{trial.text}"]
        if self._include_trajectory:
            parts.append(f"TRAJECTORY:\n{trial.trace.render()}")
        messages = [system(_INSTRUCTIONS), user("\n\n".join(parts))]
        completion = await self._provider.complete(messages, output_schema=Verdict)
        verdict = completion.parsed
        if isinstance(verdict, Verdict):
            return Grade.of(self.name, verdict.passed, reason=verdict.reason)
        # Provider returned no structured verdict — fail closed rather than silently pass.
        return Grade.of(self.name, False, reason="judge returned no verdict")
