"""Code-based graders — fast, cheap, reproducible checks over the answer, the trajectory, and the
outcome. These grade *what the agent produced or altered*, mostly ignoring the path; the exception
(`ToolSequence`, `StepEfficiency`) grades the path itself. For open-ended answers where exact
checks are too brittle, reach for `LlmJudge` (see `judge.py`).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, overload

from ..observability.tracing import MODEL_KIND, TOOL_ARGS, TOOL_KIND, TOOL_NAME
from .case import Grade, Grader, Trial


def _tool_calls(trial: Trial, name: str | None = None) -> list[dict[str, Any]]:
    """The tool spans of a trial (optionally filtered by tool name), in call order."""
    spans = sorted(trial.trace.of_kind(TOOL_KIND), key=lambda s: s.start)
    calls = [s.attributes for s in spans]
    return [c for c in calls if name is None or c.get(TOOL_NAME) == name]


# --- answer (output text) -------------------------------------------------------------------


@dataclass
class Contains(Grader):
    """Pass if the answer contains `text` (case-insensitive by default)."""

    text: str
    case_sensitive: bool = False
    name: str = "contains"

    async def grade(self, trial: Trial) -> Grade:
        haystack = trial.text if self.case_sensitive else trial.text.lower()
        needle = self.text if self.case_sensitive else self.text.lower()
        found = needle in haystack
        return Grade.of(self.name, found, reason=f"{'found' if found else 'missing'} {self.text!r}")


@dataclass
class Equals(Grader):
    """Pass if a value equals `expected` — the answer text, or state field `key` if given."""

    expected: str
    key: str | None = None
    name: str = "equals"

    async def grade(self, trial: Trial) -> Grade:
        actual = trial.text if self.key is None else str(getattr(trial.output, self.key, ""))
        ok = actual == self.expected
        return Grade.of(
            self.name,
            ok,
            reason=f"{actual!r} == {self.expected!r}" if ok else f"{actual!r} != {self.expected!r}",
        )


@dataclass
class Regex(Grader):
    """Pass if the answer matches `pattern` (searched, not anchored)."""

    pattern: str
    name: str = "regex"

    async def grade(self, trial: Trial) -> Grade:
        ok = re.search(self.pattern, trial.text) is not None
        return Grade.of(self.name, ok, reason=f"/{self.pattern}/ {'matched' if ok else 'no match'}")


# --- trajectory (tool use) ------------------------------------------------------------------


@dataclass
class ToolCalled(Grader):
    """Pass if the tool `tool` was called at least `min_times`."""

    tool: str
    min_times: int = 1
    name: str = "tool_called"

    async def grade(self, trial: Trial) -> Grade:
        n = len(_tool_calls(trial, self.tool))
        return Grade.of(
            self.name, n >= self.min_times, reason=f"{self.tool} called {n}x (>= {self.min_times})"
        )


@dataclass
class ToolArgs(Grader):
    """Pass if some call to `tool` had arguments satisfying `predicate` (grades the payload)."""

    tool: str
    predicate: Callable[[dict[str, Any]], bool]
    name: str = "tool_args"

    async def grade(self, trial: Trial) -> Grade:
        calls = _tool_calls(trial, self.tool)
        ok = any(self.predicate(dict(c.get(TOOL_ARGS, {}))) for c in calls)
        return Grade.of(
            self.name,
            ok,
            reason=f"{self.tool} args {'matched' if ok else 'did not match'} ({len(calls)} calls)",
        )


@dataclass
class ToolSequence(Grader):
    """Pass if tools ran in this order. Subsequence by default; `exact` for the full path."""

    sequence: list[str]
    exact: bool = False
    name: str = "tool_sequence"

    async def grade(self, trial: Trial) -> Grade:
        actual = [str(c.get(TOOL_NAME)) for c in _tool_calls(trial)]
        ok = actual == self.sequence if self.exact else _is_subsequence(self.sequence, actual)
        return Grade.of(self.name, ok, reason=f"expected {self.sequence} in {actual}")


@dataclass
class StepEfficiency(Grader):
    """Pass if the trajectory used no more than `max_ratio`x `optimal_steps` (loop/thrash guard).

    A step is one model call or one tool call. Score is `optimal / actual`, capped at 1.0.
    """

    optimal_steps: int
    max_ratio: float = 1.5
    name: str = "step_efficiency"

    async def grade(self, trial: Trial) -> Grade:
        steps = len(trial.trace.of_kind(MODEL_KIND)) + len(trial.trace.of_kind(TOOL_KIND))
        if self.optimal_steps <= 0 or steps == 0:
            return Grade.of(self.name, steps <= self.optimal_steps, reason=f"{steps} steps")
        ratio = steps / self.optimal_steps
        score = min(1.0, self.optimal_steps / steps)
        return Grade.of(
            self.name,
            ratio <= self.max_ratio,
            score=score,
            reason=f"{steps} steps / {self.optimal_steps} optimal (x{ratio:.2f})",
        )


# --- operational footprint ------------------------------------------------------------------


@dataclass
class CostBudget(Grader):
    """Pass if the trial's estimated cost is within `max_usd`."""

    max_usd: float
    name: str = "cost_budget"

    async def grade(self, trial: Trial) -> Grade:
        cost = trial.trace.cost_usd
        return Grade.of(
            self.name, cost <= self.max_usd, reason=f"${cost:.6f} <= ${self.max_usd:.6f}"
        )


@dataclass
class LatencyBudget(Grader):
    """Pass if the trial finished within `max_ms`."""

    max_ms: float
    name: str = "latency_budget"

    async def grade(self, trial: Trial) -> Grade:
        ms = trial.trace.duration_ms
        return Grade.of(self.name, ms <= self.max_ms, reason=f"{ms:.1f}ms <= {self.max_ms:.1f}ms")


# --- outcome (environment state) & custom ---------------------------------------------------


@dataclass
class FinalState(Grader):
    """Pass if `predicate(trial)` holds — the *outcome* check (e.g. a row exists in `trial.env`).

    The predicate gets the whole trial, so it can query the environment, the output, or the trace.
    """

    predicate: Callable[[Trial], bool | Awaitable[bool]]
    name: str = "final_state"

    async def grade(self, trial: Trial) -> Grade:
        result = self.predicate(trial)
        ok = await result if inspect.isawaitable(result) else result
        return Grade.of(self.name, bool(ok), reason="final-state predicate")


@dataclass
class Custom(Grader):
    """Wrap any callable as a grader. It gets the trial and returns a `Grade`, a bool, or
    `(passed, reason)`; sync or async."""

    fn: Callable[[Trial], Any]
    name: str = "custom"

    async def grade(self, trial: Trial) -> Grade:
        result = self.fn(trial)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Grade):
            return result
        if isinstance(result, tuple):
            passed, reason = result
            return Grade.of(self.name, bool(passed), reason=str(reason))
        return Grade.of(self.name, bool(result))


@overload
def grader(fn: Callable[[Trial], Any]) -> Custom: ...


@overload
def grader(*, name: str | None = ...) -> Callable[[Callable[[Trial], Any]], Custom]: ...


def grader(
    fn: Callable[[Trial], Any] | None = None, *, name: str | None = None
) -> Custom | Callable[[Callable[[Trial], Any]], Custom]:
    """Decorator turning a function into a `Custom` grader: `@grader` or `@grader(name="…")`."""

    def make(target: Callable[[Trial], Any]) -> Custom:
        return Custom(target, name=name or target.__name__)

    return make(fn) if fn is not None else make


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(item in it for item in needle)
