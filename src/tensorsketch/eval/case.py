"""The anatomy of an agent test: task, trial, transcript, outcome — and the grader that scores it.

An agent isn't a function from prompt to string; it's a path-dependent process. So a test here is
not "input → expected text". Following the task/trial/transcript/outcome/grader decomposition:

- a **`Case`** is the *task* — inputs, the tools/environment, and the graders that define success;
- running it produces a **`Trial`** — the *transcript* (`trace`), the final state (*outcome*), and
  the environment, everything a grader needs;
- a **`Grader`** scores one aspect of that trial and returns a **`Grade`** (pass/fail + a score).

The transcript is TensorSketch's own `Trace` (from `tensorsketch.observability`): the harness runs a
trial under an
`InMemoryTracer`, so graders can inspect every model and tool call — grading the *trajectory*, not
just the answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.schema import Schema
from ..observability.tracing import Trace


@dataclass
class Grade:
    """The result of one grader: whether it passed, a 0-1 score, and a reason."""

    name: str
    passed: bool
    score: float
    reason: str = ""

    @classmethod
    def of(cls, name: str, passed: bool, *, score: float | None = None, reason: str = "") -> Grade:
        """Build a grade; `score` defaults to `1.0`/`0.0` from `passed` (binary graders)."""
        return cls(name, passed, float(passed) if score is None else score, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
        }


def output_text(state: Any) -> str:
    """Best-effort extraction of the assistant's answer text from a final graph state."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    output = getattr(state, "output", None)
    if output:
        return str(output)
    messages = getattr(state, "messages", None) or []
    for message in reversed(messages):
        if getattr(message, "role", None) == "assistant" and getattr(message, "content", ""):
            return str(message.content)
    return ""


@dataclass
class Trial:
    """One execution of a case — the raw artifacts a grader reads.

    Attributes:
        case: The case that produced this trial (`None` for a production run scored online).
        output: The graph's final state (`None` if the run raised).
        trace: The transcript — every model/tool span, with tokens, cost, and timing.
        env: Whatever `Case.setup()` returned (the environment to assert outcomes against).
        error: The exception string if the run failed, else `None`.
    """

    case: Case | None
    output: Any
    trace: Trace
    env: Any = None
    error: str | None = None

    @property
    def text(self) -> str:
        """The agent's answer text (from `output`)."""
        return output_text(self.output)


class Grader(ABC):
    """Scores one aspect of a `Trial`. Subclass with a `name` and an async `grade`."""

    name: str = "grader"

    @abstractmethod
    async def grade(self, trial: Trial) -> Grade: ...


# Environment lifecycle hooks may be sync or async; the runner awaits either.
Setup = Callable[[], Any] | Callable[[], Awaitable[Any]]
Teardown = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]


@dataclass
class Case:
    """A task: the inputs to run, the graders that define success, and how many trials to run.

    Attributes:
        name: A stable identifier (shown in the report).
        inputs: What the graph is invoked with (a mapping or a `Schema`).
        graders: The checks; a trial *passes* only if every grader passes.
        trials: How many times to run it (agents are non-deterministic — repeat for reliability).
        setup / teardown: Optional per-trial environment lifecycle. `setup()` returns the `env`
            handed to graders (e.g. a fresh temp DB); `teardown(env)` cleans it up. Run per trial
            so trials never cross-contaminate.
        metadata: Free-form tags for slicing results.
    """

    name: str
    inputs: Mapping[str, Any] | Schema
    graders: Sequence[Grader]
    trials: int = 1
    setup: Setup | None = None
    teardown: Teardown | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Suite:
    """A named collection of cases — the offline eval dataset (the 'goldens')."""

    name: str
    cases: list[Case]

    def __iter__(self) -> Any:
        return iter(self.cases)
