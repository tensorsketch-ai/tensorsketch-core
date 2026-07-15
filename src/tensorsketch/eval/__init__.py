"""Agent evaluation — grade the *trajectory*, not just the answer.

Because an agent is a path-dependent process, a test here is a **`Case`** (task) run over multiple
**trials**, each producing a **`Trial`** (transcript + outcome) scored by one or more **graders**.
Graders are hybrid by design: fast, reproducible **code-based** checks over the answer, the tool
trajectory, and the environment state — plus **`LlmJudge`** for open-ended criteria. `evaluate`
runs a suite and returns a **`Report`** with task-completion rate, pass@k / pass^k, cost, latency,
and a per-grader breakdown; `Report.require(...)` gates CI.

    from tensorsketch.eval import Case, Suite, evaluate, Contains, ToolCalled

    suite = Suite("smoke", [Case("greets", {"query": "hi"}, [Contains("hello")])])
    report = await evaluate(agent, suite, trials=3)
    report.require(pass_pow_k=1.0)

Trials run behind a `Sandbox` seam (in-process default); the graders always run in the harness,
never inside the agent. This lives in the one `tensorsketch` package — no extra to install;
`LlmJudge`
just takes a provider.
"""

from __future__ import annotations

from .case import Case, Grade, Grader, Suite, Trial, output_text
from .drift import Baseline, DriftAlert, DriftMonitor, PageHinkley, two_proportion_z
from .graders import (
    Contains,
    CostBudget,
    Custom,
    Equals,
    FinalState,
    LatencyBudget,
    Regex,
    StepEfficiency,
    ToolArgs,
    ToolCalled,
    ToolSequence,
    grader,
)
from .judge import LlmJudge, Verdict
from .online import OnlineMonitor, score
from .reporters import CallbackReporter, JsonlReporter, MultiReporter, Reporter
from .runner import CaseResult, Report, TrialResult, evaluate, grade_trial
from .sandbox import InProcessSandbox, Sandbox

__all__ = [
    "Baseline",
    "CallbackReporter",
    "Case",
    "CaseResult",
    "Contains",
    "CostBudget",
    "Custom",
    "DriftAlert",
    "DriftMonitor",
    "Equals",
    "FinalState",
    "Grade",
    "Grader",
    "InProcessSandbox",
    "JsonlReporter",
    "LatencyBudget",
    "LlmJudge",
    "MultiReporter",
    "OnlineMonitor",
    "PageHinkley",
    "Regex",
    "Report",
    "Reporter",
    "Sandbox",
    "StepEfficiency",
    "Suite",
    "ToolArgs",
    "ToolCalled",
    "ToolSequence",
    "Trial",
    "TrialResult",
    "Verdict",
    "evaluate",
    "grade_trial",
    "grader",
    "output_text",
    "score",
    "two_proportion_z",
]
