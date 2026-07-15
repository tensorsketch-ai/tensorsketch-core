"""The evaluation harness: cases/trials/graders, trajectory + outcome checks, metrics, gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorsketch import create_agent, tool
from tensorsketch.agents.agent import AgentState
from tensorsketch.core.graph import CompiledGraph
from tensorsketch.eval import (
    Baseline,
    CallbackReporter,
    Case,
    Contains,
    CostBudget,
    Custom,
    DriftMonitor,
    Equals,
    FinalState,
    JsonlReporter,
    LatencyBudget,
    LlmJudge,
    MultiReporter,
    OnlineMonitor,
    PageHinkley,
    Reporter,
    StepEfficiency,
    Suite,
    ToolArgs,
    ToolCalled,
    ToolSequence,
    Trial,
    evaluate,
    grader,
    score,
    two_proportion_z,
)
from tensorsketch.messages import Message, ToolCall, assistant
from tensorsketch.observability.tracing import InMemoryTracer
from tensorsketch.providers.fake import FakeProvider


@tool
def search(query: str) -> str:
    """Search the web (pretend)."""
    return f"result for {query}"


def tool_then_answer(answer: str = "the capital is Paris") -> CompiledGraph[AgentState]:
    """An agent that calls `search` once, then answers — a deterministic 2-tool-turn trajectory."""

    def policy(messages: list[Message], tools: object) -> Message:
        if any(m.role == "tool" for m in messages):
            return assistant(answer)
        return assistant(
            "", tool_calls=[ToolCall(id="c1", name="search", args={"query": "capital of France"})]
        )

    return create_agent(FakeProvider(policy=policy), tools=[search])


def direct(answer: str) -> CompiledGraph[AgentState]:
    """An agent that answers immediately (one model call, no tools)."""
    return create_agent(FakeProvider(policy=lambda m, t: assistant(answer)))


# --- outcome metrics: multi-trial pass@k / pass^k -------------------------------------------


async def test_pass_pow_k_and_completion() -> None:
    suite = Suite(
        "smoke",
        [
            Case("ok", {"query": "hi"}, [Contains("paris")], trials=3),
            Case("bad", {"query": "hi"}, [Contains("NOPE")]),
        ],
    )
    report = await evaluate(tool_then_answer(), suite)

    ok, bad = report.cases
    assert ok.pass_at_k == 1.0 and ok.pass_pow_k == 1.0 and ok.k == 3
    assert bad.pass_at_k == 0.0 and bad.pass_pow_k == 0.0
    assert report.completion_rate == pytest.approx(3 / 4)
    assert report.pass_pow_k == 0.5


async def test_trials_override() -> None:
    suite = Suite("s", [Case("ok", {"query": "hi"}, [Contains("paris")])])
    report = await evaluate(tool_then_answer(), suite, trials=5)
    assert report.cases[0].k == 5


# --- trajectory graders (read the enriched tool spans) --------------------------------------


async def test_tool_trajectory_graders() -> None:
    case = Case(
        "trajectory",
        {"query": "capital of France?"},
        [
            ToolCalled("search"),
            ToolArgs("search", lambda a: "France" in a.get("query", "")),
            ToolSequence(["search"]),
            StepEfficiency(optimal_steps=3),
        ],
    )
    report = await evaluate(tool_then_answer(), [case])
    assert report.completion_rate == 1.0


async def test_tool_called_fails_when_absent() -> None:
    report = await evaluate(direct("done"), [Case("c", {"query": "x"}, [ToolCalled("search")])])
    assert report.completion_rate == 0.0
    breakdown = report.grader_breakdown()
    assert breakdown["tool_called"] == 0.0


async def test_step_efficiency_flags_thrash() -> None:
    # 2 steps (1 tool + 1 answer... plus the initial model call = 3), optimal 1 → over budget.
    case = Case("eff", {"query": "x"}, [StepEfficiency(optimal_steps=1, max_ratio=1.2)])
    report = await evaluate(tool_then_answer(), [case])
    assert report.completion_rate == 0.0


# --- operational footprint ------------------------------------------------------------------


async def test_cost_budget_passes_offline() -> None:
    # The fake provider reports no usage, so cost is 0 — within any budget.
    report = await evaluate(direct("hi"), [Case("c", {"query": "x"}, [CostBudget(max_usd=1.0)])])
    assert report.completion_rate == 1.0


# --- outcome (environment state) ------------------------------------------------------------


async def test_final_state_checks_environment() -> None:
    class Env:
        def __init__(self) -> None:
            self.rows = [1]

    case = Case(
        "outcome",
        {"query": "x"},
        [FinalState(lambda trial: len(trial.env.rows) == 1)],
        setup=Env,
    )
    report = await evaluate(direct("done"), [case])
    assert report.completion_rate == 1.0


async def test_setup_teardown_run_per_trial() -> None:
    calls = {"setup": 0, "teardown": 0}

    def setup() -> str:
        calls["setup"] += 1
        return "env"

    def teardown(env: object) -> None:
        calls["teardown"] += 1

    case = Case("c", {"query": "x"}, [Contains("hi")], trials=3, setup=setup, teardown=teardown)
    await evaluate(direct("hi"), [case])
    assert calls == {"setup": 3, "teardown": 3}


# --- custom graders -------------------------------------------------------------------------


async def test_custom_grader_and_decorator() -> None:
    @grader(name="short")
    def short_answer(trial: Trial) -> bool:
        return len(trial.text) < 20

    boolean = Custom(lambda t: t.text == "hi", name="is_hi")
    report = await evaluate(direct("hi"), [Case("c", {"query": "x"}, [short_answer, boolean])])
    assert report.completion_rate == 1.0


async def test_equals_on_state_field() -> None:
    report = await evaluate(
        direct("Paris"), [Case("c", {"query": "x"}, [Equals("Paris", key="output")])]
    )
    assert report.completion_rate == 1.0


# --- LLM-as-judge (structured verdict via the provider seam) --------------------------------


async def test_llm_judge_pass_and_fail() -> None:
    yes = FakeProvider(script=[assistant('{"passed": true, "reason": "meets it"}')])
    no = FakeProvider(script=[assistant('{"passed": false, "reason": "too vague"}')])
    suite = [
        Case("pass", {"query": "x"}, [LlmJudge(yes, "Answer is specific.")]),
        Case("fail", {"query": "x"}, [LlmJudge(no, "Answer is specific.")]),
    ]
    report = await evaluate(direct("something"), suite)
    assert report.cases[0].pass_pow_k == 1.0
    assert report.cases[1].pass_pow_k == 0.0


# --- errors and CI gating -------------------------------------------------------------------


async def test_run_error_is_captured_not_raised() -> None:
    # Invalid input names a non-existent state field → the run raises; the harness records it.
    report = await evaluate(direct("hi"), [Case("bad", {"nonexistent": 1}, [Contains("hi")])])
    assert report.completion_rate == 0.0
    assert report.cases[0].trials[0].error is not None


async def test_require_gates_on_thresholds() -> None:
    good = await evaluate(direct("paris"), [Case("c", {"query": "x"}, [Contains("paris")])])
    assert good.require(completion=1.0, pass_pow_k=1.0) is good  # returns self on success

    bad = await evaluate(direct("nope"), [Case("c", {"query": "x"}, [Contains("paris")])])
    with pytest.raises(AssertionError, match="completion"):
        bad.require(completion=1.0)


# --- serialization + emitting to a sink -----------------------------------------------------


async def test_report_to_dict_is_json_able() -> None:
    report = await evaluate(direct("paris"), [Case("c", {"query": "x"}, [Contains("paris")])])
    doc = report.to_dict()
    assert set(doc) == {"suite", "metrics", "grader_breakdown", "cases"}
    assert doc["cases"][0]["trials"][0]["grades"][0]["name"] == "contains"
    json.dumps(doc)  # must round-trip through JSON with no custom encoder


async def test_jsonl_reporter_writes_a_record(tmp_path: Path) -> None:
    path = tmp_path / "evals.jsonl"
    await evaluate(
        direct("paris"),
        Suite("s", [Case("c", {"query": "x"}, [Contains("paris")])]),
        reporter=JsonlReporter(path),
    )
    line = path.read_text().strip()
    record = json.loads(line)
    assert record["suite"] == "s"
    assert record["metrics"]["completion_rate"] == 1.0


async def test_callback_reporter_async_sink() -> None:
    captured: list[dict[str, object]] = []

    async def sink(record: dict[str, object]) -> None:
        captured.append(record)

    await evaluate(
        direct("paris"),
        [Case("c", {"query": "x"}, [Contains("paris")])],
        reporter=CallbackReporter(sink),
    )
    assert captured and captured[0]["suite"] == "eval"


# --- online scoring (reference-free, against a live trace) ----------------------------------


async def test_score_a_captured_run() -> None:
    agent = direct("Paris")
    tracer = InMemoryTracer()
    state = await agent.invoke({"query": "capital?"}, tracer=tracer)
    result = await score(state, tracer.trace, [Contains("Paris"), LatencyBudget(10_000)])
    assert result.passed
    assert {g.name for g in result.grades} == {"contains", "latency_budget"}


async def test_online_monitor_emits_and_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []
    agent = direct("Paris")
    tracer = InMemoryTracer()
    state = await agent.invoke({"query": "x"}, tracer=tracer)

    monitor = OnlineMonitor(
        [Contains("Paris")], reporter=CallbackReporter(lambda r: captured.append(r)), sample=0.5
    )

    # above the 0.5 sample rate → skipped
    monkeypatch.setattr("tensorsketch.eval.online.random.random", lambda: 0.9)
    assert await monitor.observe(state, tracer.trace) is None
    assert captured == []

    monkeypatch.setattr("tensorsketch.eval.online.random.random", lambda: 0.1)  # below 0.5 → scored
    result = await monitor.observe(state, tracer.trace)
    assert result is not None and result.passed
    assert captured and captured[0]["grades"][0]["name"] == "contains"  # type: ignore[index]


# --- drift detection over the emitted stream ------------------------------------------------


def _record(
    passed: bool, *, cost: float = 0.001, latency: float = 100.0, grader_name: str = "contains"
) -> dict[str, object]:
    """A minimal `TrialResult.to_dict()`-shaped record, the unit a `DriftMonitor` ingests."""
    return {
        "passed": passed,
        "cost_usd": cost,
        "latency_ms": latency,
        "error": None,
        "grades": [{"name": grader_name, "passed": passed, "score": 0.0, "reason": ""}],
    }


def test_two_proportion_z_flags_a_drop() -> None:
    steady = two_proportion_z(100, 100, 100, 100)  # identical rates -> ~0
    dropped = two_proportion_z(100, 100, 5, 20)  # 100% -> 25% over 20 samples
    assert abs(steady) < 1e-9
    assert dropped < -3.0  # a strong, significant regression


def test_page_hinkley_detects_a_step_change() -> None:
    ph = PageHinkley(delta=0.05, lam=1.0, direction="up")
    assert not any(ph.update(1.0) for _ in range(30))  # a stable stream never fires
    assert any(ph.update(3.0) for _ in range(10))  # a sustained 3x step does


async def test_drift_monitor_flags_a_pass_rate_drop() -> None:
    alerts: list[dict[str, object]] = []
    drift = DriftMonitor(
        Baseline(pass_rate=1.0, n=200),
        reporter=CallbackReporter(lambda a: alerts.append(a)),
        window=50,
        min_samples=20,
    )
    for _ in range(20):  # a run of all-failing production results
        await drift.emit(_record(passed=False))
    assert alerts, "a collapse from 100% to 0% pass rate should alert"
    first = alerts[0]
    assert first["metric"] == "pass_rate" and first["direction"] == "down"
    # Latched: a sustained regression fires once, not on every subsequent record.
    assert sum(1 for a in alerts if a["metric"] == "pass_rate") == 1


async def test_drift_monitor_flags_a_cost_change_point() -> None:
    drift = DriftMonitor(Baseline(pass_rate=1.0, n=200, mean_cost=0.001, mean_latency=100.0))
    for _ in range(15):  # steady cost around the baseline
        await drift.emit(_record(passed=True, cost=0.001))
    for _ in range(10):  # then a 10x cost regression
        await drift.emit(_record(passed=True, cost=0.01))
    metrics = {a.metric for a in drift.alerts}
    assert "cost_usd" in metrics  # the cost change-point fires
    assert "pass_rate" not in metrics  # quality never dropped, so no false positive


async def test_drift_monitor_is_a_reporter_and_multi_reporter_fans_out() -> None:
    stored: list[dict[str, object]] = []
    drift = DriftMonitor(Baseline(pass_rate=1.0, n=50), min_samples=5)
    assert isinstance(drift, Reporter)  # so it drops into OnlineMonitor(reporter=...)

    fan = MultiReporter(CallbackReporter(lambda r: stored.append(r)), drift)
    for _ in range(6):
        await fan.emit(_record(passed=False))
    assert len(stored) == 6  # every record persisted to the store sink...
    assert any(a.metric == "pass_rate" for a in drift.alerts)  # ...and drift saw the same stream


async def test_baseline_from_report_captures_grader_rates() -> None:
    report = await evaluate(
        direct("paris"), [Case("c", {"query": "x"}, [Contains("paris")], trials=2)]
    )
    baseline = Baseline.from_report(report)
    assert baseline.pass_rate == 1.0
    assert baseline.n == 2
    assert baseline.grader_pass_rates["contains"] == 1.0
