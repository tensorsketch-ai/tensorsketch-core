"""Evaluate an agent — grade the trajectory and the outcome, not just the answer.

Defines a small suite of goldens, runs each case several times (agents are non-deterministic), and
prints a report with task-completion rate, pass@k / pass^k, cost, latency, and which grader failed.
Uses an offline provider so it runs without an API key; the graders are the same you'd use for real.

Run:  uv run python examples/evaluation.py
"""

from __future__ import annotations

import asyncio

from tensorsketch import create_agent, tool
from tensorsketch.eval import (
    Baseline,
    CallbackReporter,
    Case,
    Contains,
    DriftMonitor,
    LatencyBudget,
    OnlineMonitor,
    StepEfficiency,
    Suite,
    ToolArgs,
    ToolCalled,
    evaluate,
)
from tensorsketch.messages import Message, ToolCall, assistant
from tensorsketch.observability.tracing import InMemoryTracer
from tensorsketch.providers.fake import FakeProvider


@tool
def search(query: str) -> str:
    """Look something up (pretend)."""
    return "Paris is the capital of France."


def build_agent() -> object:
    # Offline policy: call `search` once, then answer from the tool result.
    def policy(messages: list[Message], tools: object) -> Message:
        if any(m.role == "tool" for m in messages):
            return assistant("The capital of France is Paris.")
        return assistant(
            "", tool_calls=[ToolCall(id="c1", name="search", args={"query": "France capital"})]
        )

    return create_agent(FakeProvider(policy=policy), tools=[search])


async def main() -> None:
    agent = build_agent()

    suite = Suite(
        "capitals",
        [
            # A good case: right answer, used the tool with a sensible query, no thrashing.
            Case(
                "france",
                {"query": "What is the capital of France?"},
                graders=[
                    Contains("Paris"),
                    ToolCalled("search"),
                    ToolArgs("search", lambda a: "France" in a.get("query", "")),
                    StepEfficiency(optimal_steps=3),
                ],
                trials=3,
            ),
            # A case this agent will fail — it always answers "Paris" regardless of the question.
            Case("germany", {"query": "Capital of Germany?"}, [Contains("Berlin")], trials=3),
        ],
    )

    # Emit the whole run to a sink. Here a callback stands in for "insert into my DB / dashboard";
    # `JsonlReporter("evals.jsonl")` would write a file instead. TensorSketch emits — you own the
    # store.
    emitted: list[dict[str, object]] = []
    report = await evaluate(agent, suite, reporter=CallbackReporter(emitted.append))
    print(report.render())
    print("\nper-grader pass rate:", report.grader_breakdown())
    print("emitted record metrics:", emitted[0]["metrics"])

    # As a CI gate you'd assert a threshold; here we just show it fails on the weak case.
    try:
        report.require(completion=1.0)
    except AssertionError as exc:
        print("\nCI gate would block:", str(exc).splitlines()[0])

    # Online: score a *live* run reference-free (no golden), and emit each result to the same kind
    # of sink — this is how you'd watch production traffic. Run it right after a real invoke.
    monitor = OnlineMonitor(
        [Contains("Paris"), LatencyBudget(3000)],
        reporter=CallbackReporter(emitted.append),
    )
    tracer = InMemoryTracer()
    state = await agent.invoke({"query": "What is the capital of France?"}, tracer=tracer)
    result = await monitor.observe(state, tracer.trace)
    print("\nonline (production) trace scored:", result.passed if result else None)

    # Drift: watch that online result stream and alert when quality regresses vs a baseline (your
    # last green eval). In production you'd wire it as OnlineMonitor(reporter=MultiReporter(store,
    # drift)); here we push a stream of records in directly to show an alert firing.
    baseline = Baseline(pass_rate=0.95, n=200)  # or Baseline.from_report(offline_report)
    drift_alerts: list[dict[str, object]] = []
    drift = DriftMonitor(baseline, reporter=CallbackReporter(drift_alerts.append), min_samples=20)
    for _ in range(20):  # a production incident: runs suddenly start failing the check
        await drift.emit(
            {
                "passed": False,
                "cost_usd": 0.001,
                "latency_ms": 120.0,
                "error": None,
                "grades": [{"name": "on_policy", "passed": False}],
            }
        )
    print("drift alerts:", [a["reason"] for a in drift_alerts])


if __name__ == "__main__":
    asyncio.run(main())
