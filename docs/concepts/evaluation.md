# Evaluation

An agent isn't a function from prompt to string — it's a path-dependent process that reasons, calls
tools, and changes state. So testing it means measuring not just *what* it produced, but the *whole
trajectory* it took to get there, across *multiple runs* (agents are non-deterministic). TensorSketch's eval
harness is built for exactly that, and it consumes the [trace](tracing.md) TensorSketch already records.

It lives in the one `tensorsketch` package — no extra to install. `LlmJudge` just needs a provider.

```python
from tensorsketch.eval import Case, Suite, evaluate, Contains, ToolCalled, StepEfficiency

suite = Suite("capitals", [
    Case("france", {"query": "Capital of France?"},
         graders=[Contains("Paris"), ToolCalled("search"), StepEfficiency(optimal_steps=3)],
         trials=3),
])

report = await evaluate(agent, suite)
print(report.render())
report.require(pass_pow_k=1.0)   # gate CI on the result
```

## The anatomy of a test

| Primitive | In TensorSketch | |
|---|---|---|
| **Task** | `Case` | inputs, graders, optional environment `setup`, and a trial count |
| **Trial** | one run of a case | agents vary run-to-run, so a case runs `trials` times |
| **Transcript** | `Trial.trace` | the [span tree](tracing.md) — every model/tool call, tokens, cost, timing |
| **Outcome** | `Trial.output` / `Trial.env` | the final state, and the environment to assert against |
| **Grader** | `Grader` → `Grade` | scores one aspect (pass/fail + a 0-1 score + a reason) |

A trial **passes** only if every one of its graders passes.

## Graders: a hybrid ecosystem

No single grading mechanism is enough, so the built-ins span the three architectures from the
research — with LLM judges for what code can't check, and code for everything it can.

### Code-based — fast, cheap, reproducible

| Grader | Checks |
|---|---|
| `Contains` · `Equals` · `Regex` | the answer text (or a state field, via `Equals(key=…)`) |
| `ToolCalled` · `ToolArgs` · `ToolSequence` | the **trajectory** — that the right tool ran, with the right **payload**, in the right order |
| `StepEfficiency` | steps taken vs. optimal — catches loops and thrash |
| `CostBudget` · `LatencyBudget` | the operational footprint, straight off the trace |
| `FinalState(predicate)` | the **outcome** — e.g. a row exists in `trial.env`'s database |
| `Custom(fn)` / `@grader` | any callable returning a `Grade`, a bool, or `(passed, reason)` |

`ToolArgs` grades the *payload, not just the path* — TensorSketch's tool spans carry the arguments each tool
was called with, so you can assert the agent passed the right parameters:

```python
ToolArgs("sql_query", lambda args: args["table"] == "customers")
```

### LLM-as-a-judge — for open-ended answers

When exact matching is too brittle, `LlmJudge` scores one **atomic** criterion with a **binary**
verdict — the "one criterion, one failure mode" style that keeps a judge consistent and
calibratable. Compose several for a multi-dimensional rubric:

```python
from tensorsketch.eval import LlmJudge

judges = [
    LlmJudge(provider, "The answer cites at least one source."),
    LlmJudge(provider, "The tone is professional and free of hedging."),
]
```

The judge runs on *your* provider and its call is **not** part of the agent's trace, so it never
counts against the agent's cost or latency. Calibration is your job: validate a judge against human
labels before trusting it, and keep each criterion narrow.

### Human-in-the-loop

The gold standard for calibrating judges and labelling hard cases. TensorSketch gives you the data — a
`Trial` with its full transcript — but the annotation queue/UI is out of scope for the library (see
[deferred](#whats-deferred)).

## Metrics: outcome and trajectory

`report.summary()` and the individual properties give both halves the research calls for:

- **Outcome** — `completion_rate` (fraction of trials that succeeded), **`pass_at_k`** (succeeded at
  least once across k trials — retry-friendly) and **`pass_pow_k`** (succeeded *every* time —
  demands consistency), plus `mean_cost` / `mean_latency`.
- **Trajectory** — surfaced through the tool/step graders above; `grader_breakdown()` shows the pass
  rate per grader across all trials, so you see *which* criterion is failing.

`pass^k` is the strict one: an agent that's right 4 times out of 5 scores `pass_pow_k = 0`. Use it
where consistency matters; use `pass@k` where a retry or re-prompt is acceptable.

## Isolation

Trials run behind a **`Sandbox`** seam. Crucially — in every sandbox — the **graders run in the
harness, never inside the agent's execution**, so an agent can't tamper with its own grade (the way
agents have gamed benchmarks that graded in-process). The default `InProcessSandbox` runs the trial
here, under a fresh tracer, with a fresh environment per trial so trials never cross-contaminate —
the right choice for LLM/tool/trajectory evaluation, where the `Case` controls the tools. Stronger
isolation (a subprocess, a container, a remote runner) is a future `Sandbox` behind the same seam.

## Storing & viewing results — TensorSketch emits, you own the store

TensorSketch is stateless, so it never bundles a results database. Every result serializes to a plain JSON
record — `report.to_dict()` (and `TrialResult.to_dict()`) — and you push it through a **`Reporter`**
sink to wherever your team looks: a file, a warehouse, a dashboard, any database. It's the same
exporter pattern as [tracing](tracing.md#exporters), just for the *scores* instead of the spans.

```python
from tensorsketch.eval import JsonlReporter, CallbackReporter

# a durable, jq-friendly log file (no dependencies)
await evaluate(agent, suite, reporter=JsonlReporter("evals.jsonl"))

# ...or straight into your own store — a Reporter is one method, sync or async
await evaluate(agent, suite, reporter=CallbackReporter(lambda record: warehouse.insert(record)))
```

Connecting a database is a few lines — implement `emit(record)`:

```python
class PostgresReporter:
    def __init__(self, pool):
        self.pool = pool

    async def emit(self, record):
        await self.pool.execute("insert into evals(doc) values ($1)", record)
```

For the **transcript** side (the raw spans, tokens, cost), the tracer exporters already cover it:
`FileTracer` (JSONL), `OTelTracer` (→ Grafana / Jaeger / Honeycomb / Datadog), or a custom
`RecordingTracer._record`. So traces and eval scores can each go to the store that suits them.

## Where this fits the lifecycle

**Offline** (above) — curated goldens, run as a regression suite, gating CI with
`report.require(...)`.

**Online** — once the agent is live there's no ground truth for a novel query, so you score what a
run *actually did* with reference-free graders (safety, tool-call failures, cost/latency budgets,
on-policy `LlmJudge` criteria). It reuses the same graders — they already read a `Trace` — via
`score(...)` for a single run or an `OnlineMonitor` that samples and emits:

```python
from tensorsketch.eval import OnlineMonitor, JsonlReporter, LatencyBudget, LlmJudge

monitor = OnlineMonitor(
    [LlmJudge(judge, "The answer stays on-policy."), LatencyBudget(3000)],
    reporter=JsonlReporter("online.jsonl"),
    sample=0.1,   # score 10% of traffic
)

tracer = InMemoryTracer()
state = await agent.invoke(inputs, tracer=tracer)
await monitor.observe(state, tracer.trace)   # off the response path — sample, score, emit
```

**The feedback loop**: capture a production failure, correct the expected outcome, and add it as a
new `Case` — every novel failure becomes a permanent regression test. (The annotation UI that
automates this is [deferred](#whats-deferred); the data — a `Trial` with its transcript — is here.)

## Drift detection — alerting on the online stream

Scoring each run is one thing; noticing that the *aggregate* has quietly regressed is another. A
**`DriftMonitor`** watches the same stream `OnlineMonitor` emits and raises a `DriftAlert` when
behavior shifts from a **`Baseline`** (usually your last green offline eval, via
`Baseline.from_report(report)`):

- a **drop in pass rate** — overall or for one grader — via a **two-proportion z-test** against the
  baseline (the right test for binary pass/fail), so a safety grader collapsing trips on its own;
- a **change-point in cost or latency** via the **Page-Hinkley** test — the canonical O(1)
  streaming detector for a shift in a numeric mean (fed the value *relative to* the baseline mean,
  so one threshold works across dollars and milliseconds).

`DriftMonitor` **is a `Reporter`**, so it drops straight into the online monitor. Use `MultiReporter`
to persist every result *and* watch for drift from one hand-off:

```python
from tensorsketch.eval import Baseline, DriftMonitor, MultiReporter, OnlineMonitor, JsonlReporter

baseline = Baseline.from_report(offline_report)          # what "healthy" looked like
drift = DriftMonitor(baseline, reporter=JsonlReporter("drift-alerts.jsonl"))

monitor = OnlineMonitor(
    graders,
    reporter=MultiReporter(JsonlReporter("online.jsonl"), drift),  # store + detect in one pass
    sample=0.1,
)
```

The rolling window lives in **this process's memory** — the detector persists nothing. Consistent
with the rest of TensorSketch: it *emits* alerts to your sink and never owns a drift database. A sustained
regression is **latched**, so it fires once (not on every subsequent record) until it recovers.

## What's deferred

Logged in [decisions](../design/roadmap.md) so nothing is lost:

- **Stronger sandboxes** — `SubprocessSandbox`, `DockerSandbox`, and remote runners behind the seam.
- **Distributional drift** — `DriftMonitor` (pass-rate + cost/latency) ships now; population-shift
  detectors (PSI / KL / KS over a reference distribution) and routing alerts to a pager are next.
- **Annotation queue** — the human-in-the-loop UI and the trace→golden pipeline as tooling.
- **More trajectory metrics** — memory hit rate (for memory-enabled agents) and scope-adherence.
- **Judge calibration tooling** — measuring a judge's rank correlation against human labels.

See [`examples/evaluation.py`](../../examples/evaluation.py) for the whole thing running offline.
