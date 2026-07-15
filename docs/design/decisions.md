# Decisions log

Short, dated records of design decisions — especially reversals — so the *why* isn't lost.
Newest first. See also the [build status & backlog](status.md).

---

## D1 — TensorSketch is stateless; state and memory live outside the framework

**Decision.** The framework holds **no** state of its own. Everything persistent — run
checkpoints, the effect journal, and (later) memory — goes through **pluggable connectors** the
application points at its own database. TensorSketch ships small built-ins (`InMemoryBackend`,
`SqliteBackend`) and a clean interface (`Backend`) so any popular or custom database drops in.

**Why.** A stateless core is what makes an agent system scalable and easy to operate: any number
of workers can serve the same runs because the state isn't in the process. It also keeps the
framework unopinionated about *your* infrastructure.

**Implications / to build.** "Bring-your-own-database" connectors (Postgres, Redis, …) behind the
existing `Backend` interface; the interface is the seam, the connectors are add-ons.

**Status.** Delivered. `PostgresBackend` (psycopg 3, `tensorsketch-core[postgres]`) and `RedisBackend`
(redis-py, `tensorsketch-core[redis]`) ship behind the `Backend` ABC — lazily imported, so the core depends on
no driver. A shared `SqlBackend` base covers any DB-API database (SQLite and Postgres are thin
dialects). Serialization is a pluggable `Serializer` seam (default `PickleSerializer`). The full
durability suite runs against Redis (via fakeredis) in CI; a Postgres DSN in `LOOM_TEST_POSTGRES`
adds it too. Writing a connector for an unshipped store is just implementing the ABC.

## D2 — The first-cut memory subsystem was removed

**Decision.** The keyword-search `MemoryStore` (`InMemoryStore`, `memory_tools`, and agent recall
injection) is **removed**, not iterated on.

**Why.**
1. **Keyword search doesn't work.** Real recall needs semantic search (embeddings / a model),
   not token overlap. A toy matcher would set the wrong expectation.
2. **In-framework memory conflicts with statelessness (D1).** Conversation memory and user
   memory should be owned by the application and backed by its database, not baked into the loop.

**Future approach.** Memory will be an **external, embedding-based store the app owns**, reached
through the same bring-your-own-database story — not a built-in keyword store. Agents will read
and write it explicitly (e.g. via tools) rather than the framework auto-managing it.

## D3 — Batteries-included providers: OpenAI, Anthropic, Google — plus an easy custom path

**Decision.** Ship first-class providers for **OpenAI, Anthropic, and Google** (each an optional
install, lazily imported), and make adding **any** custom LLM API trivial via the `ChatProvider`
interface. No more than those three built in; everything else is a ~30-line custom provider.

**Why.** These three cover the vast majority of usage; beyond them, breadth is better served by a
low-friction extension point than by the core carrying every SDK.

**Status.** Anthropic, OpenAI, and Google providers built (lazy imports, optional extras,
mapping covered by fake-client tests). `OpenAIProvider` also drives OpenAI-compatible endpoints
via `base_url`. Custom path documented. Gate for Phase 1 is cleared. *Live-API verification for
all three is still pending (see status.md → Verify before shipping).*

## D4 — Phase 1 canvas adopts the Excalidraw *aesthetic* only

**Decision.** When the code⇄canvas engine (Phase 1) lands, the canvas and blocks take on
**Excalidraw's look** — hand-drawn shapes, color palette, and theme. The block *content and
behavior* are entirely TensorSketch's (typed ports, holes, the graph model); only the visual style is
borrowed.

**Why.** Excalidraw's friendly, sketchy aesthetic fits the "head-start sketch you refine in code"
positioning, and it's a familiar, well-liked visual language.

## D5 — Extensibility is named factories, not a plugin ecosystem

**Decision.** The way to select or extend TensorSketch's swappable parts is a small **registry** that maps
a **name → factory**, with lazy loading. Only two seams get one — **providers** and **backends** —
via `create_provider("anthropic", …)` / `create_backend("postgres", …)` and `register_*` /
`tensorsketch.providers` · `tensorsketch.backends` entry points. No `Plugin` bundle object, and no splitting TensorSketch
into core/community/provider packages.

**Why.**
1. **No install fragmentation.** Everything first-party stays in the single `tensorsketch` package;
   extras (`tensorsketch-core[anthropic]`, …) only gate a heavy third-party SDK/driver, never TensorSketch itself. A
   user should `pip install tensorsketch-core` and have the batteries, not assemble a constellation of
   sub-packages (the pattern that made early LangChain painful).
2. **The real win is config-driven selection.** Naming a model/database lets a config value or a
   `--flag` choose it, instead of a hard-coded `import` — the only ergonomic gap worth closing.
3. **Keep the zero-import property.** Names resolve lazily (built-ins are thunks; entry points
   `.load()` on demand), so `import tensorsketch` still pulls in no optional SDK and listing names imports
   nothing. Entry points are inherently lazy, which is why they fit.
4. **Small on purpose.** Two registries cover the seams where names pay off. Middleware, tracers,
   and tools are constructed and passed explicitly today; a registry can be added later if a real
   need appears, on this same primitive.

**Status.** Built (`tensorsketch.registry`): a generic lazy `Registry[T]`, the `providers` and `backends`
instances pre-registered with lazy thunks, `create_*`/`register_*` helpers, and entry-point
discovery (explicit registration overrides an installed name). Tested including a fresh-interpreter
check that listing/creating imports no provider SDK or DB driver.

## D6 — Serving is an optional ASGI layer over standard protocols

**Decision.** Serving a TensorSketch agent (OpenAI-compatible, A2A, AG-UI) is an **optional** `tensorsketch-core[serve]`
extra built on **Starlette**. Each factory (`openai_app` / `a2a_app` / `agui_app`) returns a
mountable **ASGI app**; the user runs it with any ASGI server (uvicorn). A2A also has a *consume*
side — `a2a_tool(url)` — so TensorSketch sits on both ends. A shared `ChatAdapter` (`to_input`/`to_reply`)
is the single seam between a graph and every protocol.

**Why.**
1. **Async-native + real SSE.** All three protocols stream over Server-Sent Events; the framework
   is async, so an ASGI foundation is the right substrate (vs. bridging an async engine into a
   sync stdlib server). Starlette is tiny, standard, and battle-tested.
2. **Dependency-free core preserved.** Starlette/httpx import only inside `tensorsketch.serve` — `import
   tensorsketch` still pulls in no web framework, consistent with the mcp/otel/db extras.
3. **Mountable, not a bundled server.** Returning an ASGI app (not a `serve()` that binds a port)
   lets users pick their server, add middleware/auth, and mount under an existing app.
4. **One adapter, three protocols.** The protocols differ only in envelope; the graph↔chat mapping
   is shared, so a non-agent graph is served by overriding `to_input`/`to_reply`, nothing else.

**Scope (deliberate subsets).** Token-level streaming waits on provider token streaming (today the
completed reply is sliced into real SSE deltas). A2A implements the agent card + `message/send` /
`message/stream` with a completed-task result, not the full task store / `tasks/*` /
push-notifications. Multi-turn history and inbound tool definitions aren't wired into the default
agent. Each is a clean extension on this same foundation.

**Status.** Built (`tensorsketch.serve`): `openai_app`, `a2a_app` + `a2a_tool`, `agui_app`, `ChatAdapter`,
`AgentCard`, shared SSE helper. Tested over the real ASGI apps (Starlette `TestClient` + httpx ASGI
transport), including an A2A consume↔expose round-trip. `import tensorsketch` verified free of Starlette/httpx.

## D7 — Evaluation grades the trajectory, over the trace we already emit

**Decision.** The eval harness (`tensorsketch.eval`) is a **code-first, in-package** subsystem (no extra to
install) built directly on TensorSketch's own `Trace`. It models the task/trial/transcript/outcome/grader
anatomy: a `Case` runs over multiple `trials`, each producing a `Trial` (trace + final state + env)
scored by a **hybrid** grader set — code-based checks (answer, tool trajectory + payload, step
efficiency, cost/latency, final-state) **and** `LlmJudge` (binary, one-criterion). `evaluate`
returns a `Report` with completion rate, **pass@k / pass^k**, cost/latency, and a per-grader
breakdown; `report.require(...)` is the CI gate. Trials run behind a **`Sandbox`** seam
(in-process default). This is the **offline** half of the lifecycle.

**Why.**
1. **The trace is the transcript.** TensorSketch already records every model/tool call with tokens, cost,
   and timing (native tracing, D-tracing). So the harness doesn't rebuild observability — it grades
   the span tree. That's the whole reason tracing came first. (One enabling edit: the agent's tool
   span now carries the call's **args** and **result**, so `ToolArgs` can grade the payload, not
   just the path.)
2. **Agents are path-dependent and non-deterministic.** Hence multi-trial by default and both
   pass@k (retry-friendly) and pass^k (consistency) — a single run is statistically invalid.
3. **No single grader suffices.** Code for the deterministic/objective (fast, cheap, reproducible),
   LLM-as-judge for the open-ended (binary, atomic criteria to stay calibratable). Human-in-the-loop
   is supported as *data* (a `Trial` with its transcript), not a bundled UI.
4. **Anti-gaming by construction.** Graders run in the harness, never inside the agent's execution;
   the `Sandbox` only runs the agent and hands back artifacts. So an agent can't overwrite the
   grader the way agents have gamed in-process benchmarks. A fresh env per trial prevents
   cross-contamination.
5. **Code, not YAML.** Cases and graders are Python, consistent with "code is the source of truth" —
   a config-matrix surface would contradict the framework's thesis.

**Deferred (logged, not lost).** Stronger sandboxes (`SubprocessSandbox` / `DockerSandbox` / remote)
behind the same seam — needed once agents write code / run bash. Online evaluation (async sampling
of production traces with the same graders; drift detection). The feedback-loop *as tooling* (an
annotation queue + trace→golden pipeline; today a failure is a `Case` you construct). More
trajectory metrics (memory hit rate, scope adherence). Judge calibration tooling (rank correlation
vs. human labels). An unbiased pass@k estimator for k < n.

**Status.** Built (`tensorsketch.eval`): `Case`/`Suite`/`Trial`/`Grade`/`Grader`, the code + judge graders,
`Sandbox`/`InProcessSandbox`, `evaluate` + `Report` (metrics, render, `require`). Tested
(`test_eval.py`, 13) and exampled (`examples/evaluation.py`). `make check` green.

## D7a — Results are emitted through a sink; online eval reuses the same graders

**Decision (extends D7).** Eval results serialize to plain JSON (`to_dict()` on `Report` /
`CaseResult` / `TrialResult` / `Grade`) and are delivered through a **`Reporter`** sink — one
method, `emit(record)`, sync or async. Built-ins: `JsonlReporter` (dependency-free file log) and
`CallbackReporter(fn)`; any database is a ~5-line custom `emit`. **Online** evaluation is
`score(output, trace, graders)` (grade one live run, reference-free) and `OnlineMonitor(graders,
reporter=, sample=)` (sample production runs, score off the response path, emit). `Trial.case`
became optional so a captured run needs no `Case`.

**Why.**
1. **Stateless: emit, don't own.** TensorSketch must not bundle a results database. Serializing to JSON and
   handing it to a sink lets a team put results in *their* store — file, warehouse, dashboard, any
   DB — exactly as the framework's persistence stance requires.
2. **A dedicated `Reporter`, not the `Backend` ABC.** `Backend` is for run durability
   (checkpoints/journal keyed by thread); eval results almost always belong in a *different* place
   (analytics/dashboard). Keeping the sinks separate lets checkpoints, traces, and eval scores each
   target the store that fits — decoupled.
3. **Symmetry with tracing.** This is the tracing-exporter pattern (`FileTracer` / `OTelTracer` /
   `_record`) applied to scores. Traces cover the transcript; `Reporter` covers the verdict.
4. **Online = the same graders on a live trace.** Graders already read a `Trace`, so production
   monitoring needs no new grading — just a reference-free subset (safety, tool-failure,
   cost/latency, on-policy judges) plus sampling and a sink. What's deferred is drift/alerting over
   the emitted stream, not the scoring.

**Status.** Built (`tensorsketch.eval`): `to_dict()` throughout; `Reporter` + `JsonlReporter` +
`CallbackReporter`; `evaluate(reporter=…)`; `score` + `OnlineMonitor`. Tested (`test_eval.py`, 18)
and exampled (`examples/evaluation.py` shows offline emit + a scored production trace). Green.

## D7b — Drift detection: lightweight streaming stats, in-process window, emit-only

**Decision (extends D7a).** A **`DriftMonitor`** consumes the online result stream and raises a
**`DriftAlert`** when behavior drifts from a **`Baseline`** (`Baseline.from_report(report)` or set
directly). Two detectors, both stdlib: a **two-proportion z-test** on the overall / per-grader pass
rate (binary outcomes -> a proportion test is exactly right; per-grader so one collapsing check trips
alone), and the **Page-Hinkley** change-point test on cost/latency (fed the value *relative to* the
baseline mean, so a single `(delta, lambda)` spans dollars and milliseconds). `DriftMonitor`
**implements the `Reporter` protocol**, so it chains straight after `OnlineMonitor`; a new
**`MultiReporter`** fans each result to a store *and* the monitor. A per-metric latch makes a
sustained regression fire once until it recovers.

**Why.**
1. **Chosen over distributional (PSI/KL/KS) for the first cut.** The z-test and Page-Hinkley are
   interpretable, threshold-light, and dependency-free — you can read *why* an alert fired. PSI/KL/KS
   over a reference distribution (population shift) is a real next layer, but heavier to calibrate;
   deferred, logged.
2. **Stateless still holds.** The rolling window lives in *this process's* memory; the monitor
   persists nothing. It **emits** alerts through the same `Reporter` seam and never owns a drift
   store — identical stance to results (D7a) and traces.
3. **No new grading, no new sink type.** It reads the `TrialResult` records the stream already
   emits and forwards alerts through the existing `Reporter`; `MultiReporter` composes rather than
   adding a bespoke pathway.

**Deferred (logged).** Distributional/population-shift detectors (PSI / KL / KS); routing alerts to
a pager/webhook (today: emit a `DriftAlert` record, wire your own alerting); auto-tuned thresholds.

**Status.** Built (`tensorsketch.eval.drift`): `DriftMonitor`, `Baseline`, `DriftAlert`, `PageHinkley`,
`two_proportion_z`, plus `MultiReporter`. Tested (`test_eval.py`, 24) and exampled
(`examples/evaluation.py` fires a pass-rate alert). Green.

## D8 — Dynamic fan-out is `Send` from a router; instances stay durable

**Decision.** Graph-level dynamic fan-out is a **`Send(node, input)`** value returned from a
router's `path` (alongside, or instead of, plain node names). The engine schedules **one superstep
task per `Send`**, each with its own payload overlaid on the shared snapshot (filtered to the target
node's `In`); all instances merge at the barrier via the target's write channels — so an
aggregating `Reducer`/`Topic` channel is the reduce half. Added `Graph.router` (the intent-named
form of `conditional`) and `Graph.loop(node, until, *, exit=END)` (repeat-until sugar). This is the
*graph-level* map/reduce; `gather_map` remains the in-node-body version.

**Why.**
1. **Reuse the conditional machinery, don't grow a new one.** A router already computes successors
   from post-barrier state; letting it also yield `Send`s means fan-out rides the existing edge/
   scheduling path. No new builder concept, no parallel API. Payloads are *state-shaped* (the
   payload overrides the worker's channel reads for that instance), so compile-time port validation
   is unchanged — a worker still declares `In` fields that exist on the state.
2. **Instances must be individually durable.** Two instances of one node in one superstep would
   have collided in the effect journal (keys were `superstep:node:index:name`). Added an `instance`
   tag to `Context` folded into the key **only when non-empty**, so a normal node's keys are
   byte-identical (no journal migration) while fan-out instances stay distinct and replay their
   *own* result on resume.
3. **Pending fan-out belongs in the checkpoint.** A crash between scheduling a `Send` and running it
   must resume the *same* work. Added `Checkpoint.sends` (`(node, payload)` pairs); because every
   backend pickles the whole `Checkpoint`, this carries through memory / SQLite / Postgres / Redis
   with no per-backend change. Instance tags are the send's ordered position, so they're stable
   across resume.

**Limitation (documented).** Successors are computed from *shared* post-barrier state, so per-
instance routing *after* a fan-out isn't supported (all instances of a node route identically —
which is exactly what converges them on a collector). Nested fan-out from a fanned-out node would
duplicate; the map -> workers -> reduce shape is the supported one.

**Deferred (logged).** `map`/`parallel` as first-class `Graph` constructs (still body-helpers);
canvas extraction of `router`/`loop`/`Send` (they run today but the Studio round-trip doesn't read
them yet — a Phase 1 follow-up).

**Status.** Built: `tensorsketch.Send`; engine `_plan` + per-instance execution; `Context.instance`;
`Checkpoint.sends`; `Graph.router`/`Graph.loop`. Tested (`test_fanout.py`, 8; plus a parametrized
crash-mid-fan-out resume in `test_durability.py` across memory/SQLite/Redis) and exampled
(`examples/dynamic_fanout.py`). Phase 2 complete. `make check` green.
