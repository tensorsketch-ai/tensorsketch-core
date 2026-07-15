# Build status & backlog

The single source of truth for **what's built** and **what's deliberately deferred** (with the
reason). Updated as work lands, so nothing gets dropped. Pairs with the [roadmap](roadmap.md)
(phase plan) and the [architecture plan](framework-design.md) (the full design).

Last updated: renamed to **TensorSketch** (`import tensorsketch`, dist `tensorsketch-core` 0.1.0) +
packaging foundation (LICENSE, classifiers, URLs, wheel verified); multi-agent coordination
(`as_tool`); the Studio's node-creation palette, project-wide hole surfacing, and layout sidecar.

## Naming & packaging ✅

- **Name:** the library is **`tensorsketch-core`** on PyPI, imported as **`tensorsketch`**; the
  product/platform is **TensorSketch**. (Both `loom` and `tensorsketch` are taken on PyPI; `loom`
  was always a placeholder.) Base error is `TensorSketchError`; trace attrs are `tensorsketch.*`.
- **Packaging:** version `0.1.0`, Apache-2.0 `LICENSE`, trove classifiers, project URLs (placeholder
  until the repo is public), `py.typed`. `uv build` produces a wheel that ships the Studio assets.
  Author/email and real URLs are marked `TODO(publish)` in `pyproject.toml`.

---

## Done

### Phase 0 — runtime & type spine ✅ (complete + hardened)

- `Schema` abstraction (Pydantic v2).
- Typed state **channels** + reducers: `LastValue`, `BinaryOperatorAggregate`, `Topic`.
- Typed **`Node`** (In/Out ports) and generic **`Graph`** builder (sequential + conditional
  edges, `START`/`END`); compile-time validation (ports, types, edges, single-successor-source,
  duplicate names); typed **holes** (`Hole`).
- **BSP superstep engine**: parallel fan-out on a shared snapshot, reducer barrier, native
  cycles, recursion-limit guard.
- **Durable execution**: per-barrier checkpoints, resume, `fork`, `get_state`/`get_history`; the
  per-effect journal (`ctx.step`) → exactly-once side effects; `InMemoryBackend` + `SqliteBackend`.
- **Streaming**: live `stream()` (run/node/values/custom events, monotonic `seq`, backpressure,
  structured-concurrency cancel), `ctx.emit`, resumable `replay`.
- Hardening: edge-case tests, benchmarks, `ruff format`, GitHub Actions CI.

### Phase 2 — agent primitives (in progress) 🚧

- **Tools**: `@tool` with schema auto-derived from the signature (sync + async); arg validation.
- **Messages**: `Message`/`ToolCall`, constructors, `add_messages` reducer.
- **Providers**: `ChatProvider` (zero SDK deps), `Completion`/`Usage`, `FakeProvider`, and three
  optional real providers — **Anthropic, OpenAI, Google** (lazy imports; OpenAI also covers
  OpenAI-compatible endpoints via `base_url`); documented custom-provider path.
- **Agents**: `Llm` single-call node; `Agent` durable loop (every model/tool call via `ctx.step`);
  `create_agent`.
- **Structured output**: `generate_structured` + provider `output_schema`, with validate-and-repair.
- **Composition patterns**: `gather_map` (durable concurrent map), `parallel`, `run_subgraph`.
- **Bring-your-own-database backends** (D1 delivered): `PostgresBackend` (`tensorsketch-core[postgres]`) and
  `RedisBackend` (`tensorsketch-core[redis]`) behind the `Backend` ABC — lazy drivers, a shared `SqlBackend`
  base (any DB-API store), and a pluggable `Serializer` codec (default `PickleSerializer`). The
  whole durability suite runs against Redis via fakeredis; Postgres via `LOOM_TEST_POSTGRES` DSN.
- **Graph-level dynamic fan-out** (`Send`, D8): a router's `path` returns `Send(node, payload)`s;
  the engine spawns one superstep task per Send (its payload overlaid on the shared snapshot,
  filtered to the node's `In`), and they merge at the barrier via a reducer/`Topic` channel. Fan-out
  is **durable** — each instance's `ctx.step` effects journal under a distinct key (a new `instance`
  tag on `Context`, empty for normal nodes so existing keys are byte-identical), and pending sends
  ride the checkpoint (a new `Checkpoint.sends`, whole-checkpoint pickled so all backends carry it).
  Plus **`loop`/`router`** builder sugar over `conditional`. Verified: map/reduce correctness,
  distinct per-instance journal keys, crash-mid-fan-out resume exactly-once across memory/SQLite/
  Redis. **Phase 2 is now complete** (its roadmap boxes all done). *See [decisions](decisions.md) D8.*

### Phase 1 — code⇄canvas engine (in progress) 🚧

- **CST extraction** (`tensorsketch.canvas.extract`, optional `canvas` extra): parses TensorSketch source with
  libcst — import-free, works on incomplete/hole code — into a JSON-able `GraphIR` (nodes + typed
  ports + `has_hole` + wiring from the `Graph(...)` builder, conditional mappings expanded).
- **Write-back** (`tensorsketch.canvas.reconstruct`): folds the graph definition into one canonical
  fluent chain, byte-preserving bodies/imports/comments; the **round-trip invariant**
  `extract(reconstruct(extract)) == extract` is a parametrized test gate. Also **generates node
  stubs** — a new node the IR names gets a synthesized `class X(Node)` (typed ports + `Hole` body).
- **`>>` wiring surface** (`tensorsketch.core.wiring`: `NodeHandle`/`Router`, `Graph.nodes()`/`graph[name]`):
  `START >> a >> Router(fn, ...)`, fan-out via `a >> [b, c]` — pure sugar over the builder.
- **Multi-style extraction + canonicalization**: fluent chain, statement-style (incl. annotated
  `g: Graph[S] = ...`), and `>>` all extract to the same IR; write-back canonicalizes any of them
  to the fluent chain with clean indentation at any nesting depth. `add(name=...)` renames handled.
- **Studio** — the visual canvas (`tensorsketch.canvas.server` + `tensorsketch/canvas/studio/`, run via
  `python -m tensorsketch.canvas <file>`): a stdlib localhost bridge serving `extract`, a hand-drawn
  Excalidraw-aesthetic frontend (layered layout, typed ports, hole/conditional rendering), and
  add/remove-edge edits written straight back through `reconstruct`. IR gained `from_dict`. A
  **+ node** palette creates nodes on-canvas (name + optional ports → a generated stub in code);
  drag a node to **move** it (positions persist in a `.layout.json` sidecar, never in the code);
  a toolbar badge surfaces **project-wide holes** (click to list every node still needing code).

### Quality bar (holds for everything above)

`ruff` + `ruff format` clean · `mypy --strict` clean · full test suite green · runnable examples
· CI on 3.11 + 3.12. Run everything with `make check`.

---

## Deferred / saved for later

Each item is real and intended — just not built yet. Grouped by area, newest-relevant first.

### Phase 2 remainder (agents)

- **Memory (re-approach)**: not in-framework; an external, embedding-based store the app owns.
  The keyword-search version was removed. *See [decisions](decisions.md).*
- **More builder sugar**: `map` / `parallel` as `Graph` constructs (vs today's body-helpers).
  *`loop`/`router` and graph-level `Send` fan-out now ship; these two remain body-helpers.*
- **Structured _agent_ output** (typed state carrying a parsed Schema). *Deferred to avoid dynamic
  In/Out on nodes; `generate_structured` covers the standalone case.*
- **More providers**: OpenAI / OpenAI-compatible, LiteLLM. *Straightforward on the abstraction.*
- **Tools**: hosted/MCP tools; per-parameter descriptions parsed from the docstring.
- ~~**Coordination**: sub-agent handoff, supervisor/orchestrator.~~ **Done** — `as_tool(graph)`
  wraps an agent as a `Tool`, so a supervisor calls specialists as tools (agents-as-tools). Built
  on a general seam: a tool declaring a `ctx` param gets the `Context` injected. Delegations are
  ordinary tool calls, so they're journaled (specialist replayed on resume) and nest in one trace.
  *A dedicated `team`/orchestrator container is still possible but unnecessary — the pattern is
  just `create_agent(tools=[as_tool(...), ...])`.*
- **Subgraph as compile-time inlining** (namespaced nodes, uniform BSP + checkpointing) vs the
  current `run_subgraph` wrapper.
- **Agent decomposition to primitives** (agent loop as a visible subgraph) vs today's single node.

### Runtime & durability

- Transaction-piggybacked exactly-once for DB steps (commit the effect result in the *same*
  transaction as the work); optional Temporal/Restate backends. *Postgres/Redis connectors done.*
- Distributed runtime (gRPC host/worker, key partitioning) with unchanged agent code.
- Rust hot-path core behind the same interface.
- Perf: prompt-cache-stable prefixes; delta + latest-only checkpointing; async persistence off the
  critical path.
- `max_steps` is currently an absolute superstep budget across resumes — revisit if per-invoke
  budgeting is wanted.

### Type system

- Richer port typing including **reducer/Topic update types** (today the assignability check is
  skipped for reducer/Topic fields); parametric generics on nodes; registered coercions.

### Streaming

- Live-tailing an in-progress run from another process (merge `replay` catch-up with live
  `stream`).
- Token-level streaming from provider nodes (flows through `ctx.emit`).

### Phase 1 — code⇄canvas engine (in progress — the headline differentiator)

- **Done:** CST extraction (fluent + statement-style + `>>`), the `>>` authoring surface,
  surgical write-back with canonicalization + clean formatting, the round-trip invariant, and the
  **Studio** (bridge + hand-drawn frontend) closing the loop end to end.
- **New authoring forms extract + round-trip:** `router(...)` (== `conditional`), and
  `loop(node, until, *, exit=END)` (a two-branch conditional: a self-loop the canvas draws as an
  arc, plus the exit edge). Dynamic routing inside an opaque callable (`Send` fan-out, lambda
  predicates) shows as a **dynamic-route stub** — the graph runs; the canvas is honest that the
  target is computed at runtime. Mapping targets render `END`/`START` as barewords.
- **Node creation from a palette:** `reconstruct` now synthesizes a `class X(Node)` stub (typed
  `In`/`Out` ports + a `Hole` body) for any node the IR names but the source never defined,
  inserts it above the graph builder, and adds `from tensorsketch import Hole` when needed. The stub
  re-extracts to the exact `NodeIR` (`has_hole=True`), so the invariant holds. Studio wires it to
  a **+ node** dialog (name + optional ports) — a created node lands unwired, ready to drag-connect.
- **Project-wide hole surfacing:** `find_holes(*paths) -> [HoleRef]` (file, node, `Hole` spec,
  line) walks a codebase syntactically (skips unreadable/unparseable files); CLI
  `python -m tensorsketch.canvas --holes [paths]`; Studio `GET /api/holes` + a clickable toolbar badge that
  lists every hole across the project.
- **Layout sidecar:** manual node positions persist in `‹file›.py.layout.json` beside the source
  (never in the code) — `POST /api/layout` writes it, `GET /api/graph` serves it; drag a node's
  body to move it, unmoved nodes keep the automatic layered layout. Malformed/stale entries ignored.
- **Next / last Phase 1 item:** style-preserving write-back (keep the author's `>>`/statement style
  vs canonicalizing).

### Phase 3 — extensibility, interop, observability (in progress) 🚧

- **MCP interop done** (`tensorsketch.interop.mcp`, optional `tensorsketch-core[mcp]`): `mcp_tools(session)` wraps a
  server's tools as TensorSketch tools; `build_mcp_server`/`serve_stdio` expose TensorSketch tools to any MCP
  client; `stdio_session` convenience transport. Round-trip tested over the real protocol via the
  SDK's in-memory transport. Needed a small `Tool` generalization (raw-JSON-schema tools).
- **Middleware done** (`tensorsketch.middleware`): wrap-style (onion) interceptors around every model and
  tool call — `wrap_model`/`wrap_tool` + `compose_*`, wired into the `Agent` loop **inside**
  `ctx.step` (so retries/tracing are journaled, never re-run on resume). Built-ins:
  `RetryMiddleware` (the `on_model_error`/`on_tool_error` primitive) and `ObservabilityMiddleware`
  (start/end/error + duration events into the stream). `create_agent(middleware=[...])`.
- **Native tracing done** (`tensorsketch.observability`): a **vendor-neutral** `Tracer` (no OTel
  dependency) + built-in `InMemoryTracer`. The engine opens run/node spans, agents open model/tool
  spans (model id, tokens, cost, status). Spans nest via a `ContextVar` (correct across `await` and
  parallel tasks). `Trace` aggregates duration/tokens/cost/errors + `render()`/`summary()`;
  `ctx.span(...)` for custom spans; overridable `estimate_cost`/`DEFAULT_PRICES`. `invoke`/`stream`
  take `tracer=`. Replayed (journaled) calls produce no span, so traces reflect real work.
  `Completion.model` is now first-class (providers set it from the response), so cost no longer
  reads a private attr.
- **Exporters done**: a reusable `RecordingTracer` base (owns lifecycle, `_record(span)` hook);
  `FileTracer` writes JSON-lines (dependency-free, `Span.to_dict()` + wall-clock `started_at`);
  optional `OTelTracer` (`tensorsketch-core[otel]`, `tensorsketch.observability.otel`) bridges each TensorSketch span to an OTel
  span (nesting + attributes preserved), tested against a real in-memory OTel exporter.
- **Registry done** (`tensorsketch.registry`): select a built-in **provider**/**backend** by name so a
  config value can pick it — `create_provider("anthropic", …)` / `create_backend("postgres", …)`,
  plus `register_*` and `tensorsketch.providers`/`tensorsketch.backends` entry points to add names. A generic lazy
  `Registry[T]` (built-ins are thunks, entry points `.load()` on demand) keeps `import tensorsketch` free
  of every SDK/driver — verified in a fresh interpreter that listing/creating imports none.
  Scoped to **two** seams on purpose; **not** a plugin ecosystem or a core/community package split
  (everything first-party stays in the one `tensorsketch` package). *See [decisions](decisions.md) D5.*
- **Serving done** (`tensorsketch.serve`, optional `tensorsketch-core[serve]`): expose an agent-shaped `CompiledGraph`
  as a mountable **ASGI app** over three standard protocols — **OpenAI-compatible** (`openai_app`:
  `/v1/chat/completions` stream + non-stream, `/v1/models`), **A2A** (`a2a_app`: agent card +
  JSON-RPC `message/send`/`message/stream`; plus `a2a_tool(url)` to *consume* a remote agent), and
  **AG-UI** (`agui_app`: `RunAgentInput` → SSE of `RUN_STARTED`/`TEXT_MESSAGE_*`/`STATE_SNAPSHOT`/
  `RUN_FINISHED`). Built on Starlette + a shared `ChatAdapter` (`to_input`/`to_reply`) and SSE
  helper; Starlette/httpx imported only in `tensorsketch.serve` (verified `import tensorsketch` pulls in neither).
  Tested over the real ASGI apps (Starlette `TestClient` + httpx ASGI transport), including an A2A
  consume↔expose round-trip. *Pragmatic subsets — token streaming, full A2A task store, and
  multi-turn/inbound-tools are deferred; see [decisions](decisions.md) D6.*
- **Eval harness done** (`tensorsketch.eval`, in-package — no extra): the task/trial/transcript/outcome/
  grader model over TensorSketch's own `Trace`. `Case` (inputs + graders + per-trial `setup` env + trial
  count) → `evaluate` → `Report`. **Hybrid graders**: code-based over the answer (`Contains`/
  `Equals`/`Regex`), the **trajectory + payload** (`ToolCalled`/`ToolArgs`/`ToolSequence` — needed
  enriching the agent's tool span with args/result), `StepEfficiency`, `CostBudget`/`LatencyBudget`,
  `FinalState` (outcome/env), `Custom`/`@grader`; plus `LlmJudge` (binary, one-criterion, structured
  `Verdict` via the provider seam — its call isn't traced, so it doesn't hit the agent's cost).
  **Multi-trial metrics**: completion rate, **pass@k** / **pass^k**, mean cost/latency, per-grader
  breakdown; `render()`/`summary()`; `Report.require(...)` as a CI gate. **`Sandbox` seam** with an
  in-process default; graders always run in the harness, never inside the agent (anti-gaming). Fresh
  env per trial. *See [decisions](decisions.md) D7 for the deferred list.*
- **Eval results emit + online done**: results serialize (`to_dict()` on Report/CaseResult/
  TrialResult/Grade) and emit through a **`Reporter`** sink — `JsonlReporter`, `CallbackReporter`,
  or a ~5-line custom `emit()` into any DB (sync/async); `evaluate(reporter=…)`. Same exporter
  pattern as tracing, for scores. **Online**: `score(output, trace, graders)` grades a live run
  reference-free; `OnlineMonitor(graders, reporter=, sample=)` samples production runs off the
  response path and emits each result. `Trial.case` made optional. TensorSketch stays stateless — emits,
  never owns the results store. Tested (`test_eval.py`, 18 total).
- **Drift detection done** (`tensorsketch.eval.drift`): a `DriftMonitor` that watches the online result
  stream and emits `DriftAlert`s when behavior drifts from a `Baseline` — a two-proportion z-test on
  pass-rate / per-grader drops, and a Page-Hinkley change-point on cost/latency (fed baseline-
  relative so one threshold spans dollars and ms). It **is** a `Reporter`, so it chains after
  `OnlineMonitor`; `MultiReporter` fans each result to a store *and* the monitor. The rolling window
  lives in-process only — TensorSketch emits alerts, never owns a drift store. Sustained regressions are
  latched (fire once until recovery). Stdlib stats, no deps. *See [decisions](decisions.md) D7b.*
- **Multi-sink trace fan-out done** (`MultiTracer`): one span lifecycle (a single `trace_id`,
  correct nesting/timing) fanned to several sinks at once — any mix of `RecordingTracer`s
  (`FileTracer`, `InMemoryTracer`) and plain `Callable[[Span], None]` viewers. A `RecordingTracer`
  sink shares the one `trace_id` and only its `_record` consumer half is driven; the callable form
  is the exact seam the Studio live overlay plugs into. Tested (`test_exporters.py`).
- **Live trace overlay in Studio done**: click ▶ live and a run paints onto the canvas — each node
  ringed by status (ok/error) with a latency · cost · calls badge, keyed off the `tensorsketch.node` span
  attribute (model/tool spans fold into their parent node). Fed by `http_span_sink(url)` (a new
  stdlib, non-blocking, drop-on-failure callable sink in `observability.export`) inside a
  `MultiTracer` from the agent's *own* process; the bridge gained an ephemeral in-memory `TraceBuffer`
  (`POST/GET /api/trace`) that persists nothing. Statelessness holds — Studio reads code + telemetry,
  owns neither (the "how does Studio work if stateless" answer, made concrete). Tested end-to-end
  (`test_studio_live.py`: buffer semantics + a real run's spans delivered through the sink).
- **Phase 3 is now complete.** Remaining observability polish (richer per-provider cost/latency) and
  the deferred eval items (distributional drift, stronger sandboxes, annotation queue) live below.

### Evaluation (offline harness shipped — these extend it)

- **Stronger sandboxes**: `SubprocessSandbox`, `DockerSandbox`, and remote runners behind the
  existing `Sandbox` seam (in-process ships today). *Needed for agents that write code / run bash.*
- **Distributional drift**: `DriftMonitor` (pass-rate two-proportion + cost/latency Page-Hinkley)
  ships now; population-shift detectors (PSI / KL / KS over a reference distribution) and routing
  alerts to a pager/webhook are the next layer.
- **Feedback loop as tooling**: the annotation queue / human-in-the-loop UI and the
  trace→golden pipeline (today a failure becomes a `Case` you construct by hand).
- **More trajectory metrics**: memory hit rate (memory-enabled agents) and scope-adherence.
- **Judge calibration**: measuring an `LlmJudge`'s rank correlation against human labels.
- **Unbiased pass@k estimator** for k < n samples (today pass@k uses k = trials run).

### Phase 4 — NL→code, optimizer, distribution, Studio (not started)

- NL→code (typed contract + validate/repair + generated tests); DSPy-style optimizer registry;
  Visual Studio debug (step/replay/time-travel) + trace overlay; TypeScript SDK parity.

### Verify before shipping

- Anthropic prompt-cache pricing/TTL numbers in the design doc (from secondary sources) against
  primary docs.
- Confirm the **Anthropic, OpenAI, and Google** provider request/response mappings against the
  live APIs (currently covered by unit tests with injected fake clients, not real calls). Also
  confirm the default model ids are current.
