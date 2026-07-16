# Changelog

All notable changes to TensorSketch are recorded here. Pre-1.0 the API may change between minor
versions; once TensorSketch reaches 1.0 it follows semantic versioning.

## [Unreleased]

### Changed — naming & packaging (0.1.0)

- **Renamed to TensorSketch.** The import namespace is now `tensorsketch` (was `loom`), the base
  exception is `TensorSketchError`, the Studio is **TensorSketch Studio**, the CLI is
  `python -m tensorsketch.canvas`, and the trace attribute vocabulary is namespaced `tensorsketch.*`.
- **Packaging foundation.** Distributed on PyPI as **`tensorsketch-core`** (import `tensorsketch`),
  version bumped to `0.1.0`. Added a real Apache-2.0 `LICENSE`, trove classifiers, project URLs, and
  a `py.typed` marker; the wheel ships the Studio assets. `pip install tensorsketch-core`; extras
  (`anthropic`, `openai`, `google`, `canvas`, `postgres`, `redis`, `mcp`, `otel`, `serve`) unchanged.

### Added — Phase 1 (code⇄canvas), complete

- **CST extraction** (`tensorsketch.canvas.extract`, optional `canvas` extra) — parses TensorSketch source into a
  JSON-able `GraphIR` (nodes, typed ports, hole detection, and the wiring). Import-free, so it
  works on incomplete/hole code. Reads all three authoring styles — the fluent chain,
  statement-style calls, and the `>>` surface — into the same IR. Node bodies stay opaque by design.
- **`>>` wiring surface** (`tensorsketch.core.wiring`) — `Graph.nodes(...)` returns handles; `START >> a`,
  `a >> b`, `a >> [b, c]`, and `a >> Router(fn, ...)` wire the graph like a diagram. Pure sugar
  over `.add`/`.edge`/`.conditional` — the compiled graph is identical.
- **Write-back** (`tensorsketch.canvas.reconstruct`) — applies an edited `GraphIR` back to source,
  regenerating only the graph definition; node bodies, imports, and comments are byte-preserved. The
  round-trip invariant `extract(reconstruct(extract(code))) == extract(code)` is a test gate.
- **Style-preserving write-back** — write-back now detects the source's authoring style and re-emits
  in it, instead of collapsing everything to one fluent chain: a fluent chain stays a chain, a
  statement-style graph (`g = Graph(S)` + `g.add(...)` lines) stays statements, and a `>>` graph
  stays `>>` (with `g.nodes(...)` handles; linear runs merge into one `a >> b >> c` spine). All three
  styles render from a single ordered wiring walk, so edges come out in the same order regardless of
  style — which is what keeps the round-trip a list equality. Completes the Phase 1 code⇄canvas engine.
- **Studio** (`tensorsketch.canvas.server`, `python -m tensorsketch.canvas <file>`) — the visual canvas: a
  stdlib localhost bridge that serves `extract` to a hand-drawn, Excalidraw-aesthetic frontend
  (`tensorsketch/canvas/studio/`) and applies `reconstruct` on every edit. Renders nodes with typed
  ports, holes, and conditional routes via an automatic layered layout; add/remove edges on the
  canvas write straight back into the code. `GraphIR.from_dict` added for the round trip.
- **Canvas reads the new authoring forms** — extraction and the round-trip invariant now cover
  `router(...)` (extracted identically to `conditional`) and `loop(node, until, *, exit=END)`
  (a two-branch conditional: a **self-loop** the Studio frontend draws as an arc, plus the exit
  edge). Routing decided inside an opaque callable — `Send` fan-out from a router, or a lambda
  predicate — is shown as a **dynamic-route stub** (the graph runs; the canvas is honest the target
  is computed at runtime). Conditional mapping targets now render `END`/`START` as barewords.
- **Node creation from a palette** — `reconstruct` now synthesizes a `class X(Node)` stub (typed
  `In`/`Out` ports + a `raise Hole(...)` body) for any node the IR names but the source never
  defined, inserting it above the graph builder and adding `from tensorsketch import Hole` when it isn't
  already imported. The stub re-extracts to the exact `NodeIR` (`has_hole=True`), so the round-trip
  invariant still holds. Studio gains a **+ node** dialog (name + optional `In`/`Out` ports); a
  created node lands unwired on the canvas, ready to drag-connect.
- **Project-wide hole surfacing** — `find_holes(*paths)` (in `tensorsketch.canvas`) walks a codebase for
  nodes still stubbed with `raise Hole(...)`, returning a `HoleRef` (file, node, spec, line) for
  each; unreadable/unparseable files are skipped. Exposed as `python -m tensorsketch.canvas --holes [paths]`
  and, in the Studio, a `GET /api/holes` endpoint with a clickable toolbar badge that lists every
  hole across the project (not just the open file).
- **Layout sidecar** — drag a node's body in the Studio to arrange it; positions are saved to a
  sidecar `‹file›.py.layout.json` beside the source (`POST /api/layout`), never written into the
  code. Unmoved nodes fall back to the automatic layered layout; malformed or stale entries are
  ignored. Node positions are presentation, so they stay out of the source of truth.
- **Multi-agent coordination** — `as_tool(graph, name=…, description=…)` wraps a compiled agent (or
  any graph) as a `Tool`, so a supervisor agent delegates to specialists by calling them
  (agents-as-tools / handoff). Delegations are ordinary tool calls, so they're journaled (a
  specialist is replayed, not re-run, on resume) and the specialist's spans nest under the
  delegating call (one trace for the whole team). Built on a general seam: **a tool function that
  declares a `ctx` parameter now gets the run `Context` injected** (never shown to the model).
  Exported as `tensorsketch.as_tool`; example `examples/multi_agent.py`.

### Added — Phase 2 (agents), in progress

- **Tools** — `@tool` deriving a JSON schema from the function signature (sync + async);
  argument validation before the call.
- **Messages** — `Message`, `ToolCall`, constructors, and the `add_messages` reducer.
- **Providers** — `ChatProvider` interface (zero SDK deps), `Completion`/`Usage`, a scripted
  `FakeProvider`, and three optional real providers: **Anthropic**, **OpenAI** (also
  OpenAI-compatible endpoints via `base_url`), and **Google/Gemini** — plus a documented
  custom-provider path.
- **Agents** — `Llm` single-call node; `Agent` autonomous loop node with **durable** model/tool
  calls (each wrapped in `ctx.step`); `create_agent` prebuilt.
- **Structured output** — `generate_structured` / provider `output_schema`, with a
  validate-and-repair loop.
- **Composition patterns** — `gather_map` (durable concurrent map/reduce), `parallel`, and
  `run_subgraph` (call one graph from another), all durable via `ctx.step`.
- **Graph-level dynamic fan-out** (`Send`) — a router's `path` may now return `Send(node, payload)`
  values; the engine schedules **one superstep task per `Send`**, each with its own payload (overlaid
  on the shared snapshot), and they merge at the next barrier through a reducer / `Topic` channel —
  the map/reduce shape at the graph level (`gather_map` remains the in-node-body version). Fan-out is
  **durable**: each instance journals its `ctx.step` effects under a distinct key (a new `instance`
  tag on `Context`, empty for normal nodes so their journal keys are unchanged), and pending sends
  are carried on the checkpoint (`Checkpoint.sends`) so a crash mid-fan-out resumes and replays
  completed workers exactly once — verified across the in-memory, SQLite, and Redis backends. Also
  adds **`Graph.router(...)`** (the intent-named form of `conditional`, and where fan-out lives) and
  **`Graph.loop(node, until, *, exit=END)`** (repeat-until sugar over a self-conditional). See
  [decisions](docs/design/decisions.md) D8. **Phase 2 is now complete.**
- **Bring-your-own-database backends** — `PostgresBackend` (`tensorsketch-core[postgres]`, psycopg 3) and
  `RedisBackend` (`tensorsketch-core[redis]`, redis-py) behind the existing `Backend` ABC, so durable
  checkpoints / effect journal / event log live in your database (the framework stays stateless).
  Drivers are imported lazily; a shared `SqlBackend` base covers any DB-API store; a pluggable
  `Serializer` (default `PickleSerializer`) is the codec seam. `SqliteBackend` moved onto the
  shared base (unchanged behavior). The durability suite runs against Redis via fakeredis in CI.
- **MCP interop** (`tensorsketch.interop.mcp`, optional `tensorsketch-core[mcp]`) — consume external MCP tool servers
  as TensorSketch tools (`mcp_tools`, `stdio_session`) and expose TensorSketch tools to any MCP client
  (`build_mcp_server`, `serve_stdio`). The MCP SDK is imported only via this module, so the core
  stays dependency-free. `Tool` gained a raw-JSON-schema path so remote tools mix with local ones.
  Round-trip tested over the real protocol using the SDK's in-memory transport.
- **Middleware** (`tensorsketch.middleware`) — wrap-style (onion) interceptors around every agent model
  and tool call: subclass `Middleware` and override `wrap_model` / `wrap_tool`; pass them via
  `create_agent(middleware=[...])`. Built-ins: `RetryMiddleware` (retry on error — the
  `on_model_error`/`on_tool_error` primitive) and `ObservabilityMiddleware` (start/end/error +
  duration events into the stream). The stack runs inside `ctx.step`, so retries and tracing are
  journaled and never re-run on resume.
- **Native tracing** (`tensorsketch.observability`) — a vendor-neutral `Tracer` with a built-in
  `InMemoryTracer`; OpenTelemetry is an *adapter*, never a dependency. The engine opens run/node
  spans and agents open model/tool spans (recording model, tokens, and estimated cost); spans nest
  automatically via a `ContextVar`. A `Trace` aggregates timing / tokens / cost / errors
  (`summary()`, `render()`), `ctx.span(...)` marks custom work, and `estimate_cost` /
  `DEFAULT_PRICES` give an overridable cost model. Pass `tracer=` to `invoke` / `stream`; the
  default is a zero-cost `NoopTracer`.
- **Trace exporters** — `FileTracer` streams spans as JSON Lines (dependency-free), and an optional
  `OTelTracer` (`tensorsketch-core[otel]`, `tensorsketch.observability.otel`) bridges TensorSketch spans to OpenTelemetry. Both
  build on a reusable `RecordingTracer` base (override `_record(span)` for a custom sink).
- **Multi-sink trace fan-out** (`MultiTracer`) — send one run's trace to several destinations at
  once: `MultiTracer(InMemoryTracer(), FileTracer("run.jsonl"), lambda span: feed.send(...))`. It
  owns a single span lifecycle (one `trace_id`, correct nesting and timing) and hands each finished
  span to every sink, so the tree is identical everywhere. A sink is any `RecordingTracer` (its
  `_record` consumes the span) or any `Callable[[Span], None]` — the callable form is the seam a
  live trace viewer plugs into. (`OTelTracer` drives OTel's own live context, so fan OTel out inside
  the OTel SDK rather than here.)
- **`http_span_sink(url)`** (`tensorsketch.observability.export`) — a stdlib, non-blocking trace sink that
  POSTs each finished span (as JSON) to an HTTP endpoint on a background daemon thread, dropping
  silently if it's unreachable. It's the `MultiTracer` callable that feeds a live viewer.
- **Live trace overlay in Studio** — click **▶ live** and a run lights up on the canvas: each node
  ringed by its status (green ok / red error) with a **latency · cost · call-count** badge, keyed
  off the `tensorsketch.node` span attribute (model/tool spans fold into their parent node). It's a
  read-only projection of the run's spans — your agent runs in its own process and ships spans via
  `MultiTracer(InMemoryTracer(), http_span_sink("http://127.0.0.1:8765/api/trace"))`; the Studio
  bridge gained an ephemeral in-memory trace buffer (`POST`/`GET /api/trace`) that persists nothing.
  So Studio stays a pure view — it reads your code and your telemetry and owns neither — which is how
  a live overlay coexists with a stateless framework.
- **`Completion.model`** — providers now report the model that produced a reply, so tracing/cost
  read it directly instead of a provider's private attribute. `FakeProvider` gained `model=` /
  `usage=` for offline cost/trace testing.
- **Evaluation harness** (`tensorsketch.eval`, in-package — no extra) — grade an agent's whole *trajectory*,
  not just its answer, over multiple trials. `Case` (task) → `evaluate` → `Report`. **Hybrid
  graders**: code-based over the answer (`Contains`/`Equals`/`Regex`), the tool **trajectory +
  payload** (`ToolCalled`/`ToolArgs`/`ToolSequence`), `StepEfficiency`, `CostBudget`/`LatencyBudget`,
  `FinalState` (environment outcome), `Custom`/`@grader`; plus `LlmJudge` (binary, one-criterion,
  structured `Verdict` via the provider seam). **Multi-run metrics**: task-completion rate, **pass@k**
  (succeeds at least once) and **pass^k** (succeeds every time), mean cost/latency, and a per-grader
  breakdown; `report.render()` / `summary()` and `report.require(...)` as a CI gate. Trials run
  behind a **`Sandbox`** seam (in-process default) with a fresh environment per trial, and graders
  always run in the harness — never inside the agent, so a run can't game its own grade. Consumes the
  trace we already emit; the agent's tool span now records call **args** and **result** so trajectory
  graders can check the payload. Deferred (stronger sandboxes, annotation queue, drift detection,
  more trajectory metrics): [decisions](docs/design/decisions.md) D7.
- **Eval results are emittable (TensorSketch stays stateless)** — `Report`/`CaseResult`/`TrialResult`/`Grade`
  gained `to_dict()` (pure JSON), and a **`Reporter`** sink (`evaluate(reporter=…)`) sends the record
  wherever you keep results: `JsonlReporter` (dependency-free file log), `CallbackReporter(fn)`, or a
  ~5-line custom `emit()` into any database (sync or async). Same exporter pattern as tracing, for
  the scores instead of the spans — so eval results, traces, and checkpoints can each target a
  different store.
- **Online evaluation** — `score(output, trace, graders)` grades a *live* run reference-free (no
  golden), and `OnlineMonitor(graders, reporter=…, sample=…)` samples finished production runs,
  scores them off the response path, and emits each result to a `Reporter`. Reuses the exact same
  graders as the offline suite (they already read a `Trace`). `Trial.case` is now optional so a
  captured production run needs no `Case`.
- **Drift detection** (`tensorsketch.eval.drift`) — a **`DriftMonitor`** watches the online result stream and
  raises a **`DriftAlert`** when behavior drifts from a **`Baseline`** (`Baseline.from_report(...)`
  from your last green eval): a **two-proportion z-test** on the overall / per-grader pass rate
  (catches a specific check collapsing) and a **Page-Hinkley** change-point on cost/latency (fed the
  value relative to the baseline mean, so one threshold spans dollars and milliseconds). It **is** a
  `Reporter`, so it chains straight after `OnlineMonitor`; a new **`MultiReporter`** fans each result
  to a store *and* the monitor in one hand-off. The rolling window lives in process memory only —
  TensorSketch emits alerts and never owns a drift store. A sustained regression is latched so it fires once
  until it recovers. Stdlib-only statistics, no new dependency. See [decisions](docs/design/decisions.md) D7b.
- **Serving** (`tensorsketch.serve`, optional `tensorsketch-core[serve]`) — expose an agent as a mountable **ASGI app**
  (Starlette) over three standard protocols: **OpenAI-compatible** (`openai_app` — `/v1/chat/
  completions` streaming + non-streaming and `/v1/models`, so any OpenAI client works), **A2A**
  (`a2a_app` — agent card at `/.well-known/agent.json` + JSON-RPC `message/send` / `message/stream`;
  plus `a2a_tool(url)` to *consume* a remote A2A agent from inside a graph), and **AG-UI**
  (`agui_app` — a `RunAgentInput` endpoint streaming `RUN_STARTED` / `TEXT_MESSAGE_*` /
  `STATE_SNAPSHOT` / `RUN_FINISHED` events to a frontend). A shared `ChatAdapter` (override
  `to_input` / `to_reply` for non-agent graphs) and SSE helper back all three. Starlette and httpx
  are imported only inside `tensorsketch.serve`, so `import tensorsketch` still pulls in no web framework. These are
  pragmatic subsets — token streaming, the full A2A task store, and multi-turn/inbound-tool
  requests are deferred (see [decisions](docs/design/decisions.md) D6).
- **Registry** (`tensorsketch.registry`) — build a built-in **provider** or **backend** by *name* so a
  config value / CLI flag can choose it: `create_provider("anthropic", model=…)`,
  `create_backend("postgres", dsn=…)`. Add names in-process with `register_provider` /
  `register_backend`, or from a published package via `tensorsketch.providers` / `tensorsketch.backends` entry
  points (explicit registration overrides an installed name). A generic lazy `Registry[T]` backs
  both — built-ins are thunks and entry points `.load()` on demand — so `import tensorsketch` still imports
  no SDK/driver and listing names imports nothing. Scoped to just these two seams on purpose:
  named factories, **not** a plugin ecosystem, and everything first-party stays in the one `tensorsketch`
  package (extras only gate heavy third-party SDKs). See [decisions](docs/design/decisions.md) D5.

### Removed

- A first-cut in-framework **memory** subsystem (keyword-search `MemoryStore`) was removed —
  in-framework memory conflicts with staying stateless, and useful recall needs embeddings, not
  keyword search. Memory/state will live outside the framework via bring-your-own-database
  connectors. See the [decisions log](docs/design/decisions.md).

## [0.0.1] — Phase 0: runtime & type spine

The foundation everything else builds on: a typed, durable, streaming BSP agent runtime.

### Added

- **Type spine (L1)**
  - `Schema` — one Pydantic-v2-backed abstraction for tool I/O, structured output, typed state,
    and node ports.
  - Typed state **channels** with reducers: `LastValue` (default), `BinaryOperatorAggregate`
    (`Reducer(op)`), and `Topic` (a concatenated stream).
- **Authoring (L2)**
  - Typed `Node` (nested `In`/`Out` Schemas as ports, opaque async body).
  - Generic `Graph` builder — `add`, `edge`, `conditional`, `START`/`END` — that compiles to the
    runtime and preserves the concrete state type through `invoke`.
  - Compile-time validation: port existence, port/channel type compatibility, edge integrity,
    single-successor-source, duplicate names.
  - Typed **holes** (`Hole`) to declare an interface and defer its body.
- **Runtime (L0)**
  - BSP/Pregel superstep engine: parallel fan-out on a shared snapshot, reducer barrier,
    native cycles, and a recursion-limit guard. Decoupled from the typed layer.
- **Durability**
  - Per-barrier **checkpoints** with resume and `fork`; `get_state` / `get_history`.
  - A per-effect **journal** via `ctx.step(name, fn)` — side effects run exactly once across
    crashes and resumes (`idempotency_key` to dedupe across a run).
  - `InMemoryBackend` and `SqliteBackend`.
- **Streaming**
  - Live `stream()` yielding namespaced `Event`s (`run_start` / `node_start` / `node_end` /
    `values` / `run_end` + custom `ctx.emit`), a monotonic `seq` cursor, and backpressure.
  - Resumable `replay(thread_id, backend, since=...)`.
- **Project**
  - 44 tests, strict `mypy`, `ruff` lint + format, benchmarks, GitHub Actions CI, and a full
    documentation set under `docs/`.
