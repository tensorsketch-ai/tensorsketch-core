# Roadmap

The full architecture is in the [architecture plan](framework-design.md). This page tracks the
phased build. For a running ledger of exactly what's built and what's deferred (and why), see
**[Build status & backlog](status.md)**.

## Phase 0 — Runtime & type spine _(complete)_

The foundation everything else stands on.

- [x] `Schema` abstraction (over Pydantic v2)
- [x] Typed state **channels** with reducers — `LastValue`, `BinaryOperatorAggregate`, `Topic`
- [x] **BSP superstep engine** — parallel execution, barrier reduction, cycles, recursion limit
- [x] Typed **`Node`** (In/Out ports) and **`Graph`** builder (sequential + conditional edges)
- [x] Design-time validation — port existence, port/channel type compatibility, edge integrity
- [x] Typed **holes** (`Hole`) for "this node needs code"
- [x] Generic typing end-to-end — `invoke` returns your concrete state type
- [x] **Durable execution** — per-barrier checkpoints, resume/fork, and a per-effect journal
      (`ctx.step`) so side effects run exactly once; in-memory + SQLite backends; crash-harness test
- [x] **Namespaced event streaming** — live `stream()` (node/values/custom events, monotonic
      cursor, backpressure), `ctx.emit`, and resumable `replay` from a cursor
- [x] Test suite (44 tests), runnable examples, and these docs
- [x] Hardening — edge-case coverage, micro-benchmarks, `ruff format`, and GitHub Actions CI
      (lint · format · strict types · tests on Python 3.11 + 3.12)

## Phase 1 — Authoring model & code⇄canvas engine _(complete — the differentiator)_

- [x] **CST extraction** (`tensorsketch.canvas.extract`, import-free) — nodes + typed ports + hole
      detection + the `Graph(...)` builder wiring → a JSON-able `GraphIR`
- [x] Surgical **write-back** (`tensorsketch.canvas.reconstruct`): rebuild only the builder chain from
      the (edited) IR, bodies/imports/comments byte-preserved; the **round-trip CI invariant**
      `extract(reconstruct(extract)) == extract` (parametrized test gate)
- [x] Ergonomic **`>>` wiring surface** (`START >> a >> Router(fn, ...)`) — handles from
      `Graph.nodes(...)`, pure sugar over `.add`/`.edge`/`.conditional`
- [x] **Statement-style** builder support (`g = Graph(...); g.add(...)`, incl. annotated
      `g: Graph[S] = ...`); `add(name=...)` renames; **clean generated formatting** at any nesting
- [x] **The visual canvas** (TensorSketch Studio) — Excalidraw aesthetic (see [decisions](decisions.md)):
      a stdlib bridge (`python -m tensorsketch.canvas <file>`) + hand-drawn frontend that renders the
      `GraphIR` and writes edits (add/remove edges) back through `reconstruct`
- [x] **Node creation from a palette** — Studio's **+ node** dialog (name + optional ports)
      generates a `class X(Node)` stub (typed ports + `Hole` body) via `reconstruct`; re-extracts
      to the exact `NodeIR`, so the round-trip invariant holds
- [x] **Project-wide hole surfacing** — `find_holes(paths)` / `python -m tensorsketch.canvas --holes`;
      Studio counts holes across the project and lists them (file · node · `Hole` message)
- [x] **Layout sidecar** — drag to arrange; positions persist in `‹file›.py.layout.json` beside
      the code (never in it), with automatic layout as the fallback
- [x] **Style-preserving write-back** — detect the source's style (fluent / statement / `>>`) and
      re-emit in it; all three render from one ordered wiring walk, so edge order (and the
      round-trip) is preserved by construction

## Phase 2 — Agent primitives & prebuilt API _(complete)_

- [x] `tool` with schema auto-derived from the function signature (sync + async)
- [x] `Llm` single-call node; `Agent` autonomous loop node (durable — every model/tool call
      wrapped in `ctx.step`)
- [x] `create_agent(...)` prebuilt returning a normal graph
- [x] Provider abstraction (`ChatProvider`, zero SDK deps) with `FakeProvider` and an optional
      `AnthropicProvider`
- [x] Structured output via `generate_structured` / provider `output_schema`
- [x] Composition patterns: `gather_map` (durable concurrent map), `parallel`, `run_subgraph`
- [x] Validate-and-repair loop for structured output
- [x] Providers: **Anthropic, OpenAI, Google** + a dead-simple custom-provider path
- [x] **Bring-your-own-database** connectors — `PostgresBackend`, `RedisBackend` behind the
      `Backend` ABC (lazy drivers), a shared `SqlBackend` base, and a pluggable `Serializer` seam
- [x] **Graph-level dynamic fan-out** (`Send`) — a router returns `Send(node, payload)`s and the
      engine spawns one superstep task per Send (its own payload), merging at the barrier via a
      reducer channel. Durable: each instance journals under a distinct key; pending sends ride the
      checkpoint. Plus **`loop`/`router` builder sugar** over `conditional`.
- [x] **Multi-agent coordination** — `as_tool(graph)` wraps an agent as a `Tool` so a supervisor
      delegates to specialists (agents-as-tools / handoff). Reuses the agent loop, so delegations
      are journaled and trace as one team; built on general `ctx`-injection into tool functions.

> **Removed:** an early in-framework memory subsystem (keyword search). Memory/state belongs
> *outside* the framework — see [decisions](decisions.md).

## Phase 3 — Extensibility, interop, observability

- [x] **MCP** client + server (`tensorsketch.interop.mcp`) — consume external tool servers as TensorSketch
      tools; expose TensorSketch tools to any MCP client. Optional `tensorsketch-core[mcp]`.
- [x] **Middleware** (`tensorsketch.middleware`) — wrap-style interceptors around every model/tool call
      (`RetryMiddleware` = `on_model_error`/`on_tool_error`, `ObservabilityMiddleware`); durable
      (journaled inside `ctx.step`)
- [x] **Native tracing** (`tensorsketch.observability`) — vendor-neutral `Tracer` + built-in
      `InMemoryTracer`; run/node/model/tool spans with timing, tokens, cost, errors; `ctx.span`;
      overridable cost model. `Completion.model` makes model/cost first-class.
- [x] **Exporters** — `FileTracer` (JSON-lines, dependency-free) and an optional
      **`OTelTracer`** (`tensorsketch-core[otel]`) bridging TensorSketch spans to OpenTelemetry — adapters over the
      `Tracer`, both driven by a reusable `RecordingTracer` base.
- [x] **Registry** (`tensorsketch.registry`) — select a built-in provider/backend by name
      (`create_provider("anthropic", …)` / `create_backend("postgres", …)`) so config can choose
      it; `register_*` and `tensorsketch.providers`/`tensorsketch.backends` entry points add names. Lazy (no SDK
      imported to list or resolve). *Named factories, not a plugin ecosystem — see
      [decisions](decisions.md) D5.*
- [x] **Serving** (`tensorsketch.serve`, optional `tensorsketch-core[serve]`) — expose an agent as a mountable ASGI
      app over **OpenAI-compatible** (`openai_app`), **A2A** (`a2a_app` + `a2a_tool` to consume),
      and **AG-UI** (`agui_app`) protocols; Starlette-based, imported only in `tensorsketch.serve`.
- [x] **Eval harness** (`tensorsketch.eval`) — `Case`/`Trial`/`Grader`/`Report`; code-based graders
      (answer, tool trajectory + payload, step efficiency, cost/latency, final-state) and
      `LlmJudge`; multi-trial **pass@k / pass^k** + completion rate; `Sandbox` seam (in-process
      default); `report.require(...)` CI gate. Consumes the trace for cost/latency/trajectory.
- [x] **Eval results emit + online** — `to_dict()` everywhere + a `Reporter` sink
      (`JsonlReporter`, `CallbackReporter`, custom `emit` to any DB); `evaluate(reporter=…)`.
      **Online**: `score(...)` / `OnlineMonitor` grade live production traces reference-free and
      emit to a sink. TensorSketch stays stateless (emits, never owns the store).
- [x] **Drift detection** (`tensorsketch.eval.drift`) — a `DriftMonitor` (itself a `Reporter`) over the
      online result stream: two-proportion z-test on pass-rate / per-grader drops + a Page-Hinkley
      change-point on cost/latency, against a `Baseline` (`from_report`). Emits `DriftAlert`s to a
      sink; the window is in-memory only (nothing persisted). `MultiReporter` fans each result to a
      store *and* the monitor. Stdlib-only stats — see [decisions](decisions.md) D7b.
- [x] **Multi-sink trace fan-out** (`MultiTracer`) — one span lifecycle (one `trace_id`, correct
      nesting) fanned to several sinks at once: any mix of `RecordingTracer`s (`FileTracer`,
      `InMemoryTracer`) and `Callable[[Span], None]` viewers. The callable sink is the seam the
      live overlay plugs into.
- [x] **Live trace overlay in Studio** — click ▶ live and a run lights up on the canvas: each node
      ringed by status (ok/error) with a latency · cost · calls badge. A read-only projection of the
      run's spans, fed by `MultiTracer(..., http_span_sink(url))` from the agent's own process; the
      bridge buffers spans in memory only. Statelessness holds — Studio reads code + telemetry,
      owns neither.

## Phase 4 — NL→code, optimizer, distribution, Studio

- Natural-language → code (typed contract + validate/repair + generated tests)
- DSPy-style optimizer registry
- Distributed runtime (gRPC host/worker) with unchanged agent code; Rust hot-path core
- Visual Studio debugging (step/replay/time-travel) + live trace overlay; TypeScript SDK parity

---

**Cross-cutting from day one:** a clean SDK↔core boundary, `@pure`/`@effect` markers, a
crash-harness for durability, SemVer'd core contracts, and the round-trip CI invariant.
