# TensorSketch — a code-first, visually-editable, durable agentic framework

**Design plan, v1.** Codename **TensorSketch** (placeholder). Synthesized from 9 deep research
reports (LangGraph, OpenAI Agents SDK, Claude Agent SDK+MCP, Agno/CrewAI/AutoGen, wider
field, visual builders; + code↔visual bidirectional sync, execution runtime, DX/
extensibility). 2026-07-08.

---

## 0. Thesis & positioning

**One-line:** *An agentic framework where **code is the single ground truth**, a **visual
canvas is a losslessly-synced projection** of that code, execution runs on a **durable
BSP runtime**, and **every capability is a plugin** — so it's easy to start, impossible to
outgrow, fast, and absorbs whatever agent research comes next.*

**The gap we exploit.** Each incumbent fails on a different axis, and no one has all of
this together:

| Incumbent | What it does well | Where it breaks (what TensorSketch fixes) |
|---|---|---|
| **LangGraph** | Graph model, checkpoints, streaming | "Abstractions over abstractions," dependency bloat, breaking changes; **checkpoints ≠ durable execution** (mid-node crash → duplicate side effects); Functional API isn't visualizable; `Send` fan-out is sharp-edged |
| **OpenAI Agents SDK / CrewAI** | Simple, loved | Low ceiling; CrewAI "retries same approach and loops rather than adapts"; no durable execution; not visual |
| **AutoGen** | Actor runtime, distribution | Emergent control flow hard to debug/test; effectively maintenance-mode |
| **Agno / n8n / Dify / Vellum** | Ship visual builders | Either JSON-as-truth (code is second-class → no diff/review/test) **or** read-only visualizers; canvas hits a ceiling; no durable runtime |
| **Temporal / Restate** | True durable execution | Not agent-shaped; determinism straitjacket (Temporal); no graph/streaming/agent primitives |

**TensorSketch's bet:** combine (a) code-as-truth + synced canvas, (b) a durable BSP runtime, and
(c) an all-plugins core — the three things no single framework has together.

---

## 1. The five architectural commitments (everything follows from these)

1. **Code is the single ground truth; the canvas is a projection.** Only the *graph wiring
   + typed interfaces* round-trip; **node bodies are opaque**. (This is a computability
   necessity — Rice's theorem — not a choice; see §4.)
2. **One `Schema` abstraction does four jobs:** tool I/O, structured output, typed state
   channels, and design-time typed-port validation. (Pydantic v2 core / Standard Schema.)
3. **BSP/Pregel scheduler on an actor/message substrate**, persisted by a **durable
   journal** — cycles, deterministic parallel fan-out, per-superstep checkpoints, and
   single-process→distributed with unchanged code.
4. **The core knows *interfaces*, never *implementations*.** Every node type, pattern,
   tool, provider, memory backend, channel, reducer, optimizer, and protocol is a plugin
   discovered via entry points. Two customization tiers: **middleware** (per-agent onion) +
   **plugins** (global, with error hooks).
5. **Durability contract collapses to one rule:** wrap side effects in a `durable step`;
   the framework journals the result and never re-runs it on resume. No determinism
   straitjacket (journal-results-as-data, à la Restate).

---

## 2. The layered stack

```
┌───────────────────────────────────────────────────────────────────────────┐
│  L7  SURFACES:  Visual Studio (canvas)  ·  Layered API  ·  CLI  ·  Servers  │
│  L6  OBSERVABILITY & EVAL:  OTel GenAI spans · eval harness · optimizer      │
│  L5  INTEROP ADAPTERS:  MCP · A2A · AG-UI · OpenAI-compat  (plugins)         │
│  L4  EXTENSIBILITY:  entry-point registries · middleware · plugins · providers│
│  L3  CODE⇄CANVAS ENGINE:  CST extract · surgical write-back · holes · NL→code │
│  L2  AUTHORING MODEL:  typed Node classes + declarative wiring block + IR     │
│  L1  TYPE SYSTEM:  Schema (ports/state/tools/output) + reducers + validation  │
│  L0  RUNTIME CORE:  BSP scheduler · durable journal · message bus · streaming │
└───────────────────────────────────────────────────────────────────────────┘
```

Bodies (agent logic) run in the host language (Python/TS); L0 drives scheduling and calls
back to execute a node, journals the result, advances.

---

## 3. The authoring model (L2) — what a developer actually writes

**Typed node classes + a declarative wiring block** in real, plain-text host code (no
separate DSL — keep the truth in the language devs already use). Blends Pydantic-Graph
(types define contracts), Vellum (declarative `graph`), and Dagster (import-free
extractable).

```python
class Classify(Node):
    class In(Schema):  query: str
    class Out(Schema): intent: Literal["billing", "tech", "other"]
    async def run(self, ctx, inp: In) -> Out:        # BODY: opaque, arbitrary code
        ...                                          # LLM calls, tools, parsing — anything

class BillingAgent(Node):
    class In(Schema):  query: str
    class Out(Schema): answer: str
    async def run(self, ctx, inp): ...

class Support(Graph):
    # WIRING: declarative, statically inspectable, round-trips losslessly.
    graph = (
        Classify
        >> Router.on(intent="billing") >> BillingAgent
        >> Router.on(intent="tech")    >> TechAgent
        >> Router.otherwise()          >> Fallback
    )
```

Rules that make it work:
- **Typed ports are mandatory** (`In`/`Out` schemas) → the canvas draws them, the compiler
  checks edges, NL→code has a contract.
- **Wiring lives in a dedicated declarative block**, never in native `if/for/while`. (The
  Prefect-1→2 lesson: the moment control flow drives the graph, the static graph is gone.)
- **Bodies may contain anything**; the canvas renders each node as a box with its ports and
  never introspects the body.
- **Escape hatch:** an imperative `@durable async def` that `await`s `step(...)` compiles to
  the same journal, for devs who don't want to think in graphs. Both surfaces, one runtime.

---

## 4. The code⇄canvas engine (L3) — the crux

**Why only wiring round-trips:** "what are this node's out-edges" is a semantic property of
a Turing-complete program → undecidable (Rice). So we keep *wiring* a syntactic, declarative
surface and treat *bodies* as opaque. This is exactly why Dagster/Pydantic-Graph/Vellum/
LangGraph-`StateGraph` round-trip and Prefect-2/LangGraph-Functional don't.

Mechanism:
- **Extract (code→graph):** parse with a **CST** (`libcst`/`tree-sitter`), *import-free* —
  so it works on **incomplete/broken code** (essential for holes). Read node classes, their
  `In`/`Out`, and the `graph` wiring block.
- **Project (graph→canvas):** lay out; store layout (x/y, color, collapsed) in a **sidecar**
  (`*.canvas.json`), never fused into the truth (n8n's mistake).
- **Reconstruct (canvas→code):** a canvas edit is a structured mutation ("add edge A→B")
  applied as a **surgical CST patch of the wiring block + class headers only** — bodies,
  comments, imports **provably untouched**. Enforced by a CI invariant:
  `extract(reconstruct(extract(code))) == extract(code)`, and reconstruct is a byte no-op
  when wiring didn't change.
- **Degrade loudly:** wiring the parser can't model → shown as one opaque "custom subgraph"
  node, code preserved verbatim. Never silently drop.

**Incomplete → prompt for code = a typed hole:**
```python
class BillingAgent(Node):
    class In(Schema):  query: str
    class Out(Schema): answer: str
    async def run(self, ctx, inp: In) -> Out:
        raise Hole("Answer billing questions using the KB tool")   # greppable, type-checked
```
The interface is fully declared and round-trips; the body is a stub. The system surfaces
"3 nodes need code" by grepping `Hole(...)`.

**NL→code, made reliable by the typed contract:** the `In`/`Out` schema + docstring is the
generation spec → generate body → **compile + type-check against the ports + pass
auto-generated contract tests** → only then replace the `Hole` with real code. NL is an
*input method*, never a stored representation. (BAML/DSPy "typed target + validate/repair"
model.)

**Dynamic behavior is never faked:** autonomous loops and dynamic fan-out get an **authored
envelope + a runtime-trace overlay** (Temporal Event-History model). Two visual layers:
the static editable graph (*what's possible*) and a read-only execution trace (*what
happened*). Never fabricate edges you can't derive statically.

---

## 5. The node / primitive vocabulary (L2)

The "periodic table" of primitives, each a plugin implementing the `Node` contract:

- **Compute:** `agent` (encapsulated autonomous loop: model + tools + memory + stop/budget),
  `llm` (single call), `tool` (fn/hosted/MCP), `code` (typed body).
- **Control flow:** `sequence`, `router`/`switch` (typed conditional), `map` (fan-out over a
  collection → reduce), `loop` (iterate-until + **mandatory guard**), `parallel` (+ join).
- **State:** typed **channels** with **reducers** (`last`/`append`/`topic`/custom); `memory`
  (short/long/typed); `retriever`.
- **Structure:** `subgraph` (nestable, promoted params, hidden internals), `reroute`,
  `comment`.
- **Coordination:** `handoff`/`sub-agent`, `supervisor`/`orchestrator`, `team` container.
- **Reliability:** `human` (HITL/interrupt), `guardrail`/`validator`, `evaluator`.

Edges: `sequential` · `conditional` · `handoff` · **`soft/dynamic`** (LLM-decided routing,
drawn distinctly) · `data-dependency`. Ports are typed; connection legality is checked at
edit time.

---

## 6. The runtime (L0)

- **Scheduler: BSP/Pregel supersteps** (plan → execute-in-parallel → apply-reducers-at-
  barrier). Cycles, deterministic parallel fan-out, and a natural checkpoint boundary fall
  out of the model. Fan-out via a `Send`-style dynamic-branch primitive, but with
  **per-branch journaling/commit** (so one straggler/failure doesn't discard siblings) and
  **pass-IDs-not-payloads** (avoids O(branches×state) blowup) — fixing LangGraph's sharp
  edges.
- **Actors underneath:** nodes communicate only via an abstract message bus → transport is
  swappable (in-proc → gRPC), so *the same graph runs single-process or distributed
  unchanged* (AutoGen-Core's trick). Partition by `thread_id`/agent key for single-writer
  state and horizontal scale (Restate Virtual Objects / Dapr virtual actors).
- **Durability: a Restate/DBOS-style journal**, not just state checkpoints.
  - Two tiers: **per-superstep StateSnapshot** (resume/time-travel/fork, latest fetched
    O(1)) **+ per-effect journal entries** (LLM/tool/message steps memoized → *not re-run*
    on resume → no duplicate side effects, no reasoning drift).
  - **Journal-results-as-data** → no determinism straitjacket on orchestration code.
  - **Postgres transaction-piggyback** (DBOS) for exactly-once DB-backed steps.
  - Pluggable backends: in-memory (dev) → SQLite (local) → Postgres (prod); Temporal/
    Restate/Inngest only as *optional* backends, never required (embeddability first).
- **Streaming:** everything is a **namespaced event** (`run_id`, `thread_id`, `node_path`,
  `agent_id`) → coherent multi-agent lanes from one stream; backpressure + structured
  concurrency (TaskGroup/AnyIO) for clean parallel-tool cancellation; **resumable streaming**
  (replay from a cursor).
- **Fast by construction:** parallel/DAG tool calls by default (1.8–3.7× wall-clock);
  **prompt-cache-stable prefixes** (byte-stable tool ordering/serialization, auto
  `cache_control`); **delta + latest-only checkpointing** (constant per step regardless of
  conversation length); async persistence off the critical path; sticky hot-state cache.
- **Language strategy:** **pure-Python reference runtime first** (fastest path to a real
  product; validate semantics), with a clean SDK↔core boundary from day one, then **swap the
  hot path to a Rust core** (scheduler/journal/bus/streaming) behind the same interface —
  escaping LangGraph's two-codebases drift. Python-primary + TypeScript-parity SDKs; cross-
  language interop over MCP/A2A/AG-UI.
- **The one durability rule for authors:** wrap side effects in `step(name, fn,
  idempotency_key=...)` (LLM/tool calls auto-wrapped). Mark `@pure` vs `@effect` so
  "replay re-ran my LLM call" is never a surprise. Ship a **crash-harness** (kill at each
  replay boundary) — durability that isn't crash-tested is theater.

---

## 7. Type system (L1)

- **One `Schema` protocol** over Pydantic-v2 (Rust core, 5–50× faster; validates bare values
  → full models) / Standard Schema (TS). Drives: **tool I/O, structured output, typed state
  channels+reducers, typed ports.**
- **Tools: auto-schema from signature** (inspect + docstring via griffe) — zero boilerplate.
- **Structured output:** strict constrained-decoding when the provider supports it (hard
  guarantee), **validate-and-repair** (`ModelRetry`/reask, minimal repair context) otherwise.
- **Design-time port validation** (ComfyUI insight, via types not `Any`): edge legal iff
  src type assignable to dst (or a registered coercion); illegal connections rejected at edit
  time with teaching errors ("expected `str`, got `list[Doc]`; did you mean `.candidates`?").
  Parametric generics (`Node[TIn,TOut]`), never bare wildcards.

---

## 8. Extensibility (L4) — "scales with new research"

- **Discovery via entry points:** third-party packages self-register into typed registries
  (`tensorsketch.providers`, `tensorsketch.nodes`, `tensorsketch.tools`, `tensorsketch.memory`, `tensorsketch.channels`,
  `tensorsketch.reducers`, `tensorsketch.optimizers`). `pip install tensorsketch-core-voice` = new modality, no core
  release.
- **Two customization tiers:**
  - **Middleware** (per-agent, onion order): `before/after_model`, `before/after_tool`,
    `before/after_agent`, `wrap_model_call`, `wrap_tool_call`; can short-circuit / rewrite
    requests. (LangChain 1.x model.)
  - **Plugins** (app-global, precedence over middleware): logging, OTel export, policy,
    quotas, caching, **`on_model_error`/`on_tool_error`** (adaptive recovery — fixes
    CrewAI's retry-loop). (Google ADK model.)
- **Provider abstraction:** `ChatProvider` interface with a `capabilities` probe (strict
  JSON, tools, vision, audio, cache); **core depends on no provider SDK** — each is an
  optional package (kills the LangChain bloat complaint).
- **Why it future-proofs:** the core exports a fixed small set of contracts (`Node`,
  `Schema`, `ChatProvider`, `MemoryStore`, `Channel`, `Middleware`, `Plugin`, `Reducer`,
  `Optimizer`). Any new capability = implement one contract + register. A 2027 pattern (new
  node + reducer + escalation middleware + streaming modality + memory + optimizer + a new
  wire protocol) ships as ~7 zero-core-change plugins.

---

## 9. Interop (L5) & observability/eval (L6)

- **Protocol-native, bidirectional adapter plugins:** **MCP** (tools/resources/prompts —
  consume any server, expose any agent; stateless-by-default per the 2026 direction), **A2A**
  (every agent auto-publishes an Agent Card; remote agents are typed nodes), **AG-UI** (native
  streaming event format → live UI/canvas for free), **OpenAI-compat** (consume + expose
  `/v1/chat/completions`). Protocol churn absorbed at the adapter layer.
- **OTel GenAI semconv emitted natively** (spans `invoke_agent`/`execute_tool`/`chat`;
  token/duration metrics; MCP trace propagation; privacy modes) → LangSmith/Langfuse/Phoenix
  work with zero custom instrumentation, via a built-in `TracingPlugin`.
- **Eval harness** (datasets, LLM-judge/custom evaluators, scenario tests, CI gating; every
  run is a replayable trace → "prod trace → eval case" free) and a **DSPy-style optimizer
  registry** (MIPROv2) as a distinct capability — agents that improve with data.

---

## 10. The layered API (L7) — easy to start, impossible to outgrow

- **L7a Prebuilt agents (default door):** `create_agent(model, tools, memory, middleware,
  output=Schema)` — *is a graph factory* that decomposes to the same primitives (no cliff).
- **L7b Graph builder:** `Graph(State); g.add(...); g.connect(port, port); g.validate();
  g.compile(checkpointer=Postgres(), middleware=[...])`.
- **L7c Primitives:** `Node`, `Port/Schema`, `Edge`, `State+Reducer`, `ChatProvider`,
  `Middleware`, `Plugin`, `Channel`, `MemoryStore`.
- **L7d Surfaces:** `serve_openai_compat`, `serve_a2a`, `serve_mcp`, `stream_agui`, and the
  **Visual Studio** (canvas over the code⇄canvas engine + live trace overlay + debug: step,
  replay-a-node, edit-state, time-travel).
- **DX principles:** few primitives / progressive disclosure; prebuilt decomposes to
  primitives; slim core + opt-in integrations; errors that teach; SemVer'd core contracts
  (churn in plugins, not interfaces); code-first + testable (in-memory runner, fake
  providers, scenario tests).

---

## 11. How TensorSketch beats each incumbent (scorecard)

| Limitation (incumbent) | TensorSketch's fix |
|---|---|
| Checkpoints ≠ durable execution → dup side effects (LangGraph/CrewAI/ADK) | Per-effect journal + idempotent step wrappers → exactly-once effects |
| Determinism straitjacket (Temporal/DBOS) | Journal-results-as-data → no determinism rules on orchestration |
| Functional/imperative code isn't visualizable | Wiring-round-trips + opaque bodies + trace overlay for dynamics |
| Canvas is master / code second-class (n8n/Dify) | Code is truth; canvas is a lossless projection; layout in sidecar |
| Visualizer is read-only (LangGraph Studio) | Fully bidirectional editing via surgical CST write-back |
| Abstraction bloat + heavy deps (LangChain) | Slim core, contracts-only, opt-in provider packages |
| Retry-and-loop, no adaptation (CrewAI) | `on_tool_error`/`on_model_error` plugin hooks for adaptive recovery |
| `Send` fan-out sharp edges (LangGraph) | Per-branch commit + pass-IDs + built-in rate-limit/flow-control |
| Emergent control flow hard to debug (AutoGen) | Declared BSP graph on top of the actor substrate + namespaced events |
| Low ceiling (OpenAI SDK/CrewAI) | Prebuilt decomposes to primitives — no rewrite when you outgrow it |
| Two drifting codebases (LangGraph py/js) | Shared core (Rust later) behind one interface; thin SDKs |

---

## 12. Phased build roadmap

**Phase 0 — Runtime & type spine (the foundation).**
- Pure-Python BSP scheduler; typed channels + reducers; `Schema` abstraction; typed ports +
  design-time validation; in-memory + SQLite journal (two-tier: snapshot + effect journal);
  `durable step`; namespaced event streaming. *Acceptance:* a graph runs, persists, resumes
  from any checkpoint, and re-runs no journaled effect on resume.

**Phase 1 — Authoring model & code⇄canvas engine (the differentiator).**
- Typed `Node` classes + declarative wiring block; CST extract (import-free); surgical
  write-back with the CI round-trip invariant; layout sidecar; typed **holes** + hole
  surfacing. *Acceptance:* edit on canvas → code changes with bodies untouched; edit code →
  canvas updates; incomplete graph runs up to its holes.

**Phase 2 — Agent primitives & prebuilt API.**
- `agent` (autonomous-loop node), `llm`, `tool` (fn + auto-schema), `router`/`map`/`loop`/
  `parallel`, `memory`, `subgraph`; `create_agent`; structured output (strict + repair);
  provider abstraction (OpenAI/Anthropic/LiteLLM as optional packages). *Acceptance:* build a
  real multi-tool agent both in code and on canvas; run it durably.

**Phase 3 — Extensibility, interop, observability.**
- Entry-point registries; middleware + plugins (incl. error hooks); OTel GenAI tracing
  plugin; MCP client+server; A2A Agent Cards; AG-UI streaming; OpenAI-compat serve. Eval
  harness. *Acceptance:* add a node type + a provider + an MCP server as external packages,
  zero core edits; traces land in Langfuse unmodified.

**Phase 4 — NL→code, optimizer, distribution, Studio polish.**
- NL→code (typed contract + validate/repair + generated tests); DSPy-style optimizer
  registry; distributed runtime (gRPC host/worker, key partitioning) with unchanged agent
  code; Rust hot-path core behind the same interface; Studio debug (step/replay/time-travel)
  + trace overlay. TypeScript SDK parity.

**Cross-cutting from day one:** the SDK↔core boundary, the `@pure`/`@effect` marker, the
crash-harness, SemVer'd contracts, and the round-trip CI invariant.

---

## 13. Open decisions / risks

- **Rust-core timing.** Pure-Python first is right, but the FFI boundary (per-node/token
  marshaling) must be designed early or the later swap leaks. *Mitigation:* opaque state
  handles in the core, marshal only deltas, batch token events.
- **Wiring-DSL ergonomics.** The `>>`/`Router.on` surface must stay both human-writable and
  cleanly CST-patchable. Needs real dogfooding; may need a small set of canonical forms the
  reconstructor emits.
- **How much control flow lives in wiring vs bodies.** Too little in wiring → canvas is
  anemic; too much → we recreate Prefect-2's dynamic-graph loss. The `map`/`loop`/`router`
  primitives are the negotiated line; validate with real agents.
- **NL→code trust.** Only as good as the generated tests; needs a strong contract-test
  generator and a visible "unverified" state until it passes.
- **Naming / scope.** "TensorSketch" is a placeholder; and whether this ships as its own product vs.
  the engine under Agent Arena is a strategic call (Agent Arena becomes the first
  *application* of TensorSketch — a constrained, fully-visual deployment where user code is
  sandboxed away).
- **Verify before shipping:** Anthropic prompt-cache pricing/TTL numbers (from secondary
  sources) against primary docs.
