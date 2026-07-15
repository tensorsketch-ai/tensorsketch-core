# TensorSketch

**A code-first, visually-editable, durable agentic framework.**

TensorSketch is an agentic framework where **code is the single ground truth**, a **visual canvas
is a losslessly-synced projection** of that code, execution runs on a **durable BSP
runtime**, and **every capability is a plugin** — so it's easy to start, impossible to
outgrow, fast, and absorbs whatever agent research comes next.

> Status: **Phases 0, 2, and 3 complete; Phase 1 (code⇄canvas) nearly there.** The type spine and
> BSP runtime; durable execution (checkpoints, resume/fork, exactly-once effects); streaming; the
> full agent layer (tools, three providers, the durable agent loop, structured output, dynamic
> fan-out, and multi-agent coordination via `as_tool`); interop and observability (MCP, middleware,
> tracing + exporters, a name registry, OpenAI/A2A/AG-UI serving, an eval harness with drift
> detection); and **TensorSketch Studio** — the visual canvas: render a graph, create nodes from a
> palette, wire and rearrange them, surface holes across the project, and watch a live trace
> overlay — all round-tripping through your code. Only style-preserving write-back remains in
> Phase 1. Strict types, green test suite. The API is pre-1.0 and may still change. See
> [`docs/design/framework-design.md`](docs/design/framework-design.md) for the architecture,
> [`docs/design/roadmap.md`](docs/design/roadmap.md) for what's next, and [`docs/`](docs/) for the docs.

## Install

```bash
pip install tensorsketch-core            # the core (pydantic + typing only)
pip install "tensorsketch-core[anthropic,canvas,serve]"   # add a provider, the canvas, serving…
```

The core pulls in almost nothing; model SDKs, the canvas engine, database backends, and serving
are all opt-in [extras](docs/guides/installation.md).

## The five commitments

1. **Code is ground truth; the canvas is a projection.** Only wiring + typed interfaces
   round-trip; node bodies are opaque.
2. **One `Schema` abstraction** for tool I/O, structured output, typed state channels, and
   design-time port validation.
3. **A BSP/Pregel scheduler on an actor substrate, persisted by a durable journal** —
   cycles, deterministic parallel fan-out, resumable, single-process → distributed with the
   same code.
4. **The core knows interfaces, never implementations.** Every node type, tool, provider,
   memory backend, channel, and protocol is a plugin.
5. **Durability is one rule:** wrap side effects in a `durable step`; the framework journals
   the result and never re-runs it on resume.

## Quickstart (dev)

```bash
cd tensorsketch
uv sync                                # create the environment
uv run pytest                          # run the test suite
uv run python examples/support_router.py
uv run python examples/durable_resume.py
```

Common dev commands are wrapped in the `Makefile`: `make check` runs exactly what CI does
(lint · format check · strict types · tests); `make bench` runs the micro-benchmarks.

## Layout

```
tensorsketch/
├── src/tensorsketch/
│   ├── core/         # L1–L2: Schema, channels, nodes, graphs (the type + authoring spine)
│   └── runtime/      # L0: the BSP superstep engine
├── tests/
├── examples/
└── docs/             # user-facing documentation (Markdown; one file per doc "tab")
    ├── concepts/
    ├── guides/
    └── design/       # the architecture plan
```

Providers (OpenAI, Anthropic, …), interop protocols (MCP, A2A, AG-UI), and storage backends
will live as optional packages under `packages/*` so the core stays slim.
