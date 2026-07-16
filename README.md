<div align="center">

# TensorSketch

### _Create agents out of thin air._

**A code-first, visually-editable, durable agentic framework.**

[![CI](https://github.com/tensorsketch-ai/tensorsketch-core/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorsketch-ai/tensorsketch-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org)
[![Types](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org)

[**Quickstart**](#60-seconds-a-durable-tool-using-agent)&nbsp;·&nbsp;[**Docs**](docs/)&nbsp;·&nbsp;[**Examples**](examples/)&nbsp;·&nbsp;[**Roadmap**](docs/design/roadmap.md)&nbsp;·&nbsp;[**Architecture**](docs/design/framework-design.md)

</div>

---

Sketch a node onto the canvas and it's born in your code as a typed [hole](docs/guides/getting-started.md);
fill the hole with logic and it lights up on the canvas. **Code and canvas are the same thing from
two angles** — that's the whole idea. TensorSketch is an agentic framework built on four ideas:

- **Code is the single source of truth.** Your agents are plain, typed Python.
- **A visual canvas is a lossless projection of that code** — edit either side; they stay in sync.
- **Execution is durable by default** — a BSP runtime with checkpoints, crash-resume, and
  exactly-once side effects.
- **Every capability is a plugin** — providers, tools, memory, storage, protocols.

So it's easy to start, hard to outgrow, and it absorbs whatever agent research comes next —
without locking you into a vendor, a database, or a UI.

> **Pre-1.0.** The runtime, the full agent layer, interop & observability, and the code⇄canvas
> engine are all in place and green — the API may still change. See the [roadmap](docs/design/roadmap.md).

## Install

```bash
pip install tensorsketch-core                              # the core: pydantic + typing, nothing else
pip install "tensorsketch-core[anthropic,canvas,serve]"    # add a provider, the canvas, serving…
```

The core pulls in almost nothing. Model SDKs, the canvas engine, database backends, interop
protocols, and serving are all opt-in [extras](docs/guides/installation.md) — `pip install
tensorsketch-core` never drags in an LLM SDK or a database driver.

## 60 seconds: a durable, tool-using agent

```python
from tensorsketch import create_agent, tool
from tensorsketch.providers.anthropic import AnthropicProvider


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = create_agent(
    AnthropicProvider(model="claude-sonnet-4-6"),   # needs ANTHROPIC_API_KEY
    tools=[multiply],
    system="You are a helpful calculator. Use tools for arithmetic.",
)

result = await agent.invoke({"query": "what is 6 * 7?"})
print(result.output)   # -> "6 times 7 is 42."
```

`@tool` derives the JSON schema from the signature. `create_agent` returns an ordinary graph, so it
composes like any other. And every model and tool call in that loop is wrapped in a **durable
step** — crash halfway and `resume` picks up exactly where it left off, *without* re-calling the
model or re-running the tool.

No API key handy? [`examples/calculator_agent.py`](examples/calculator_agent.py) runs the same agent
offline with a scripted `FakeProvider`.

## The same graph, drawn — and edited — on a canvas

Because the graph *is* the code, it renders as a diagram and edits round-trip losslessly. This is a
plain graph…

```python
app = (
    Graph(Support)
    .add(Classify)
    .add(Billing)
    .add(Tech)
    .edge(START, "Classify")
    .conditional("Classify", route)     # a declared route, not an if/else buried in a node
    .edge("Billing", END)
    .edge("Tech", END)
).compile()
```

…which the Studio draws as this — the same graph, no second source of truth:

```text
                         START
                           │
                       ┌───▼────┐
                       │Classify│
                       └───┬────┘
              route(state) │  conditional
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
            ┌────────┐ ┌──────┐ ┌────────┐
            │Billing │ │ Tech │ │Fallback│
            └────┬───┘ └───┬──┘ └───┬────┘
                 └─────────┼─────────┘
                           ▼
                          END
```

Open it with:

```bash
python -m tensorsketch.canvas examples/support_router.py
```

Drag to wire nodes, create them from a palette, rearrange the layout — every gesture writes
straight back into your file, **preserving your authoring style** (a fluent chain stays a chain,
`>>` stays `>>`), touching nothing but the wiring. Node bodies, imports, and comments are
byte-preserved. Click **▶ live** to watch a run light up the graph with per-node latency, cost, and
status. See [the Studio guide](docs/guides/studio.md) and [Code & Canvas](docs/concepts/code-and-canvas.md).

## Why TensorSketch — the five commitments

1. **Code is ground truth; the canvas is a projection.** Only wiring and typed interfaces
   round-trip; node bodies are opaque and never rewritten.
2. **One `Schema` abstraction** for tool I/O, structured output, typed state channels, and
   design-time port validation.
3. **A BSP/Pregel scheduler on an actor substrate, persisted by a durable journal** — native
   cycles, deterministic parallel fan-out, resumable; single-process today, distributed later with
   the same agent code.
4. **The core knows interfaces, never implementations.** Every node type, tool, provider, memory
   backend, channel, and protocol is a plugin.
5. **Durability is one rule:** wrap a side effect in a `durable step`; the framework journals the
   result and never re-runs it on resume.

## What's in the box

| Area | Highlights |
| --- | --- |
| **Runtime** | BSP superstep engine · typed state channels + reducers · native cycles · durable checkpoints, `resume`/`fork` · exactly-once effects (`ctx.step`) · live `stream()` + resumable `replay` · `InMemory`/`Sqlite` backends |
| **Agents** | `@tool` (schema from the signature) · `Llm` node · durable `Agent` loop · `create_agent` · structured output (validate-and-repair) · `gather_map`/`parallel`/`run_subgraph` · graph-level `Send` fan-out · **multi-agent coordination** (`as_tool`) |
| **Providers** | `ChatProvider` (zero-SDK interface) · **Anthropic · OpenAI** (+ OpenAI-compatible) **· Google** — lazy imports · documented custom-provider path |
| **Code ⇄ Canvas** | CST extraction · `>>` wiring surface · **style-preserving write-back** with the round-trip invariant as a CI gate · node-stub generation · project-wide hole surfacing · **TensorSketch Studio** (live trace overlay, layout sidecar) |
| **Interop** | MCP (consume + expose) · serve one agent over **OpenAI / A2A / AG-UI** · `a2a_tool` to consume a remote agent |
| **Observability** | vendor-neutral tracing (tokens/cost/status) · `File`/`OTel`/`Multi` tracers · middleware (retry, observability) · a name registry for providers/backends |
| **Evaluation** | trajectory-aware graders + `LlmJudge` · pass@k / pass^k · CI gate (`report.require`) · emittable results · online scoring + **drift detection** |
| **Storage** | bring-your-own-database: `Postgres` and `Redis` backends behind one `Backend` ABC — the framework stays **stateless** |

## Docs

Full documentation lives in [`docs/`](docs/) (one Markdown file per concept):

- **Start here:** [Getting started](docs/guides/getting-started.md) · [Installation & extras](docs/guides/installation.md)
- **Concepts:** [Nodes & graphs](docs/concepts/nodes-and-graphs.md) · [Durability](docs/concepts/durability.md) · [Agents](docs/concepts/agents.md) · [Tools](docs/concepts/tools.md) · [Coordination](docs/concepts/coordination.md) · [Code & Canvas](docs/concepts/code-and-canvas.md) · [Evaluation](docs/concepts/evaluation.md)
- **The canvas:** [TensorSketch Studio](docs/guides/studio.md)
- **Design:** [Architecture plan](docs/design/framework-design.md) · [Roadmap](docs/design/roadmap.md) · [Build status](docs/design/status.md) · [Decisions](docs/design/decisions.md)

## Develop

```bash
git clone https://github.com/tensorsketch-ai/tensorsketch-core.git
cd tensorsketch-core
uv sync                                    # create the environment
uv run pytest                              # run the suite
uv run python examples/research_desk.py    # ⭐ a multi-agent research desk: graph + sub-agents,
                                           #    a revision loop, one trace, and an eval — all offline
uv run python examples/calculator_agent.py # a single tool-using agent
uv run python examples/multi_agent.py      # a supervisor delegating to specialists
```

[`examples/`](examples/) is runnable and offline by default (a scripted `FakeProvider` stands in
for a model, a one-line swap from a real provider). `research_desk.py` is the tour: an explicit
orchestration graph whose stages delegate to real tool-using sub-agents, unified under one trace and
graded by the eval harness — and, because it's a plain graph, `python -m tensorsketch.canvas
examples/research_desk.py` opens it on the canvas.

`make check` runs exactly what CI runs — lint · format check · strict types · tests; `make bench`
runs the micro-benchmarks. Every change lands green.

## Layout

```
tensorsketch-core/
├── src/tensorsketch/
│   ├── core/            # Schema, channels, nodes, the Graph builder, the `>>` wiring surface
│   ├── runtime/         # the BSP superstep engine + durable backends
│   ├── agents/          # tools loop, Llm node, create_agent, coordination
│   ├── providers/       # ChatProvider + Anthropic / OpenAI / Google (lazy)
│   ├── canvas/          # extract ⇄ reconstruct, the IR, and TensorSketch Studio
│   ├── interop/         # MCP: consume external tool servers, expose your own
│   ├── serve/           # serve one agent over OpenAI / A2A / AG-UI
│   ├── observability/   # tracing, exporters, the OTel bridge
│   └── eval/            # the trajectory-aware evaluation harness
├── examples/            # runnable, offline-by-default
├── tests/
└── docs/                # user-facing docs (Markdown, one file per concept)
```

Providers, protocols, and storage backends are optional **extras** on this one package — the core
stays slim, and nothing heavy is imported until you ask for it.

## License

[Apache-2.0](LICENSE).
