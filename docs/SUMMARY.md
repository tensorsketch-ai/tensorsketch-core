# Documentation

This is the navigation index for TensorSketch's documentation. Each entry is a standalone Markdown
file (one "tab"). Run `uv run python docs/build_docs.py` to render this whole set into a
single, self-contained HTML site at `docs/site/index.html` — sidebar, search, offline syntax
highlighting, light/dark — with no dependencies and no build step.

- [Home](index.md)
- **Guides**
  - [Installation](guides/installation.md)
  - [Getting started](guides/getting-started.md)
  - [Studio — the visual canvas](guides/studio.md)
- **Concepts**
  - [State & channels](concepts/state-and-channels.md)
  - [Nodes & graphs](concepts/nodes-and-graphs.md)
  - [Execution model](concepts/execution-model.md)
  - [Durability](concepts/durability.md)
  - [Streaming](concepts/streaming.md)
  - [Tools](concepts/tools.md)
  - [Providers](concepts/providers.md)
  - [Choosing by name (registry)](concepts/registry.md)
  - [Agents](concepts/agents.md)
  - [Multi-agent coordination](concepts/coordination.md)
  - [Composition patterns](concepts/patterns.md)
  - [Middleware](concepts/middleware.md)
  - [Tracing & observability](concepts/tracing.md)
  - [Evaluation](concepts/evaluation.md)
  - [MCP interop](concepts/mcp.md)
  - [Serving (OpenAI, A2A, AG-UI)](concepts/serving.md)
  - [Code ⇄ canvas](concepts/code-and-canvas.md)
- **Design**
  - [Architecture plan](design/framework-design.md)
  - [Roadmap](design/roadmap.md)
  - [Build status & backlog](design/status.md)
  - [Decisions log](design/decisions.md)
