# Installation

> TensorSketch is pre-1.0 and not yet published to PyPI; install from the source tree. The public API
> may still change between minor versions.

## Requirements

- **Python 3.11+** (the runtime uses `asyncio.TaskGroup` and modern typing).
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.

## Set up the environment

```bash
cd tensorsketch
uv sync
```

This creates a virtual environment in `.venv/` and installs TensorSketch (editable) plus the dev
tools. The core has only two runtime dependencies — **pydantic** and **typing-extensions** —
by design: providers, protocols, and storage backends are optional packages, so installing
TensorSketch never pulls in an LLM SDK or a database driver.

### Optional extras

Everything third-party is opt-in — install only what you use. `import tensorsketch` never pulls in an LLM
SDK, a database driver, or a web framework unless you ask for it.

| Extra | Installs | Enables |
|---|---|---|
| `anthropic` · `openai` · `google` | the provider SDK | that model provider (`import tensorsketch` stays SDK-free) |
| `postgres` · `redis` | psycopg 3 · redis-py | a bring-your-own-database `Backend` |
| `canvas` | libcst | code⇄canvas extraction/write-back and **Studio** |
| `mcp` | the MCP SDK | consume/expose tools over Model Context Protocol |
| `otel` | opentelemetry-sdk | export traces to OpenTelemetry |
| `serve` | starlette + httpx | serve an agent over OpenAI / A2A / AG-UI |

```bash
uv sync --extra anthropic --extra canvas --extra serve
# or, as a dependency:  pip install "tensorsketch-core[anthropic,canvas,serve]"
```

## Verify

```bash
uv run pytest            # the test suite
uv run python examples/support_router.py
uv run python examples/counting_loop.py
```

## Development commands

```bash
uv run ruff check src tests examples benchmarks   # lint
uv run ruff format src tests examples benchmarks   # format
uv run mypy                                        # strict type-check (src + tests)
uv run pytest                                      # tests
uv run python benchmarks/bench.py                  # micro-benchmarks
```

Or use the `Makefile`:

```bash
make check    # lint · format check · strict types · tests (exactly what CI runs)
make bench    # micro-benchmarks
```

The project is configured for **strict** mypy and a broad ruff ruleset — TensorSketch aims for
zero-compromise, fully-typed code. CI (GitHub Actions) runs `make check` on Python 3.11 and 3.12.
