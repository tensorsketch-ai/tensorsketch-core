# Benchmarks

Indicative micro-benchmarks for the pure-Python reference runtime. They exist to **track
overhead and catch regressions**, not to advertise throughput — the node bodies are trivial, so
the numbers isolate scheduler + channel + checkpoint cost, not real agent work (which is
dominated by LLM/tool latency).

```bash
uv run python benchmarks/bench.py
```

## What it measures

- **Sequential chain (N supersteps)** — per-superstep scheduling overhead, with no backend, the
  in-memory backend (checkpoint per barrier), and the SQLite backend.
- **Parallel fan-out (width W)** — W nodes running in one superstep, merged through a reducer.

## Indicative results

On an Apple-silicon dev machine (Python 3.12), median of 25 runs:

| Shape | Result (indicative) |
|---|---|
| Chain per-superstep overhead | ~50–60 µs/superstep |
| Checkpointing overhead (in-memory) | ~+20–30% |
| Checkpointing overhead (SQLite `:memory:`) | ~+30% |
| Fan-out, 200 branches in one superstep | ~1.4 ms |

Numbers vary by machine; treat them as a relative baseline. When the Rust hot-path core lands
(a later phase), this same script is how we'll measure the speedup behind the unchanged API.
