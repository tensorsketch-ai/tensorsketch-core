# Durability

A long-running agent will crash, be redeployed, or hit a rate limit mid-flight. When it comes
back, two things must be true: it should **pick up where it left off**, and it must **not repeat
side effects** it already performed (don't charge the card twice, don't re-send the email). TensorSketch
gives you both, and the second is where most frameworks stop short.

Durability is **opt-in**: pass a `backend` and a `thread_id` to `invoke`. Without them, a graph
runs purely in memory.

```python
from tensorsketch import InMemoryBackend, SqliteBackend

backend = SqliteBackend("runs.db")        # or InMemoryBackend() for dev/tests
out = await app.invoke({"query": "..."}, thread_id="user-42", backend=backend)
```

## Two layers

TensorSketch's durability has two layers, both behind one `Backend` interface.

### 1. Checkpoints — "where are we"

At **every superstep barrier**, the runtime snapshots the channel values and the set of nodes
about to run next, and writes a `Checkpoint`. Because the barrier is the point where state is
consistent (see the [execution model](execution-model.md)), a checkpoint is a clean place to
stop and restart. Checkpoints form a **tree** (each has a `parent_id`), which is what makes
forking possible.

### 2. The effect journal — "don't do it twice"

Checkpointing alone isn't enough. If a crash happens *during* a superstep, resuming re-runs that
whole step — and naively that repeats every side effect in it. The **journal** fixes this: wrap
a side effect in `ctx.step(...)` and its result is recorded the moment it completes. On resume,
the recorded result is **returned from the journal instead of re-running the effect**.

```python
class Charge(Node):
    class In(Schema):  user: str
    class Out(Schema): charged: int

    async def run(self, ctx, inp):
        # Runs once, ever — even across crashes and resumes of this thread.
        amount = await ctx.step("charge_card", lambda: payment_api.charge(inp.user))
        return self.Out(charged=amount)
```

`ctx.step(name, fn)`:

- `fn` is a zero-argument callable returning an awaitable (e.g. an async API call). Wrap
  synchronous work with `asyncio.to_thread`.
- Each call is keyed by `(superstep, node, call-order)` so replays line up deterministically.
- Pass `idempotency_key="..."` to dedupe an effect across the *whole* run (e.g. a payment id),
  not just per step.
- **Results are stored as data.** There's no "your orchestration code must be deterministic"
  rule (the Temporal gotcha) — only the effects you explicitly wrap are memoized.

This is the line between *checkpointing* and *durable execution*. See it in action in
[`examples/durable_resume.py`](../../examples/durable_resume.py): a run charges a card, crashes,
resumes — and the payment API is called **exactly once**.

## Resuming

Call `invoke` again with the same `thread_id` and `backend`:

```python
# ... process crashes ...
out = await app.invoke(thread_id="user-42", backend=backend)   # continues from the last checkpoint
```

If a checkpoint exists for that thread, the run restores its state and continues. Journaled
effects are replayed; un-run nodes execute normally. You can pass new `input` on resume to inject
additional state before continuing; omit it to just continue.

## Inspecting and forking

```python
state   = app.get_state("user-42", backend)      # latest checkpointed state (or None)
history = app.get_history("user-42", backend)     # every checkpoint, oldest first

# Branch a new run from a past checkpoint with different input — a fresh journal, so effects
# run anew down the new branch. Great for "what if it had routed differently here".
forked = await app.fork(backend, "user-42", history[2].id, "user-42-alt", {"query": "..."})
```

## Backends — bring your own database

TensorSketch is **stateless**: the framework keeps no durable state of its own. Checkpoints, the effect
journal, and the event log all live in whatever `Backend` you pass. That's what keeps an agent
stateless and horizontally scalable — state is in *your* database, not in the process. Switching
stores is a one-line change:

| Backend | Install | Use |
|---|---|---|
| `InMemoryBackend()` | core | dev, tests (process memory) |
| `SqliteBackend(path)` | core | local / single-writer (a file, or `":memory:"`) |
| `PostgresBackend(dsn)` | `tensorsketch-core[postgres]` | multi-writer production (psycopg 3) |
| `RedisBackend(url)` | `tensorsketch-core[redis]` | fast, shared, distributed (redis-py) |

```python
from tensorsketch.runtime.backends import PostgresBackend, RedisBackend

backend = PostgresBackend("postgresql://user:pass@host/db")   # or:
backend = RedisBackend("redis://localhost:6379/0")
out = await app.invoke({"query": "..."}, thread_id="user-42", backend=backend)
```

The database drivers are **optional and imported lazily**, so `pip install tensorsketch-core` never pulls in
psycopg or redis. Each connector auto-creates its schema (three `thread_id`-keyed tables/keys) on
first use. You can also hand a connector an existing `connection=` / `client=` to share a pool.

### The serializer seam

Every backend turns values into bytes through a `Serializer`. The default, `PickleSerializer`,
round-trips any Python object (Pydantic models included) — but pickle executes code on load, so
only point a pickle-backed store at data you trust. Swap the codec to change that:

```python
from tensorsketch.runtime.backends import PostgresBackend

backend = PostgresBackend(dsn, serializer=MySignedSerializer())   # or a JSON/msgpack codec
```

A `Serializer` is just `dumps(obj) -> bytes` / `loads(bytes) -> Any`.

### Writing your own

Any store works — implement the `Backend` ABC (eight methods: save/latest/get/list checkpoint,
record/lookup effect, append/read events) and pass an instance. The `SqlBackend` base already
covers any DB-API 2.0 database; subclass it with your dialect's placeholder, blob type, and
upsert clause (that's all `PostgresBackend` and `SqliteBackend` are).

## Roadmap

Still ahead in this area: transaction-piggybacked exactly-once for DB-backed steps (commit the
effect result in the *same* transaction as the work), optional Temporal/Restate backends, and a
distributed runtime. See the [architecture plan](../design/framework-design.md).
