"""Bring your own database: durable state lives *outside* the framework, in a store you choose.

TensorSketch is stateless — checkpoints and the effect journal live in whatever `Backend` you pass,
not
in the process. To prove it, this run persists to a SQLite *file*, then throws that backend away
and opens a **brand-new backend on the same file** (standing in for a fresh process / another
worker) to resume. The side effect still runs exactly once, because its result was journaled in
the database, not in memory.

Switching stores is a one-line change — see the commented alternatives in `main`.

Run:  uv run python examples/bring_your_own_database.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from tensorsketch import END, START, Context, Graph, Node, Schema, SqliteBackend

EFFECT_RUNS = 0


class State(Schema):
    user: str
    charged: int = 0


class Charge(Node):
    class In(Schema):
        user: str

    class Out(Schema):
        charged: int

    async def run(self, ctx: Context, inp: In) -> Out:
        async def call_payment_api() -> int:
            global EFFECT_RUNS
            EFFECT_RUNS += 1
            return 4200  # cents

        # Journaled in the database — replayed, not re-run, when another process resumes.
        amount = await ctx.step("charge_card", call_payment_api)
        return self.Out(charged=amount)


async def main() -> None:
    app = Graph(State).add(Charge).edge(START, "Charge").edge("Charge", END).compile()

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "runs.db")

        # --- "process A": run against a database file -----------------------------------
        backend_a = SqliteBackend(db)
        # One-line swaps to a networked store (state is identical, just elsewhere):
        #   from tensorsketch.runtime.backends import PostgresBackend, RedisBackend
        #   backend_a = PostgresBackend("postgresql://user:pass@host/db")
        #   backend_a = RedisBackend("redis://localhost:6379/0")
        result = await app.invoke({"user": "ada"}, thread_id="order-123", backend=backend_a)
        backend_a.close()
        print(f"process A: charged {result.charged} cents; payment API called {EFFECT_RUNS}x")

        # --- "process B": a *new* backend on the same file resumes the thread ------------
        backend_b = SqliteBackend(db)
        resumed = app.get_state("order-123", backend_b)
        assert resumed is not None  # state survived outside the process
        print(f"process B: read state straight from the database → charged {resumed.charged}")

        again = await app.invoke(thread_id="order-123", backend=backend_b)
        backend_b.close()
        print(
            f"process B: resumed to {again.charged} cents; "
            f"payment API still called {EFFECT_RUNS}x (exactly once ✓)"
        )


if __name__ == "__main__":
    asyncio.run(main())
