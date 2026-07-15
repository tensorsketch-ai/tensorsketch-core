"""Durable execution: a run "crashes" mid-way, then resumes without repeating its side effect.

This is the difference between *checkpointing* and *durable execution*. A naive checkpoint would
re-run the whole failed step on resume — calling the (expensive, non-idempotent) effect twice.
TensorSketch journals each `ctx.step(...)` effect, so on resume the recorded result is replayed and
the
effect runs **exactly once**.

Run:  uv run python examples/durable_resume.py
"""

from __future__ import annotations

import asyncio

from tensorsketch import END, START, Context, Graph, InMemoryBackend, Node, Schema

# Tracks how many times the "expensive" effect actually executes, and whether we've crashed yet.
EFFECT_RUNS = 0
HAS_CRASHED = False


class State(Schema):
    user: str
    charged: int = 0


class Charge(Node):
    class In(Schema):
        user: str

    class Out(Schema):
        charged: int

    async def run(self, ctx: Context, inp: In) -> Out:
        global EFFECT_RUNS, HAS_CRASHED

        async def call_payment_api() -> int:
            # Pretend this hits a real, non-idempotent payment API.
            global EFFECT_RUNS
            EFFECT_RUNS += 1
            return 4200  # cents

        amount = await ctx.step("charge_card", call_payment_api)

        # Simulate a crash *after* the charge succeeded but *before* the step committed.
        if not HAS_CRASHED:
            HAS_CRASHED = True
            raise RuntimeError("process crashed after charging, before finishing")

        return self.Out(charged=amount)


async def main() -> None:
    backend = InMemoryBackend()
    app = Graph(State).add(Charge).edge(START, "Charge").edge("Charge", END).compile()

    # First attempt: the card is charged, then the process crashes.
    try:
        await app.invoke({"user": "ada"}, thread_id="order-123", backend=backend)
    except RuntimeError as exc:
        print(f"crashed: {exc}")
    print(f"payment API called {EFFECT_RUNS} time(s) so far")

    # Resume the same thread. The charge is replayed from the journal — not re-executed.
    result = await app.invoke(thread_id="order-123", backend=backend)
    print(f"resumed and finished: charged {result.charged} cents to {result.user}")
    print(f"payment API called {EFFECT_RUNS} time(s) total  (exactly once ✓)")


if __name__ == "__main__":
    asyncio.run(main())
