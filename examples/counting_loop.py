"""A guarded loop (a real cycle in the graph): tick a counter until it hits a limit.

Shows two runtime features at once:

* **Cycles** — the conditional edge points back to the same node, so the graph loops. The BSP
  scheduler handles this natively; the recursion limit is the safety net if the guard is wrong.
* **Reducers** — `log` accumulates across supersteps via `Reducer(add)`, so each tick appends
  rather than overwrites.

Run:  uv run python examples/counting_loop.py
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated

from tensorsketch import END, START, Context, Graph, Node, Reducer, Schema


class Counter(Schema):
    count: int = 0
    limit: int = 5
    log: Annotated[list[str], Reducer(add)] = []


class Tick(Node):
    class In(Schema):
        count: int

    class Out(Schema):
        count: int
        log: list[str]

    async def run(self, ctx: Context, inp: In) -> Out:
        nxt = inp.count + 1
        return self.Out(count=nxt, log=[f"superstep {ctx.superstep}: {inp.count} -> {nxt}"])


def keep_going(state: Counter) -> str:
    """Loop back to Tick until the limit is reached, then stop."""
    return END if state.count >= state.limit else "Tick"


async def main() -> None:
    app = (Graph(Counter).add(Tick).edge(START, "Tick").conditional("Tick", keep_going)).compile()

    out = await app.invoke({"limit": 5})
    print(f"final count: {out.count}")
    for line in out.log:
        print(f"  {line}")


if __name__ == "__main__":
    asyncio.run(main())
