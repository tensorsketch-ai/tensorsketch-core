"""Dynamic fan-out — a graph-level map/reduce with `Send`.

A router fans a list out to **one worker instance per item** (each its own superstep task with its
own payload). The workers merge into a reducer channel that a collector reads. Then a `loop` keeps
refining a value until a threshold is met. No provider needed — it runs offline.

Run:  uv run python examples/dynamic_fanout.py
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated

from tensorsketch import END, START, Context, Graph, Node, Reducer, Schema, Send


class State(Schema):
    numbers: list[int] = []
    n: int = 0  # per-worker input slot — each Send overrides it for its own instance
    squares: Annotated[list[int], Reducer(add)] = []  # workers merge their results here
    total: int = 0


class Split(Node):
    """The fan-out source: it just holds the list; its router decides the Sends."""

    class In(Schema):
        numbers: list[int]

    class Out(Schema):
        numbers: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(numbers=inp.numbers)


class Square(Node):
    """One worker instance per number — reads its own `n` from the Send payload."""

    class In(Schema):
        n: int

    class Out(Schema):
        squares: list[int]

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(squares=[inp.n * inp.n])


class Total(Node):
    """The reduce step: reads the merged reducer channel once every worker has run."""

    class In(Schema):
        squares: list[int]

    class Out(Schema):
        total: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(total=sum(inp.squares))


async def main() -> None:
    app = (
        Graph(State)
        .add(Split)
        .add(Square)
        .add(Total)
        .edge(START, "Split")
        # one Square instance per number, each with its own payload → all merge at the barrier
        .router("Split", lambda s: [Send("Square", {"n": x}) for x in s.numbers])
        .edge("Square", "Total")
        .edge("Total", END)
    ).compile()

    out = await app.invoke({"numbers": [1, 2, 3, 4, 5]})
    print("squares:", sorted(out.squares))  # [1, 4, 9, 16, 25]
    print("total:  ", out.total)  # 55


if __name__ == "__main__":
    asyncio.run(main())
