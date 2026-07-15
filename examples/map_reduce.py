"""Map/reduce with gather_map: process a collection concurrently and durably.

Counts the words in each document in parallel, then reduces to a total — all inside one node,
with each item journaled (so a crash mid-batch would resume without re-processing finished
items). No model needed; the per-item work is a plain function.

Run:  uv run python examples/map_reduce.py
"""

from __future__ import annotations

import asyncio

from tensorsketch import END, START, Context, Graph, Node, Schema, gather_map


class State(Schema):
    docs: list[str]
    counts: list[int] = []
    total: int = 0


class WordCount(Node):
    class In(Schema):
        docs: list[str]

    class Out(Schema):
        counts: list[int]
        total: int

    async def run(self, ctx: Context, inp: In) -> Out:
        async def count(doc: str) -> int:
            return len(doc.split())

        counts = await gather_map(ctx, inp.docs, count, max_concurrency=4)
        return self.Out(counts=counts, total=sum(counts))


async def main() -> None:
    app = Graph(State).add(WordCount).edge(START, "WordCount").edge("WordCount", END).compile()
    docs = ["the quick brown fox", "jumps over", "the lazy dog once again please"]
    out = await app.invoke({"docs": docs})

    for doc, count in zip(docs, out.counts, strict=True):
        print(f"{count:>2}  {doc!r}")
    print(f"total words: {out.total}")


if __name__ == "__main__":
    asyncio.run(main())
