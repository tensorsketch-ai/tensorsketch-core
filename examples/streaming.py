"""Streaming: watch a run unfold as a live event stream.

Instead of `await app.invoke(...)` (which returns only the final state), `app.stream(...)` yields
an `Event` for each thing that happens — nodes starting and finishing, the merged state after
each superstep, and any custom events a node emits via `ctx.emit`. Every event is namespaced by
node and carries a monotonic `seq` cursor.

Run:  uv run python examples/streaming.py
"""

from __future__ import annotations

import asyncio

from tensorsketch import END, START, Context, Graph, Node, Schema


class Pipeline(Schema):
    text: str
    cleaned: str = ""
    result: str = ""


class Clean(Node):
    class In(Schema):
        text: str

    class Out(Schema):
        cleaned: str

    async def run(self, ctx: Context, inp: In) -> Out:
        await ctx.emit("status", {"stage": "cleaning"})
        return self.Out(cleaned=inp.text.strip().lower())


class Analyze(Node):
    class In(Schema):
        cleaned: str

    class Out(Schema):
        result: str

    async def run(self, ctx: Context, inp: In) -> Out:
        await ctx.emit("status", {"stage": "analyzing"})
        verdict = "question" if inp.cleaned.endswith("?") else "statement"
        return self.Out(result=verdict)


async def main() -> None:
    app = (
        Graph(Pipeline)
        .add(Clean)
        .add(Analyze)
        .edge(START, "Clean")
        .edge("Clean", "Analyze")
        .edge("Analyze", END)
    ).compile()

    async for event in app.stream({"text": "  Is this a Question?  "}):
        node = event.node or "-"
        print(f"seq={event.seq:>2}  {event.type:<11} [{node:<8}]  {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
