"""Conditional routing: classify a query, then route to a specialist.

Every edge is typed and the routing is declared, not buried in `if`/`else` inside a node — so
this graph is fully inspectable (and, in a later phase, drawable on a canvas). The classifier
here is rule-based to keep the example deterministic and dependency-free; in a real agent its
body would call an LLM. The *shape* of the graph would be identical.

Run:  uv run python examples/support_router.py
"""

from __future__ import annotations

import asyncio
from typing import Literal

from tensorsketch import END, START, Context, Graph, Node, Schema

Intent = Literal["billing", "tech", "other"]


class Support(Schema):
    """The graph's shared, typed state. Each field is a channel."""

    query: str
    intent: Intent = "other"
    answer: str = ""


class Classify(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        intent: Intent

    async def run(self, ctx: Context, inp: In) -> Out:
        q = inp.query.lower()
        if any(w in q for w in ("refund", "charge", "invoice", "billing")):
            return self.Out(intent="billing")
        if any(w in q for w in ("error", "crash", "bug", "broken")):
            return self.Out(intent="tech")
        return self.Out(intent="other")


class Billing(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(answer=f"[Billing] Looking into your billing question: {inp.query!r}")


class Tech(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(answer=f"[Tech] Let's debug that: {inp.query!r}")


class Fallback(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(answer="[General] Happy to help — could you tell me a bit more?")


def route(state: Support) -> str:
    """Pick the specialist for the classified intent (a declared conditional edge)."""
    return {"billing": "Billing", "tech": "Tech"}.get(state.intent, "Fallback")


async def main() -> None:
    app = (
        Graph(Support)
        .add(Classify)
        .add(Billing)
        .add(Tech)
        .add(Fallback)
        .edge(START, "Classify")
        .conditional("Classify", route)
        .edge("Billing", END)
        .edge("Tech", END)
        .edge("Fallback", END)
    ).compile()

    for query in (
        "I'd like a refund on my last invoice",
        "the app crashes every time I log in",
        "hello, what can you do?",
    ):
        out = await app.invoke({"query": query})
        print(f"{query!r:45} -> intent={out.intent:8} | {out.answer}")


if __name__ == "__main__":
    asyncio.run(main())
