"""The `>>` wiring surface — the same support router, authored as a diagram.

`Graph.nodes(...)` hands back a handle per node; `>>` then reads like the graph looks. This is
pure sugar over `.add`/`.edge`/`.conditional` — the compiled graph is identical to the fluent
form in `support_router.py`, and it round-trips through the code⇄canvas engine just the same.

Run:  uv run python examples/wiring_operators.py
"""

from __future__ import annotations

import asyncio
from typing import Literal

from tensorsketch import END, START, Context, Graph, Node, Router, Schema

Intent = Literal["billing", "tech", "other"]


class Support(Schema):
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
        return self.Out(answer=f"[Billing] Looking into: {inp.query!r}")


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
        return self.Out(answer="[General] Happy to help — tell me a bit more?")


def route(state: Support) -> str:
    return {"billing": "Billing", "tech": "Tech"}.get(state.intent, "Fallback")


async def main() -> None:
    g: Graph[Support] = Graph(Support)
    classify, billing, tech, fallback = g.nodes(Classify, Billing, Tech, Fallback)

    START >> classify
    classify >> Router(route, billing=billing, tech=tech, other=fallback)
    billing >> END
    tech >> END
    fallback >> END

    app = g.compile()
    for query in (
        "I'd like a refund on my last invoice",
        "the app crashes every time I log in",
        "hello, what can you do?",
    ):
        out = await app.invoke({"query": query})
        print(f"{query!r:45} -> intent={out.intent:8} | {out.answer}")


if __name__ == "__main__":
    asyncio.run(main())
