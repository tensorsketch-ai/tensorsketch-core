"""The `>>` wiring surface — sugar that must compile to the same graph as the fluent builder."""

from __future__ import annotations

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
        return self.Out(intent="billing" if "refund" in inp.query else "tech")


class Billing(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(answer="billing")


class Tech(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        answer: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(answer="tech")


def route(state: Support) -> str:
    return {"billing": "Billing", "tech": "Tech"}.get(state.intent, "Tech")


def _fluent() -> Graph[Support]:
    return (
        Graph(Support)
        .add(Classify)
        .add(Billing)
        .add(Tech)
        .edge(START, "Classify")
        .conditional("Classify", route, {"billing": "Billing", "tech": "Tech"})
        .edge("Billing", END)
        .edge("Tech", END)
    )


def _rshift() -> Graph[Support]:
    g: Graph[Support] = Graph(Support)
    classify, billing, tech = g.nodes(Classify, Billing, Tech)
    START >> classify
    classify >> Router(route, billing=billing, tech=tech)
    billing >> END
    tech >> END
    return g


async def test_rshift_matches_fluent() -> None:
    """The two authoring styles produce graphs that behave identically."""
    for query in ("please refund me", "the app crashed"):
        fluent = await _fluent().compile().invoke({"query": query})
        sugar = await _rshift().compile().invoke({"query": query})
        assert fluent.answer == sugar.answer


async def test_rshift_sets_entry_and_wiring() -> None:
    g = _rshift()
    assert g._entry == "Classify"
    assert g._branches["Classify"].mapping == {"billing": "Billing", "tech": "Tech"}
    assert "Billing" in g._nodes and "Tech" in g._nodes


async def test_rshift_chaining_returns_rhs() -> None:
    g: Graph[Support] = Graph(Support)
    classify, billing = g.nodes(Classify, Billing)
    # `START >> classify >> billing` should read left-to-right and wire both hops.
    START >> classify >> billing
    assert g._entry == "Classify"
    assert g._edges["Classify"] == {"Billing"}


async def test_fan_out_list() -> None:
    class Seed(Node):
        class In(Schema):
            query: str

        class Out(Schema):
            query: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(query=inp.query)

    g: Graph[Support] = Graph(Support)
    seed, billing, tech = g.nodes(Seed, Billing, Tech)
    START >> seed >> [billing, tech]
    assert g._edges["Seed"] == {"Billing", "Tech"}


async def test_getitem_handle() -> None:
    g: Graph[Support] = Graph(Support).add(Classify).add(Billing)
    START >> g["Classify"] >> g["Billing"]
    assert g._entry == "Classify"
    assert g._edges["Classify"] == {"Billing"}


async def test_dynamic_router_no_mapping() -> None:
    g: Graph[Support] = Graph(Support)
    classify, billing, tech = g.nodes(Classify, Billing, Tech)
    START >> classify >> Router(route)
    billing >> END
    tech >> END
    assert g._branches["Classify"].mapping is None
    out = await g.compile().invoke({"query": "please refund me"})
    assert out.answer == "billing"
