"""Extracting a GraphIR from source — including incomplete, import-free code."""

from __future__ import annotations

import json

from tensorsketch.canvas import extract
from tensorsketch.canvas.ir import END, START

# Deliberately has NO imports and undefined names (Node, Schema, Graph, route, Hole, ...) and an
# unfilled hole — extraction is syntactic, so all of this is fine.
SOURCE = """
class Classify(Node):
    class In(Schema):
        query: str
    class Out(Schema):
        intent: Literal["billing", "tech"]
    async def run(self, ctx, inp):
        return self.Out(intent="billing")


class Billing(Node):
    class In(Schema):
        query: str
    class Out(Schema):
        answer: str
    async def run(self, ctx, inp):
        raise Hole("Answer billing questions using the KB tool")


class Tech(Node):
    class In(Schema):
        query: str
    class Out(Schema):
        answer: str
    async def run(self, ctx, inp):
        return self.Out(answer="...")


app = (
    Graph(Support)
    .add(Classify)
    .add(Billing)
    .add(Tech)
    .edge(START, "Classify")
    .conditional("Classify", route, {"billing": "Billing", "tech": "Tech"})
    .edge("Billing", END)
    .edge("Tech", END)
    .compile()
)
"""


def test_extracts_nodes_and_ports() -> None:
    ir = extract(SOURCE)
    assert {n.name for n in ir.nodes} == {"Classify", "Billing", "Tech"}

    classify = ir.node("Classify")
    assert classify is not None
    assert [(p.name, p.type) for p in classify.inputs] == [("query", "str")]
    assert classify.outputs[0].name == "intent"
    assert classify.outputs[0].type == 'Literal["billing", "tech"]'


def test_detects_holes() -> None:
    ir = extract(SOURCE)
    assert ir.node("Billing").has_hole is True  # type: ignore[union-attr]
    assert ir.node("Classify").has_hole is False  # type: ignore[union-attr]


def test_extracts_wiring() -> None:
    ir = extract(SOURCE)
    assert ir.state == "Support"
    assert ir.entry == "Classify"

    triples = {(e.source, e.target, e.kind) for e in ir.edges}
    assert (START, "Classify", "sequential") in triples
    assert ("Classify", "Billing", "conditional") in triples
    assert ("Classify", "Tech", "conditional") in triples
    assert ("Billing", END, "sequential") in triples
    assert ("Tech", END, "sequential") in triples

    # Conditional edges carry the routing function name.
    conditional = [e for e in ir.edges if e.kind == "conditional"]
    assert all(e.condition == "route" for e in conditional)


def test_ir_is_json_serializable() -> None:
    ir = extract(SOURCE)
    blob = json.dumps(ir.to_dict())
    restored = json.loads(blob)
    assert restored["state"] == "Support"
    assert restored["entry"] == "Classify"
    assert len(restored["nodes"]) == 3


def test_conditional_without_mapping_is_dynamic() -> None:
    source = """
class Tick(Node):
    class In(Schema):
        count: int
    class Out(Schema):
        count: int
    async def run(self, ctx, inp):
        return self.Out(count=inp.count + 1)

app = Graph(State).add(Tick).edge(START, "Tick").conditional("Tick", keep_going)
"""
    ir = extract(source)
    dynamic = [e for e in ir.edges if e.kind == "conditional"]
    assert len(dynamic) == 1
    assert dynamic[0].source == "Tick"
    assert dynamic[0].target is None  # targets not statically known
    assert dynamic[0].condition == "keep_going"


_NODES = SOURCE[: SOURCE.index("app = (")]  # just the three node classes


def test_statement_style_builder_extracts_same_wiring() -> None:
    source = _NODES + (
        "g = Graph(Support)\n"
        "g.add(Classify)\n"
        "g.add(Billing)\n"
        "g.add(Tech)\n"
        'g.edge(START, "Classify")\n'
        'g.conditional("Classify", route, {"billing": "Billing", "tech": "Tech"})\n'
        'g.edge("Billing", END)\n'
        'g.edge("Tech", END)\n'
    )
    ir = extract(source)
    assert ir.state == "Support"
    assert ir.entry == "Classify"
    assert ir.added == ["Classify", "Billing", "Tech"]
    triples = {(e.source, e.target, e.kind) for e in ir.edges}
    assert (START, "Classify", "sequential") in triples
    assert ("Classify", "Billing", "conditional") in triples
    assert ("Classify", "Tech", "conditional") in triples
    assert ("Billing", END, "sequential") in triples


def test_rshift_surface_extracts_same_wiring() -> None:
    source = _NODES + (
        "g = Graph(Support)\n"
        "classify, billing, tech = g.nodes(Classify, Billing, Tech)\n"
        "START >> classify\n"
        "classify >> Router(route, billing=billing, tech=tech)\n"
        "billing >> END\n"
        "tech >> END\n"
    )
    ir = extract(source)
    assert ir.state == "Support"
    assert ir.entry == "Classify"
    assert ir.added == ["Classify", "Billing", "Tech"]
    triples = {(e.source, e.target, e.kind) for e in ir.edges}
    assert (START, "Classify", "sequential") in triples
    assert ("Classify", "Billing", "conditional") in triples
    assert ("Classify", "Tech", "conditional") in triples
    assert ("Billing", END, "sequential") in triples
    assert ("Tech", END, "sequential") in triples


def test_rshift_fan_out_and_chaining() -> None:
    source = _NODES + (
        "g = Graph(Support)\n"
        "classify, billing, tech = g.nodes(Classify, Billing, Tech)\n"
        "START >> classify >> [billing, tech]\n"
    )
    ir = extract(source)
    assert ir.entry == "Classify"
    pairs = {(e.source, e.target) for e in ir.edges if e.kind == "sequential"}
    assert (START, "Classify") in pairs
    assert ("Classify", "Billing") in pairs
    assert ("Classify", "Tech") in pairs


def test_add_with_name_keyword_uses_alias() -> None:
    source = _NODES + (
        'g = Graph(Support)\ng.add(Classify, name="Router1")\ng.edge(START, "Router1")\n'
    )
    ir = extract(source)
    assert ir.added == ["Router1"]
    assert ir.entry == "Router1"


def test_no_graph_builder_still_extracts_nodes() -> None:
    source = """
class Lonely(Node):
    class In(Schema):
        x: int
    class Out(Schema):
        y: int
    async def run(self, ctx, inp):
        raise Hole("todo")
"""
    ir = extract(source)
    assert ir.state is None
    assert ir.entry is None
    assert [n.name for n in ir.nodes] == ["Lonely"]
    assert ir.nodes[0].has_hole is True
