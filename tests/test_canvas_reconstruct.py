"""Write-back (canvas → code): the round-trip invariant and byte-preservation of bodies."""

from __future__ import annotations

import pytest

from tensorsketch.canvas import extract, reconstruct
from tensorsketch.canvas.ir import END, EdgeIR, NodeIR, Port

SRC = """from tensorsketch import END, START, Graph, Node, Schema
from tensorsketch import Hole


# The classifier decides where to route.
class Classify(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        intent: str

    async def run(self, ctx, inp):
        # opaque body — must survive write-back verbatim
        return self.Out(intent="billing")


class Billing(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        answer: str

    async def run(self, ctx, inp):
        raise Hole("Answer billing using the KB tool")


app = (
    Graph(Support)
    .add(Classify)
    .add(Billing)
    .edge(START, "Classify")
    .conditional("Classify", route, {"billing": "Billing"})
    .edge("Billing", END)
    .compile()
)
"""


_DYNAMIC_CONDITIONAL = """
class Tick(Node):
    class In(Schema):
        count: int

    class Out(Schema):
        count: int

    async def run(self, ctx, inp):
        return self.Out(count=inp.count + 1)


app = Graph(State).add(Tick).edge(START, "Tick").conditional("Tick", keep_going)
"""

_FAN = """
class Seed(Node):
    class In(Schema):
        seed: int

    class Out(Schema):
        seed: int

    async def run(self, ctx, inp):
        return self.Out(seed=inp.seed)


app = (
    Graph(State)
    .add(Seed)
    .add(A)
    .add(B)
    .edge(START, "Seed")
    .edge("Seed", "A")
    .edge("Seed", "B")
    .edge("A", END)
    .edge("B", END)
)
"""


_ROUTER = """
class Classify(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        intent: str

    async def run(self, ctx, inp):
        return self.Out(intent="billing")


app = Graph(State).add(Classify).edge(START, "Classify").router("Classify", route, {"b": "Bill"})
"""

_LOOP = """
class Refine(Node):
    class In(Schema):
        score: float

    class Out(Schema):
        score: float

    async def run(self, ctx, inp):
        return self.Out(score=inp.score + 0.1)


app = Graph(State).add(Refine).edge(START, "Refine").loop("Refine", good_enough)
"""


@pytest.mark.parametrize("source", [SRC, _DYNAMIC_CONDITIONAL, _FAN, _ROUTER, _LOOP])
def test_round_trip_invariant(source: str) -> None:
    """extract(reconstruct(extract(code))) == extract(code) — the write-back safety gate."""
    ir = extract(source)
    assert extract(reconstruct(source, ir)) == ir


def test_reconstruct_preserves_bodies_imports_and_comments() -> None:
    out = reconstruct(SRC, extract(SRC))
    assert "from tensorsketch import END, START, Graph, Node, Schema" in out
    assert "from tensorsketch import Hole" in out
    assert "# The classifier decides where to route." in out
    assert "# opaque body — must survive write-back verbatim" in out
    assert 'return self.Out(intent="billing")' in out
    assert 'raise Hole("Answer billing using the KB tool")' in out
    assert ".compile()" in out  # the non-wiring tail is preserved


def test_conditional_mapping_round_trips() -> None:
    out = reconstruct(SRC, extract(SRC))
    assert '.conditional("Classify", route, {"billing": "Billing"})' in out
    edge = next(e for e in extract(out).edges if e.kind == "conditional")
    assert edge.key == "billing"
    assert edge.target == "Billing"


def test_router_extracts_as_a_conditional() -> None:
    # `router` is the intent-named alias of `conditional`, so it extracts identically.
    edges = extract(_ROUTER).edges
    cond = [e for e in edges if e.kind == "conditional"]
    assert {(e.source, e.key, e.target) for e in cond} == {("Classify", "b", "Bill")}
    assert "route" in reconstruct(_ROUTER, extract(_ROUTER))  # named path preserved


def test_loop_extracts_self_and_exit_branches() -> None:
    # `loop("Refine", good_enough)` becomes a two-branch conditional: back to itself, or out to END.
    edges = extract(_LOOP).edges
    cond = {(e.source, e.key, e.target) for e in edges if e.kind == "conditional"}
    assert cond == {("Refine", "loop", "Refine"), ("Refine", "exit", END)}
    out = reconstruct(_LOOP, extract(_LOOP))
    assert '"exit": END' in out  # the END target renders as the sentinel, not "__end__"


def test_add_edge_writes_back() -> None:
    ir = extract(SRC)
    ir.edges.append(EdgeIR(source="Billing", target="Classify", kind="sequential"))
    out = reconstruct(SRC, ir)

    triples = {(e.source, e.target, e.kind) for e in extract(out).edges}
    assert ("Billing", "Classify", "sequential") in triples
    # bodies still intact after the edit
    assert 'raise Hole("Answer billing using the KB tool")' in out


def test_remove_edge_writes_back() -> None:
    ir = extract(SRC)
    ir.edges = [e for e in ir.edges if not (e.source == "Billing" and e.target == END)]
    out = reconstruct(SRC, ir)

    triples = {(e.source, e.target) for e in extract(out).edges}
    assert ("Billing", END) not in triples


def test_entry_is_canonicalized_to_start_edge() -> None:
    source = """
class A(Node):
    class In(Schema):
        x: int

    class Out(Schema):
        x: int

    async def run(self, ctx, inp):
        return self.Out(x=inp.x)


app = Graph(State).add(A).entry("A").edge("A", END)
"""
    ir = extract(source)
    out = reconstruct(source, ir)
    assert '.edge(START, "A")' in out
    assert ".entry(" not in out
    assert extract(out) == ir  # canonicalization preserves the graph


def test_no_builder_returns_source_unchanged() -> None:
    source = "x = 1\n"
    assert reconstruct(source, extract(source)) == source


_CLASSES = SRC[: SRC.index("app = (")]

_STATEMENT_STYLE = _CLASSES + (
    "g = Graph(Support)\n"
    "g.add(Classify)\n"
    "g.add(Billing)\n"
    'g.edge(START, "Classify")\n'
    'g.conditional("Classify", route, {"billing": "Billing"})\n'
    'g.edge("Billing", END)\n'
    "app = g.compile()\n"
)

_RSHIFT_STYLE = _CLASSES + (
    "g = Graph(Support)\n"
    "classify, billing = g.nodes(Classify, Billing)\n"
    "START >> classify\n"
    "classify >> Router(route, billing=billing)\n"
    "billing >> END\n"
    "app = g.compile()\n"
)


@pytest.mark.parametrize("source", [_STATEMENT_STYLE, _RSHIFT_STYLE])
def test_alt_styles_round_trip(source: str) -> None:
    """Statement-style and `>>` sources satisfy the invariant and keep bodies intact."""
    ir = extract(source)
    out = reconstruct(source, ir)
    assert extract(out) == ir
    # node bodies survive verbatim
    assert 'raise Hole("Answer billing using the KB tool")' in out
    assert "# opaque body — must survive write-back verbatim" in out


@pytest.mark.parametrize("source", [_STATEMENT_STYLE, _RSHIFT_STYLE])
def test_alt_styles_canonicalize_to_fluent_chain(source: str) -> None:
    """Both styles fold into one canonical fluent chain assigned to the graph variable."""
    out = reconstruct(source, extract(source))
    assert "g = (\n    Graph(Support)" in out  # a single, cleanly-indented chain
    assert '.edge(START, "Classify")' in out
    assert '.conditional("Classify", route, {"billing": "Billing"})' in out
    # the folded statements are gone
    assert ">>" not in out
    assert "g.nodes(" not in out
    assert "g.add(" not in out
    # the separate compile() line is left alone
    assert "app = g.compile()" in out


def test_annotated_graph_var_round_trips() -> None:
    """`g: Graph[Support] = Graph(...)` binds the graph var too (an AnnAssign, not Assign)."""
    source = _CLASSES + (
        "g: Graph[Support] = Graph(Support)\n"
        "classify, billing = g.nodes(Classify, Billing)\n"
        "START >> classify >> billing\n"
        "billing >> END\n"
    )
    ir = extract(source)
    assert ir.entry == "Classify"
    assert ir.added == ["Classify", "Billing"]
    out = reconstruct(source, ir)
    assert extract(out) == ir
    assert ">>" not in out and "g.nodes(" not in out


# -- node creation (canvas palette) ---------------------------------------------------------

_NO_HOLE = """from tensorsketch import END, START, Graph, Node, Schema


class A(Node):
    class In(Schema):
        x: int

    class Out(Schema):
        x: int

    async def run(self, ctx, inp):
        return self.Out(x=inp.x)


app = Graph(State).add(A).edge(START, "A").edge("A", END)
"""


def _with_new_node(source: str, node: NodeIR, *, after: str) -> str:
    """Simulate a palette edit: create `node`, wire `after -> node`, and write it back."""
    ir = extract(source)
    ir.nodes.append(node)
    ir.added.append(node.name)
    ir.edges.append(EdgeIR(source=after, target=node.name, kind="sequential"))
    return reconstruct(source, ir)


def test_create_node_generates_a_stub() -> None:
    """A node the source never defined is synthesized as an idiomatic `class X(Node)` stub."""
    node = NodeIR(
        name="Escalate",
        inputs=[Port("query", "str")],
        outputs=[Port("ticket", "str")],
        has_hole=True,
    )
    out = _with_new_node(SRC, node, after="Billing")
    assert "class Escalate(Node):" in out
    assert "query: str" in out
    assert "ticket: str" in out
    assert 'raise Hole("Escalate needs code")' in out
    # the stub sits above the builder, so `.add(Escalate)` resolves
    assert out.index("class Escalate(Node):") < out.index(".add(Escalate)")


def test_created_node_round_trips() -> None:
    """The generated stub re-extracts to the exact NodeIR the canvas sent (has_hole and all)."""
    node = NodeIR(
        name="Escalate",
        inputs=[Port("query", "str")],
        outputs=[Port("ticket", "str")],
        has_hole=True,
    )
    out = _with_new_node(SRC, node, after="Billing")
    ir2 = extract(out)
    assert ir2.node("Escalate") == node
    assert "Escalate" in ir2.added
    assert EdgeIR("Billing", "Escalate", kind="sequential") in ir2.edges


def test_create_node_adds_hole_import_when_missing() -> None:
    """The stub raises `Hole`, so a `from tensorsketch import Hole` is added when it isn't already
    bound."""
    node = NodeIR(name="B", inputs=[Port("x", "int")], outputs=[Port("x", "int")], has_hole=True)
    out = _with_new_node(_NO_HOLE, node, after="A")
    assert "from tensorsketch import Hole" in out
    # the original import line is left intact, not rewritten
    assert "from tensorsketch import END, START, Graph, Node, Schema" in out


def test_create_node_does_not_duplicate_existing_hole_import() -> None:
    node = NodeIR(name="Escalate", has_hole=True)
    out = _with_new_node(SRC, node, after="Billing")  # SRC already imports Hole
    assert out.count("from tensorsketch import Hole") == 1


def test_create_portless_node_uses_pass() -> None:
    node = NodeIR(name="Blank", has_hole=True)
    out = _with_new_node(SRC, node, after="Billing")
    stub = out[out.index("class Blank(Node):") :]
    assert "class In(Schema):\n        pass" in stub
    assert "class Out(Schema):\n        pass" in stub
    assert extract(out).node("Blank") == node


def test_nested_builder_indents_cleanly() -> None:
    """A builder inside a function body indents its continuation lines to match the block."""
    source = (
        "def build():\n"
        "    g = Graph(Support)\n"
        "    g.add(Classify)\n"
        '    g.edge(START, "Classify")\n'
        '    g.edge("Classify", END)\n'
        "    return g.compile()\n"
    )
    out = reconstruct(source, extract(source))
    assert "    g = (\n        Graph(Support)" in out  # base indent 4, continuation 8
    assert extract(out) == extract(source)
