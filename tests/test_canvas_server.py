"""The Studio bridge — load extracts, save reconstructs, and the file stays the source of truth."""

from __future__ import annotations

from pathlib import Path

from tensorsketch.canvas.ir import END
from tensorsketch.canvas.server import apply_edits, load_graph

SOURCE = """from tensorsketch import END, START, Graph, Node, Schema
from tensorsketch import Hole


class Classify(Node):
    class In(Schema):
        query: str

    class Out(Schema):
        intent: str

    async def run(self, ctx, inp):
        # opaque body — must survive save verbatim
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


def _write(tmp_path: Path) -> Path:
    file = tmp_path / "graph.py"
    file.write_text(SOURCE)
    return file


def test_load_extracts_graph(tmp_path: Path) -> None:
    result = load_graph(_write(tmp_path))
    assert result["path"].endswith("graph.py")
    ir = result["ir"]
    assert ir["state"] == "Support"
    assert ir["entry"] == "Classify"
    assert {n["name"] for n in ir["nodes"]} == {"Classify", "Billing"}
    assert next(n for n in ir["nodes"] if n["name"] == "Billing")["has_hole"] is True


def test_save_writes_edit_back_and_preserves_bodies(tmp_path: Path) -> None:
    file = _write(tmp_path)
    ir = load_graph(file)["ir"]
    ir["edges"].append(
        {
            "source": "Billing",
            "target": "Classify",
            "kind": "sequential",
            "condition": None,
            "key": None,
        }
    )

    result = apply_edits(file, ir)

    # the edit is now in the code, and the re-extracted IR reflects it
    triples = {(e["source"], e["target"], e["kind"]) for e in result["ir"]["edges"]}
    assert ("Billing", "Classify", "sequential") in triples
    on_disk = file.read_text()
    assert '.edge("Billing", "Classify")' in on_disk
    # bodies + comments untouched
    assert 'raise Hole("Answer billing using the KB tool")' in on_disk
    assert "# opaque body — must survive save verbatim" in on_disk


def test_save_can_remove_an_edge(tmp_path: Path) -> None:
    file = _write(tmp_path)
    ir = load_graph(file)["ir"]
    ir["edges"] = [e for e in ir["edges"] if not (e["source"] == "Billing" and e["target"] == END)]

    result = apply_edits(file, ir)

    pairs = {(e["source"], e["target"]) for e in result["ir"]["edges"]}
    assert ("Billing", END) not in pairs
