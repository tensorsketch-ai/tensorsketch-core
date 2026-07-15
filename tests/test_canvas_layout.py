"""The layout sidecar — manual node positions kept beside the code, never in it."""

from __future__ import annotations

import json
from pathlib import Path

from tensorsketch.canvas.server import _layout_path, load_graph, load_layout, save_layout

_GRAPH = """from tensorsketch import END, START, Context, Graph, Node, Schema


class A(Node):
    class In(Schema):
        x: int

    class Out(Schema):
        x: int

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(x=inp.x)


app = Graph(S).add(A).edge(START, "A").edge("A", END)
"""


def _write_graph(tmp_path: Path) -> Path:
    path = tmp_path / "graph.py"
    path.write_text(_GRAPH)
    return path


def test_no_sidecar_means_empty_layout(tmp_path: Path) -> None:
    assert load_layout(_write_graph(tmp_path)) == {"nodes": {}}


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    path = _write_graph(tmp_path)
    saved = save_layout(path, {"nodes": {"A": {"x": 12.5, "y": -40}}})
    assert saved == {"ok": True, "nodes": {"A": {"x": 12.5, "y": -40.0}}}
    assert load_layout(path) == {"nodes": {"A": {"x": 12.5, "y": -40.0}}}


def test_sidecar_lives_beside_the_source(tmp_path: Path) -> None:
    path = _write_graph(tmp_path)
    save_layout(path, {"nodes": {"A": {"x": 1, "y": 2}}})
    assert _layout_path(path) == tmp_path / "graph.py.layout.json"
    assert _layout_path(path).is_file()


def test_malformed_positions_are_dropped(tmp_path: Path) -> None:
    path = _write_graph(tmp_path)
    saved = save_layout(
        path,
        {
            "nodes": {
                "A": {"x": 1, "y": 2},  # kept
                "B": {"x": "nope", "y": 2},  # non-numeric x → dropped
                "C": {"x": True, "y": 2},  # bool is not a coordinate → dropped
                "D": {"x": 1},  # missing y → dropped
                "E": "garbage",  # not a dict → dropped
            }
        },
    )
    assert saved["nodes"] == {"A": {"x": 1.0, "y": 2.0}}


def test_corrupt_sidecar_yields_empty_not_error(tmp_path: Path) -> None:
    path = _write_graph(tmp_path)
    _layout_path(path).write_text("{ this is not json")
    assert load_layout(path) == {"nodes": {}}


def test_load_graph_includes_the_layout(tmp_path: Path) -> None:
    path = _write_graph(tmp_path)
    save_layout(path, {"nodes": {"A": {"x": 5, "y": 6}}})
    payload = load_graph(path)
    assert payload["layout"] == {"nodes": {"A": {"x": 5.0, "y": 6.0}}}
    assert payload["ir"]["added"] == ["A"]  # graph still extracted normally


def test_sidecar_is_written_pretty_and_versioned(tmp_path: Path) -> None:
    path = _write_graph(tmp_path)
    save_layout(path, {"nodes": {"A": {"x": 1, "y": 2}}})
    data = json.loads(_layout_path(path).read_text())
    assert data["version"] == 1
    assert data["nodes"] == {"A": {"x": 1.0, "y": 2.0}}
