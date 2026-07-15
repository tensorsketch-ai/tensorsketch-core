"""Project-wide hole surfacing — "N nodes still need code" across a codebase."""

from __future__ import annotations

from pathlib import Path

from tensorsketch.canvas import HoleRef, find_holes
from tensorsketch.canvas.server import load_holes

_WITH_HOLES = """from tensorsketch import Context, Hole, Node, Schema


class Draft(Node):
    class In(Schema):
        topic: str

    class Out(Schema):
        draft: str

    async def run(self, ctx: Context, inp: In) -> Out:
        raise Hole("Write a first draft from the topic")


class Ready(Node):
    class In(Schema):
        draft: str

    class Out(Schema):
        draft: str

    async def run(self, ctx: Context, inp: In) -> Out:
        return self.Out(draft=inp.draft)
"""

_HOLE_NO_MESSAGE = """from tensorsketch import Node, Schema


class Polish(Node):
    class In(Schema):
        draft: str

    class Out(Schema):
        final: str

    async def run(self, ctx, inp):
        raise Hole()
"""


def test_finds_holes_and_captures_spec_and_line(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(_WITH_HOLES)
    holes = find_holes(tmp_path)
    assert len(holes) == 1  # only Draft; Ready has a real body
    (hole,) = holes
    assert hole.node == "Draft"
    assert hole.spec == "Write a first draft from the topic"
    assert hole.line == 4  # the `class Draft` line
    assert hole.file.endswith("a.py")


def test_hole_without_a_message_has_empty_spec(tmp_path: Path) -> None:
    (tmp_path / "p.py").write_text(_HOLE_NO_MESSAGE)
    (hole,) = find_holes(tmp_path)
    assert hole.node == "Polish"
    assert hole.spec == ""


def test_scans_a_directory_tree_sorted_by_file_then_line(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(_WITH_HOLES)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text(_HOLE_NO_MESSAGE)
    holes = find_holes(tmp_path)
    assert [h.node for h in holes] == ["Draft", "Polish"]  # a.py before sub/b.py
    assert [Path(h.file).name for h in holes] == ["a.py", "b.py"]


def test_non_recursive_skips_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(_WITH_HOLES)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text(_HOLE_NO_MESSAGE)
    holes = find_holes(tmp_path, recursive=False)
    assert [h.node for h in holes] == ["Draft"]


def test_unparseable_files_are_skipped_not_raised(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(_WITH_HOLES)
    (tmp_path / "broken.py").write_text("def oops(:\n")  # syntax error
    holes = find_holes(tmp_path)
    assert [h.node for h in holes] == ["Draft"]  # the broken file didn't hide the good one


def test_no_holes_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text(
        "from tensorsketch import Node, Schema\n\n\n"
        "class A(Node):\n"
        "    class In(Schema):\n        x: int\n\n"
        "    class Out(Schema):\n        x: int\n\n"
        "    async def run(self, ctx, inp):\n        return self.Out(x=inp.x)\n"
    )
    assert find_holes(tmp_path) == []


def test_to_dict_is_json_shaped() -> None:
    ref = HoleRef(file="a.py", node="Draft", spec="do the thing", line=4)
    assert ref.to_dict() == {"file": "a.py", "node": "Draft", "spec": "do the thing", "line": 4}


def test_server_load_holes_scans_the_files_directory(tmp_path: Path) -> None:
    served = tmp_path / "graph.py"
    served.write_text(_WITH_HOLES)
    (tmp_path / "more.py").write_text(_HOLE_NO_MESSAGE)
    result = load_holes(served)
    assert result["root"] == str(tmp_path)
    nodes = {h["node"] for h in result["holes"]}
    assert nodes == {"Draft", "Polish"}
