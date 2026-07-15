"""Project-wide hole surfacing — "N nodes still need code", across a whole codebase.

A node can declare its typed `In`/`Out` interface while its body is left as a `raise Hole(...)`
stub. `find_holes` scans a file or directory tree for those stubs and reports where each one is,
so tooling — the CLI, the Studio — can answer *what's left to implement?* at a glance.

Like `extract`, it's purely syntactic (a CST walk): it never imports your code, so it works on a
half-finished project with undefined names and missing imports.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .extract import _is_node_class, _trailing_name


@dataclass
class HoleRef:
    """One unfilled node body: which node needs code, in which file, its spec, and where."""

    file: str
    node: str
    spec: str  # the `Hole("…")` message, or "" if none was given
    line: int  # 1-based line of the node's `class` statement

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "node": self.node, "spec": self.spec, "line": self.line}


def find_holes(*paths: str | Path, recursive: bool = True) -> list[HoleRef]:
    """Every node stubbed with `raise Hole(...)` under the given files/directories.

    Directories are walked (recursively by default) for `*.py`; unreadable or unparseable files
    are skipped rather than raising, so a broken file elsewhere never hides the rest. With no
    paths, the current directory is scanned. Results are sorted by file, then line.
    """
    holes: list[HoleRef] = []
    for source_file in _iter_python_files(paths, recursive):
        try:
            text = source_file.read_text()
        except OSError:
            continue
        holes.extend(_holes_in(text, str(source_file)))
    holes.sort(key=lambda h: (h.file, h.line))
    return holes


def _iter_python_files(paths: tuple[str | Path, ...], recursive: bool) -> Iterator[Path]:
    seen: set[Path] = set()
    for raw in paths or (Path("."),):
        root = Path(raw)
        candidates = (
            (root.rglob("*.py") if recursive else root.glob("*.py")) if root.is_dir() else [root]
        )
        for path in candidates:
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _holes_in(source: str, filename: str) -> list[HoleRef]:
    try:
        wrapper = MetadataWrapper(cst.parse_module(source), unsafe_skip_copy=True)
    except cst.ParserSyntaxError:
        return []
    positions = wrapper.resolve(PositionProvider)
    holes: list[HoleRef] = []
    for stmt in wrapper.module.body:
        if not (isinstance(stmt, cst.ClassDef) and _is_node_class(stmt)):
            continue
        spec = _hole_spec(stmt)
        if spec is not None:
            line = positions[stmt].start.line
            holes.append(HoleRef(file=filename, node=stmt.name.value, spec=spec, line=line))
    return holes


class _HoleSpecFinder(cst.CSTVisitor):
    """Captures the message of a `raise Hole(...)` in a node body (None = no hole at all)."""

    def __init__(self) -> None:
        self.spec: str | None = None

    def visit_Raise(self, node: cst.Raise) -> None:
        if isinstance(node.exc, cst.Call) and _trailing_name(node.exc.func) == "Hole":
            self.spec = _first_string_arg(node.exc)


def _hole_spec(cls: cst.ClassDef) -> str | None:
    finder = _HoleSpecFinder()
    cls.body.visit(finder)
    return finder.spec


def _first_string_arg(call: cst.Call) -> str:
    for arg in call.args:
        if arg.keyword is None and isinstance(arg.value, cst.SimpleString):
            value = arg.value.evaluated_value
            return value if isinstance(value, str) else ""
    return ""
