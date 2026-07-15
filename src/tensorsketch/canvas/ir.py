"""The graph IR — the neutral structure that sits between code and canvas.

Extraction produces a `GraphIR`; a canvas renders one; write-back consumes edits to one. It
captures exactly what round-trips — nodes, their typed ports, and the wiring — and nothing about
node *bodies* (which are opaque). Everything is plain data with a `to_dict()` for JSON, so a
frontend can consume it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Sentinels mirroring `tensorsketch.START` / `tensorsketch.END` in extracted wiring.
START = "__start__"
END = "__end__"


@dataclass
class Port:
    """One typed port of a node — a field of its `In` or `Out` schema."""

    name: str
    type: str  # the annotation as source text, e.g. "str" or "Literal['a', 'b']"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Port:
        return cls(name=data["name"], type=data["type"])


@dataclass
class NodeIR:
    """A node: its name, typed input/output ports, and whether its body is an unfilled hole."""

    name: str
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    has_hole: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "has_hole": self.has_hole,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeIR:
        return cls(
            name=data["name"],
            inputs=[Port.from_dict(p) for p in data.get("inputs", [])],
            outputs=[Port.from_dict(p) for p in data.get("outputs", [])],
            has_hole=data.get("has_hole", False),
        )


@dataclass
class EdgeIR:
    """A wire between nodes. `target` is None for a dynamic conditional with no static mapping."""

    source: str
    target: str | None
    kind: str = "sequential"  # "sequential" | "conditional"
    condition: str | None = None  # the routing function name, for conditional edges
    key: str | None = None  # the routing value that selects `target`, for a mapped conditional

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "condition": self.condition,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeIR:
        return cls(
            source=data["source"],
            target=data.get("target"),
            kind=data.get("kind", "sequential"),
            condition=data.get("condition"),
            key=data.get("key"),
        )


@dataclass
class GraphIR:
    """A whole graph: the state type, entry node, defined nodes, added nodes, and edges.

    `nodes` are the `Node` classes defined in the file (with ports — the canvas palette). `added`
    is the ordered list of node names actually wired into this graph via `.add(...)`.
    """

    state: str | None = None
    entry: str | None = None
    nodes: list[NodeIR] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    edges: list[EdgeIR] = field(default_factory=list)

    def node(self, name: str) -> NodeIR | None:
        return next((n for n in self.nodes if n.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "entry": self.entry,
            "nodes": [n.to_dict() for n in self.nodes],
            "added": list(self.added),
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphIR:
        return cls(
            state=data.get("state"),
            entry=data.get("entry"),
            nodes=[NodeIR.from_dict(n) for n in data.get("nodes", [])],
            added=list(data.get("added", [])),
            edges=[EdgeIR.from_dict(e) for e in data.get("edges", [])],
        )
