"""Ergonomic `>>` wiring — an operator surface over the `Graph` builder.

`Graph.nodes(A, B, ...)` adds nodes and hands back **handles**; the handles overload `>>` so
wiring reads like a diagram::

    classify, billing, tech = g.nodes(Classify, Billing, Tech)

    START >> classify                                       # entry
    classify >> Router(route, billing=billing, tech=tech)   # conditional fan-out
    billing >> END                                          # terminate a branch
    tech >> END

Every operator just calls the same `.add`/`.edge`/`.conditional` underneath, so this is pure
sugar: the compiled graph is byte-for-byte identical to the fluent-builder form. `START` and
`END` stay plain strings — the handles' `__rshift__`/`__rrshift__` do all the work, so nothing
in the runtime changes.

Supported shapes:

* ``a >> b``            — sequential edge ``a → b``
* ``a >> [b, c]``       — fan-out (two sequential edges)
* ``START >> a``        — set the entry node
* ``a >> END``          — terminate the branch out of ``a``
* ``a >> Router(fn)``   — dynamic conditional (targets decided at runtime)
* ``a >> Router(fn, {"k": b})`` / ``Router(fn, k=b)`` — mapped conditional
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .graph import START

if TYPE_CHECKING:
    from .graph import Graph


def _name(target: NodeHandle | str) -> str:
    """The graph name of a wiring endpoint (a handle or an already-resolved node name)."""
    return target.name if isinstance(target, NodeHandle) else target


class Router:
    """A conditional edge in the `>>` surface: a routing function and an optional target map.

    ``Router(fn)`` routes dynamically — `fn(state)` returns the next node name(s) directly.
    ``Router(fn, {"billing": billing})`` or ``Router(fn, billing=billing)`` map `fn`'s return
    value through named targets (handles or names). The two forms mirror `Graph.conditional`'s
    optional `mapping`.
    """

    def __init__(
        self,
        path: Callable[[Any], str | list[str]],
        mapping: dict[str, NodeHandle | str] | None = None,
        **targets: NodeHandle | str,
    ) -> None:
        merged: dict[str, NodeHandle | str] = {**(mapping or {}), **targets}
        self.path = path
        self.mapping: dict[str, str] | None = (
            {key: _name(value) for key, value in merged.items()} if merged else None
        )


class NodeHandle:
    """A reference to a node already added to a `Graph`, wired with `>>`.

    Handles come from `Graph.nodes(...)` or `graph[name]`. Chaining returns the right-hand
    operand so `a >> b >> c` reads left to right.
    """

    __slots__ = ("_graph", "name")

    def __init__(self, graph: Graph[Any], name: str) -> None:
        self._graph = graph
        self.name = name

    def __rshift__(self, other: NodeHandle | Router | list[Any] | tuple[Any, ...] | str) -> Any:
        if isinstance(other, Router):
            self._graph.conditional(self.name, other.path, other.mapping)
            return other
        if isinstance(other, (list, tuple)):
            for target in other:
                self._graph.edge(self.name, _name(target))
            return other
        if isinstance(other, (NodeHandle, str)):
            self._graph.edge(self.name, _name(other))
            return other
        return NotImplemented

    def __rrshift__(self, other: str) -> NodeHandle:
        # Reached for `START >> handle` / `"Node" >> handle` — the left side (a plain str) has
        # no `__rshift__`, so Python defers to us.
        if other == START:
            self._graph.entry(self.name)
        else:
            self._graph.edge(other, self.name)
        return self

    def __repr__(self) -> str:
        return f"NodeHandle({self.name!r})"
