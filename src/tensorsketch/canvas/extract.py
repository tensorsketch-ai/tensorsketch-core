"""Extract a `GraphIR` from TensorSketch source with a CST (code → graph).

Parsing is purely syntactic (libcst), so extraction works on **incomplete** code — undefined
names, missing imports, unfilled holes. We read two things and nothing else:

* every `class X(Node)` — its name, its `In`/`Out` port fields, and whether its body is a hole;
* the fluent `Graph(State).add(...).edge(...).conditional(...)` builder chain — the wiring.

Node *bodies* are never interpreted. That's the whole point: wiring + typed interfaces are a
syntactic surface that round-trips; bodies are opaque.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import libcst as cst

from .ir import END, START, EdgeIR, GraphIR, NodeIR, Port


def extract(source: str) -> GraphIR:
    """Parse TensorSketch source and return its graph structure."""
    module = cst.parse_module(source)
    nodes = [
        _node_ir(stmt, module)
        for stmt in module.body
        if isinstance(stmt, cst.ClassDef) and _is_node_class(stmt)
    ]
    state, entry, added, edges = _collect_wiring(module)
    return GraphIR(state=state, entry=entry, nodes=nodes, added=added, edges=edges)


# -- nodes -----------------------------------------------------------------------------------


def _trailing_name(expr: cst.BaseExpression) -> str | None:
    """The final name in `Name` or `a.b.Name` — used to match `Node`, `Graph`, `Hole`, etc."""
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        return expr.attr.value
    return None


def _is_node_class(cls: cst.ClassDef) -> bool:
    return any(_trailing_name(base.value) == "Node" for base in cls.bases)


def _node_ir(cls: cst.ClassDef, module: cst.Module) -> NodeIR:
    inputs: list[Port] = []
    outputs: list[Port] = []
    has_hole = False
    for stmt in cls.body.body:
        if isinstance(stmt, cst.ClassDef):
            if stmt.name.value == "In":
                inputs = _ports(stmt, module)
            elif stmt.name.value == "Out":
                outputs = _ports(stmt, module)
        elif isinstance(stmt, cst.FunctionDef) and stmt.name.value == "run":
            has_hole = _has_hole(stmt)
    return NodeIR(name=cls.name.value, inputs=inputs, outputs=outputs, has_hole=has_hole)


def _ports(cls: cst.ClassDef, module: cst.Module) -> list[Port]:
    ports: list[Port] = []
    for stmt in cls.body.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
                type_src = module.code_for_node(small.annotation.annotation).strip()
                ports.append(Port(name=small.target.value, type=type_src))
    return ports


class _HoleFinder(cst.CSTVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_Raise(self, node: cst.Raise) -> None:
        if isinstance(node.exc, cst.Call) and _trailing_name(node.exc.func) == "Hole":
            self.found = True


def _has_hole(fn: cst.FunctionDef) -> bool:
    finder = _HoleFinder()
    fn.body.visit(finder)
    return finder.found


# -- wiring ----------------------------------------------------------------------------------
#
# Wiring can be authored three ways, all of which extract to the same GraphIR:
#   * a fluent chain           — `app = Graph(S).add(A).edge(x, y).conditional(...)`
#   * statement style          — `g = Graph(S); g.add(A); g.edge(x, y)`
#   * the `>>` operator surface — `a, b = g.nodes(A, B); START >> a; a >> Router(fn, ...)`
# We locate the `Graph(...)` construction, then fold every wiring operation on it — whether
# chained, statement-level, or via `>>` — into (entry, added, edges).


class _ChainFinder(cst.CSTVisitor):
    """Finds the outermost `Graph(...)....` builder expression (the first one wins)."""

    def __init__(self) -> None:
        self.chain: cst.Call | None = None

    def visit_Call(self, node: cst.Call) -> bool:
        if self.chain is None and _is_graph_chain(node):
            self.chain = node
            return False  # keep the outermost; don't descend into inner calls
        return True


def _unwind(call: cst.Call) -> tuple[cst.Call | None, list[tuple[str, list[cst.Arg]]]]:
    """Split `Graph(S).add(A).edge(x, y)` into the base `Graph(S)` call and its method calls."""
    methods: list[tuple[str, list[cst.Arg]]] = []
    node: cst.BaseExpression = call
    while isinstance(node, cst.Call):
        if isinstance(node.func, cst.Attribute):
            methods.append((node.func.attr.value, list(node.args)))
            node = node.func.value
        else:
            return node, list(reversed(methods))
    return None, list(reversed(methods))


def _is_graph_chain(call: cst.Call) -> bool:
    base, _ = _unwind(call)
    return base is not None and _trailing_name(base.func) == "Graph"


class _AssignFinder(cst.CSTVisitor):
    """Finds the variable a builder expression is assigned to (`g = Graph(...)` → `g`)."""

    def __init__(self, value: cst.BaseExpression) -> None:
        self.value = value
        self.name: str | None = None

    def visit_Assign(self, node: cst.Assign) -> None:
        if node.value is self.value and len(node.targets) == 1:
            target = node.targets[0].target
            if isinstance(target, cst.Name):
                self.name = target.value

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        # `g: Graph[Support] = Graph(Support)` binds the graph var just like a plain assignment.
        if node.value is self.value and isinstance(node.target, cst.Name):
            self.name = node.target.value


class _StatementWiring(cst.CSTVisitor):
    """Collects statement-level wiring on the graph variable, in source order.

    Three op shapes, each normalized: ``("method", name, args)`` for `g.add(...)`/`g.edge(...)`;
    ``("nodes", targets, args)`` for `a, b = g.nodes(A, B)`; ``("rshift", operands)`` for a
    `START >> a >> ...` statement.
    """

    def __init__(self, gvar: str) -> None:
        self.gvar = gvar
        self.ops: list[tuple[str, Any, Any]] = []

    def visit_Assign(self, node: cst.Assign) -> None:
        if (
            isinstance(node.value, cst.Call)
            and _receiver_name(node.value) == self.gvar
            and _method_name(node.value) == "nodes"
            and len(node.targets) == 1
        ):
            targets = _target_names(node.targets[0].target)
            self.ops.append(("nodes", targets, list(node.value.args)))

    def visit_Expr(self, node: cst.Expr) -> None:
        value = node.value
        if isinstance(value, cst.BinaryOperation) and isinstance(value.operator, cst.RightShift):
            self.ops.append(("rshift", _flatten_rshift(value), None))
        elif isinstance(value, cst.Call) and _receiver_name(value) == self.gvar:
            _, methods = _unwind(value)
            for name, args in methods:
                self.ops.append(("method", name, args))


def _receiver_name(call: cst.Call) -> str | None:
    """The root variable a method chain is called on: `g.add(A).edge(x)` → `g` (a bare Name)."""
    node: cst.BaseExpression = call
    while isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute):
        node = node.func.value
    return node.value if isinstance(node, cst.Name) else None


def _method_name(call: cst.Call) -> str | None:
    return call.func.attr.value if isinstance(call.func, cst.Attribute) else None


def _target_names(target: cst.BaseExpression) -> list[str]:
    """The bound names of an assignment target: `a` → [a]; `a, b` / `[a, b]` → [a, b]."""
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)):
        return [e.value.value for e in target.elements if isinstance(e.value, cst.Name)]
    return []


def _flatten_rshift(expr: cst.BaseExpression) -> list[cst.BaseExpression]:
    """Flatten a left-associative `a >> b >> c` chain into `[a, b, c]`."""
    if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.RightShift):
        return [*_flatten_rshift(expr.left), expr.right]
    return [expr]


def _collect_wiring(
    module: cst.Module,
) -> tuple[str | None, str | None, list[str], list[EdgeIR]]:
    finder = _ChainFinder()
    module.visit(finder)
    if finder.chain is None:
        return None, None, [], []

    base, methods = _unwind(finder.chain)
    state = _endpoint(base.args[0].value) if base and base.args else None

    assign = _AssignFinder(finder.chain)
    module.visit(assign)

    ops: list[tuple[str, Any, Any]] = [("method", name, args) for name, args in methods]
    if assign.name is not None:
        collector = _StatementWiring(assign.name)
        module.visit(collector)
        ops.extend(collector.ops)

    return _fold(state, ops)


def _fold(
    state: str | None, ops: list[tuple[str, Any, Any]]
) -> tuple[str | None, str | None, list[str], list[EdgeIR]]:
    entry: str | None = None
    added: list[str] = []
    edges: list[EdgeIR] = []
    handles: dict[str, str] = {}

    for kind, a, b in ops:
        if kind == "method":
            entry = _apply_method(a, b, added, edges) or entry
        elif kind == "nodes":
            for var, node_name in zip(a, _node_names(b), strict=False):
                added.append(node_name)
                handles[var] = node_name
        elif kind == "rshift":
            entry = _apply_rshift(a, handles, edges) or entry
    return state, entry, added, edges


def _apply_method(
    name: str, args: list[cst.Arg], added: list[str], edges: list[EdgeIR]
) -> str | None:
    """Fold one builder method into added/edges.

    Handles `add`/`edge`/`entry`, `conditional` and its intent-named alias `router` (identical
    routing), and `loop(node, until, *, exit=END)` — sugar for a self-conditional, extracted as a
    two-branch conditional (`loop` back to the node, `exit` onward) so the canvas shows the cycle.
    """
    positional = [a.value for a in args if a.keyword is None]
    if name == "add" and positional:
        added.append(_add_name(args, positional))
    elif name == "edge" and len(positional) >= 2:
        source, target = _endpoint(positional[0]), _endpoint(positional[1])
        edges.append(EdgeIR(source=source, target=target, kind="sequential"))
        if source == START:
            return target
    elif name == "entry" and positional:
        entry = _endpoint(positional[0])
        edges.append(EdgeIR(source=START, target=entry, kind="sequential"))
        return entry
    elif name in ("conditional", "router") and positional:
        src = _endpoint(positional[0])
        condition = _trailing_name(positional[1]) if len(positional) >= 2 else None
        mapping = _dict(positional[2]) if len(positional) >= 3 else None
        _emit_conditional(src, condition, mapping, edges)
    elif name == "loop" and positional:
        node = _endpoint(positional[0])
        condition = _trailing_name(positional[1]) if len(positional) >= 2 else None
        _emit_conditional(node, condition, {"loop": node, "exit": _loop_exit(args)}, edges)
    return None


def _loop_exit(args: list[cst.Arg]) -> str:
    """The `exit=` target of a `.loop(...)` call — where it goes when it stops (default END)."""
    for arg in args:
        if arg.keyword is not None and arg.keyword.value == "exit":
            return _endpoint(arg.value)
    return END


def _apply_rshift(
    operands: list[cst.BaseExpression], handles: dict[str, str], edges: list[EdgeIR]
) -> str | None:
    """Fold a `START >> a >> Router(...)` statement into edges; return an entry if one is set."""
    resolved = [_resolve_operand(op, handles) for op in operands]
    entry: str | None = None
    for left, right in pairwise(resolved):
        if left[0] == "start" and right[0] == "node":
            entry = right[1]
            edges.append(EdgeIR(source=START, target=right[1], kind="sequential"))
        elif left[0] == "node" and right[0] == "end":
            edges.append(EdgeIR(source=left[1], target=END, kind="sequential"))
        elif left[0] == "node" and right[0] == "router":
            _emit_conditional(left[1], right[1], right[2], edges)
        elif left[0] == "node" and right[0] == "list":
            edges.extend(EdgeIR(left[1], t, kind="sequential") for t in right[1])
        elif left[0] == "node" and right[0] == "node":
            edges.append(EdgeIR(source=left[1], target=right[1], kind="sequential"))
    return entry


def _resolve_operand(expr: cst.BaseExpression, handles: dict[str, str]) -> tuple[str, Any, Any]:
    """Resolve a `>>` operand to a tagged descriptor the folder can pair up."""
    trailing = _trailing_name(expr)
    if trailing == "START":
        return ("start", None, None)
    if trailing == "END":
        return ("end", None, None)
    if isinstance(expr, cst.Call) and _trailing_name(expr.func) == "Router":
        return ("router", *_router_spec(expr, handles))
    if isinstance(expr, (cst.List, cst.Tuple)):
        names = [handles.get(_name_of(e.value), _name_of(e.value)) for e in expr.elements]
        return ("list", names, None)
    if isinstance(expr, cst.Subscript):  # g["Name"]
        return ("node", _subscript_key(expr), None)
    name = _name_of(expr)
    return ("node", handles.get(name, name), None)


def _router_spec(
    call: cst.Call, handles: dict[str, str]
) -> tuple[str | None, dict[str, str] | None]:
    """The `(condition, mapping)` of a `Router(fn, {...}/**targets)` call in a `>>` chain."""
    positional = [a for a in call.args if a.keyword is None]
    condition = _trailing_name(positional[0].value) if positional else None
    mapping: dict[str, str] = {}
    if len(positional) >= 2 and isinstance(positional[1].value, cst.Dict):
        raw = _dict(positional[1].value) or {}
        mapping.update({k: handles.get(v, v) for k, v in raw.items()})
    for arg in call.args:
        if arg.keyword is not None:
            target = _name_of(arg.value)
            mapping[arg.keyword.value] = handles.get(target, target)
    return condition, (mapping or None)


def _emit_conditional(
    src: str, condition: str | None, mapping: dict[str, str] | None, edges: list[EdgeIR]
) -> None:
    if mapping:
        for key, target in mapping.items():
            edges.append(EdgeIR(src, target, kind="conditional", condition=condition, key=key))
    else:
        edges.append(EdgeIR(src, None, kind="conditional", condition=condition))


def _node_names(args: list[cst.Arg]) -> list[str]:
    return [_name_of(a.value) for a in args if a.keyword is None]


def _name_of(expr: cst.BaseExpression) -> str:
    if isinstance(expr, cst.SimpleString):
        value = expr.evaluated_value
        return value if isinstance(value, str) else expr.value
    if isinstance(expr, cst.Subscript):
        return _subscript_key(expr)
    return _trailing_name(expr) or ""


def _subscript_key(expr: cst.Subscript) -> str:
    element = expr.slice[0].slice
    if isinstance(element, cst.Index):
        return _name_of(element.value)
    return ""


def _add_name(args: list[cst.Arg], positional: list[cst.BaseExpression]) -> str:
    """The node's graph name: an explicit `name=` keyword, else the added class's name."""
    for arg in args:
        if arg.keyword is not None and arg.keyword.value == "name":
            return _endpoint(arg.value)
    return _trailing_name(positional[0]) or ""


def _endpoint(expr: cst.BaseExpression) -> str:
    """Resolve a node reference: START/END sentinels, a string node name, or a bare name."""
    if isinstance(expr, cst.SimpleString):
        value = expr.evaluated_value
        return value if isinstance(value, str) else expr.value
    name = _trailing_name(expr)
    if name == "START":
        return START
    if name == "END":
        return END
    return name if name is not None else ""


def _dict(expr: cst.BaseExpression) -> dict[str, str] | None:
    if not isinstance(expr, cst.Dict):
        return None
    result: dict[str, str] = {}
    for element in expr.elements:
        if isinstance(element, cst.DictElement):
            key = _endpoint(element.key)
            value = _endpoint(element.value)
            result[key] = value
    return result
