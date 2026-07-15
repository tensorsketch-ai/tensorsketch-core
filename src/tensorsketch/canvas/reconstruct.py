"""Write a (possibly edited) `GraphIR` back into source (canvas → code).

A canvas edit changes the *wiring*, never a node body. So write-back regenerates only the graph
definition — from the IR, in a canonical form — and leaves everything else (node classes and
their bodies, imports, comments, unrelated code) untouched.

Whatever authoring style the source used — a fluent chain, statement-style calls, or the `>>`
operator surface — write-back emits **one canonical fluent chain** assigned to the graph
variable, and drops the now-redundant wiring statements. That keeps a single normal form, which
is what makes the round-trip invariant hold:

    extract(reconstruct(source, extract(source))) == extract(source)

— re-extracting reconstructed source yields the same graph. As a corollary, reconstruct also
*canonicalizes* wiring: `.entry(x)` becomes `.edge(START, "x")`, `a >> b` becomes `.edge(...)`,
and so on — without ever changing what the graph *is*.
"""

from __future__ import annotations

import libcst as cst

from .extract import (
    _AssignFinder,
    _ChainFinder,
    _is_node_class,
    _method_name,
    _receiver_name,
    _trailing_name,
    _unwind,
)
from .ir import END, START, EdgeIR, GraphIR, NodeIR, Port

_WIRING_METHODS = {"add", "edge", "conditional", "router", "loop", "entry"}

#: A module-body statement — the element type of `cst.Module.body`.
_Stmt = cst.SimpleStatementLine | cst.BaseCompoundStatement


def reconstruct(source: str, ir: GraphIR) -> str:
    """Return `source` with its graph definition rebuilt from `ir` (everything else intact).

    If the source has no `Graph(...)` builder, it is returned unchanged.
    """
    module = cst.parse_module(source)

    finder = _ChainFinder()
    module.visit(finder)
    if finder.chain is None:
        return source

    anchor = _AnchorFinder(finder.chain)
    module.visit(anchor)
    if anchor.value is None or anchor.line is None:
        return source

    assign = _AssignFinder(finder.chain)
    module.visit(assign)

    wiring_lines: list[cst.SimpleStatementLine] = []
    if assign.name is not None:
        lines = _WiringLines(assign.name, anchor.line)
        module.visit(lines)
        wiring_lines = lines.lines

    tail = _tail_methods(finder.chain, module)
    new_expr = cst.parse_expression(_chain_source(ir, tail))

    rewriter = _GraphRewriter(anchor.value, wiring_lines, new_expr)
    rewritten = module.visit(rewriter)

    # Node-creation: any node the IR references but the source never defined is a canvas-created
    # stub. Generate its `class X(Node)` (typed ports + a `Hole` body) and splice it in.
    new_classes = _new_node_classes(module, ir)
    return _insert_defs(rewritten, new_classes).code


class _AnchorFinder(cst.CSTVisitor):
    """Finds the statement that holds the builder chain — the one whose value we rewrite."""

    def __init__(self, chain: cst.Call) -> None:
        self.chain = chain
        self.value: cst.Assign | cst.AnnAssign | cst.Expr | None = None
        self.line: cst.SimpleStatementLine | None = None

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        for small in node.body:
            if (
                isinstance(small, (cst.Assign, cst.AnnAssign, cst.Expr))
                and small.value is self.chain
            ):
                self.value = small
                self.line = node


class _WiringLines(cst.CSTVisitor):
    """Collects the statement-level wiring lines to drop (folded into the canonical chain)."""

    def __init__(self, gvar: str, anchor: cst.SimpleStatementLine) -> None:
        self.gvar = gvar
        self.anchor = anchor
        self.lines: list[cst.SimpleStatementLine] = []

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        if node is not self.anchor and _is_wiring_line(node, self.gvar):
            self.lines.append(node)


def _is_wiring_line(line: cst.SimpleStatementLine, gvar: str) -> bool:
    for small in line.body:
        value = getattr(small, "value", None)
        if isinstance(small, cst.Expr):
            if isinstance(value, cst.BinaryOperation) and isinstance(
                value.operator, cst.RightShift
            ):
                return True
            if (
                isinstance(value, cst.Call)
                and _receiver_name(value) == gvar
                and _wiring_call(value)
            ):
                return True
        elif isinstance(small, cst.Assign):
            if (
                isinstance(value, cst.Call)
                and _receiver_name(value) == gvar
                and _method_name(value) == "nodes"
            ):
                return True
    return False


def _wiring_call(call: cst.Call) -> bool:
    """True if every method on `g.add(...).edge(...)` is a wiring method (not e.g. `.compile()`)."""
    _, methods = _unwind(call)
    return bool(methods) and all(name in _WIRING_METHODS for name, _ in methods)


class _GraphRewriter(cst.CSTTransformer):
    """Swaps the anchor statement's value for the canonical chain; removes folded wiring lines."""

    def __init__(
        self,
        anchor: cst.Assign | cst.AnnAssign | cst.Expr,
        wiring_lines: list[cst.SimpleStatementLine],
        new_expr: cst.BaseExpression,
    ) -> None:
        self.anchor = anchor
        self.wiring = {id(line) for line in wiring_lines}
        self.new_expr = new_expr

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        if original_node is self.anchor:
            return updated_node.with_changes(value=self.new_expr)
        return updated_node

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.AnnAssign:
        if original_node is self.anchor:
            return updated_node.with_changes(value=self.new_expr)
        return updated_node

    def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> cst.Expr:
        if original_node is self.anchor:
            return updated_node.with_changes(value=self.new_expr)
        return updated_node

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
        if id(original_node) in self.wiring:
            return cst.RemoveFromParent()
        return updated_node


# -- canonical chain rendering --------------------------------------------------------------


def _tail_methods(chain: cst.Call, module: cst.Module) -> list[str]:
    """Preserve trailing non-wiring calls on the chain (e.g. `.compile()`) verbatim."""
    _, methods = _unwind(chain)
    cut = len(methods)
    while cut > 0 and methods[cut - 1][0] not in _WIRING_METHODS:
        cut -= 1
    tail: list[str] = []
    for name, args in methods[cut:]:
        rendered = ", ".join(_render_arg(arg, module) for arg in args)
        tail.append(f".{name}({rendered})")
    return tail


def _render_arg(arg: cst.Arg, module: cst.Module) -> str:
    value = module.code_for_node(arg.value)
    if arg.keyword is not None:
        return f"{arg.keyword.value}={value}"
    return value


def _chain_source(ir: GraphIR, tail: list[str]) -> str:
    parts = [f"Graph({ir.state})"]
    parts.extend(f".add({name})" for name in ir.added)

    emitted_conditionals: set[str] = set()
    for edge in ir.edges:
        if edge.kind == "conditional":
            if edge.source in emitted_conditionals:
                continue
            emitted_conditionals.add(edge.source)
            group = [e for e in ir.edges if e.kind == "conditional" and e.source == edge.source]
            parts.append(_render_conditional(edge.source, group))
        else:
            parts.append(f".edge({_endpoint(edge.source)}, {_endpoint(edge.target)})")

    parts.extend(tail)
    # Indentation is *relative*: libcst prepends the enclosing block's indent when it renders
    # this expression, so a fixed 4-space continuation lands correctly at any nesting depth.
    return "(\n    " + "\n    ".join(parts) + "\n)"


def _render_conditional(source: str, group: list[EdgeIR]) -> str:
    condition = group[0].condition or "..."
    mapping = {e.key: e.target for e in group if e.key is not None and e.target is not None}
    if mapping:
        pairs = ", ".join(f'"{key}": {_endpoint(target)}' for key, target in mapping.items())
        return f".conditional({_endpoint(source)}, {condition}, {{{pairs}}})"
    return f".conditional({_endpoint(source)}, {condition})"


def _endpoint(value: str | None) -> str:
    if value == START:
        return "START"
    if value == END:
        return "END"
    if value is None:
        return "None"
    return f'"{value}"'


# -- node-class generation (canvas node-creation) -------------------------------------------
#
# A canvas edit is usually pure wiring, but *creating* a node has no wiring-only expression: the
# new node needs a `class X(Node)` to exist. So when the IR carries a node the source never
# defined, we synthesize a stub — its typed `In`/`Out` ports and a `raise Hole(...)` body — and
# insert it before the graph builder. The stub re-extracts to the identical `NodeIR`
# (`has_hole=True`), so the round-trip invariant still holds after a node is born on the canvas.


def _new_node_classes(module: cst.Module, ir: GraphIR) -> list[_Stmt]:
    """Parse a `class X(Node)` stub for every IR node with no class defined in `module`."""
    defined = {
        stmt.name.value
        for stmt in module.body
        if isinstance(stmt, cst.ClassDef) and _is_node_class(stmt)
    }
    fresh = [node for node in ir.nodes if node.name not in defined]
    if not fresh:
        return []
    block = "\n\n\n".join(_node_class_source(node) for node in fresh)
    return list(cst.parse_module(block).body)


def _node_class_source(node: NodeIR) -> str:
    """Render an idiomatic node stub: typed `In`/`Out` schemas and a `Hole` body."""
    lines = [
        f"class {node.name}(Node):",
        "    class In(Schema):",
        *(f"        {line}" for line in _schema_lines(node.inputs)),
        "",
        "    class Out(Schema):",
        *(f"        {line}" for line in _schema_lines(node.outputs)),
        "",
        "    async def run(self, ctx: Context, inp: In) -> Out:",
        f'        raise Hole("{node.name} needs code")',
    ]
    return "\n".join(lines)


def _schema_lines(ports: list[Port]) -> list[str]:
    """One `name: type` field per port, or `pass` for a port-less schema."""
    if not ports:
        return ["pass"]
    return [f"{port.name}: {port.type}" for port in ports]


def _insert_defs(module: cst.Module, new_classes: list[_Stmt]) -> cst.Module:
    """Splice generated node classes in before the graph builder, adding a `Hole` import if used."""
    if not new_classes:
        return module
    body = list(module.body)
    anchor = _toplevel_anchor_index(module)
    first = new_classes[0].with_changes(leading_lines=[cst.EmptyLine(), cst.EmptyLine()])
    body[anchor:anchor] = [first, *new_classes[1:]]
    if not _binds_name(module, "Hole"):
        import_line = cst.parse_statement("from tensorsketch import Hole\n")
        body.insert(_import_insert_index(module), import_line)
    return module.with_changes(body=body)


class _Contains(cst.CSTVisitor):
    """Flags whether a specific node (by identity) appears anywhere in a subtree."""

    def __init__(self, target: cst.CSTNode) -> None:
        self.target = target
        self.found = False

    def on_visit(self, node: cst.CSTNode) -> bool:
        if node is self.target:
            self.found = True
        return not self.found


def _toplevel_anchor_index(module: cst.Module) -> int:
    """Index of the top-level statement containing the graph builder (new classes go before it)."""
    finder = _ChainFinder()
    module.visit(finder)
    if finder.chain is None:
        return len(module.body)
    for index, stmt in enumerate(module.body):
        matcher = _Contains(finder.chain)
        stmt.visit(matcher)
        if matcher.found:
            return index
    return len(module.body)


def _import_insert_index(module: cst.Module) -> int:
    """Where a new import line belongs: after the last existing import (past any docstring)."""
    index = 0
    for i, stmt in enumerate(module.body):
        if _is_import_line(stmt):
            index = i + 1
        elif i == 0 and index == 0 and _is_docstring(stmt):
            index = 1
    return index


def _is_import_line(stmt: cst.BaseStatement) -> bool:
    return isinstance(stmt, cst.SimpleStatementLine) and any(
        isinstance(small, (cst.Import, cst.ImportFrom)) for small in stmt.body
    )


def _is_docstring(stmt: cst.BaseStatement) -> bool:
    return (
        isinstance(stmt, cst.SimpleStatementLine)
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], cst.Expr)
        and isinstance(stmt.body[0].value, cst.SimpleString)
    )


def _binds_name(module: cst.Module, name: str) -> bool:
    """True if `name` is already bound by an import (so we don't add a duplicate)."""
    for stmt in module.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if any(_bound_name(alias) == name for alias in _import_aliases(small)):
                return True
    return False


def _import_aliases(small: cst.BaseSmallStatement) -> list[cst.ImportAlias]:
    """The bound aliases of an import statement (empty for `from x import *` or a non-import)."""
    if isinstance(small, cst.ImportFrom):
        return [] if isinstance(small.names, cst.ImportStar) else list(small.names)
    if isinstance(small, cst.Import):
        return list(small.names)
    return []


def _bound_name(alias: cst.ImportAlias) -> str | None:
    asname = alias.asname
    if asname is not None and isinstance(asname.name, cst.Name):
        return asname.name.value
    return _trailing_name(alias.name)
