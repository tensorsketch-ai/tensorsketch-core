"""Write a (possibly edited) `GraphIR` back into source (canvas → code).

A canvas edit changes the *wiring*, never a node body. So write-back regenerates only the graph
definition — from the IR — and leaves everything else (node classes and their bodies, imports,
comments, unrelated code) untouched.

Write-back is **style-preserving**: it detects how the source authored its wiring and re-emits in
that same style, so a `>>`-operator graph stays `>>` and a statement-style graph stays statements
(rather than everything collapsing to one canonical chain). The three styles are:

* **fluent** — `app = Graph(S).add(A).edge(x, y).conditional(...)` (one chained expression);
* **statement** — `g = Graph(S)` then separate `g.add(A)` / `g.edge(x, y)` lines;
* **arrow** — `a, b = g.nodes(A, B)` then `START >> a >> Router(...)` lines.

All three share one wiring walk (`_wiring_items`), so whichever style is emitted lists the edges
in the *same order* — which is what makes the round-trip invariant hold as a list equality:

    extract(reconstruct(source, extract(source))) == extract(source)

— re-extracting reconstructed source yields the same graph. Within a style, minor forms are still
normalized (`.entry(x)` → `.edge(START, "x")`; a fluent chain is re-indented) — never changing
what the graph *is*, only tidying how it reads.
"""

from __future__ import annotations

import keyword

import libcst as cst

from .extract import (
    _AssignFinder,
    _ChainFinder,
    _is_node_class,
    _method_name,
    _receiver_name,
    _StatementWiring,
    _trailing_name,
    _unwind,
)
from .ir import END, START, EdgeIR, GraphIR, NodeIR, Port

_WIRING_METHODS = {"add", "edge", "conditional", "router", "loop", "entry"}

#: A module-body statement — the element type of `cst.Module.body`.
_Stmt = cst.SimpleStatementLine | cst.BaseCompoundStatement


def reconstruct(source: str, ir: GraphIR) -> str:
    """Return `source` with its graph definition rebuilt from `ir` (everything else intact).

    The rebuild keeps the source's authoring style (fluent / statement / arrow). If the source has
    no `Graph(...)` builder, it is returned unchanged.
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
    style = "fluent"
    if assign.name is not None:
        lines = _WiringLines(assign.name, anchor.line)
        module.visit(lines)
        wiring_lines = lines.lines
        style = _detect_style(module, assign.name, finder.chain)

    tail = _tail_methods(finder.chain, module)

    if style == "fluent" or assign.name is None:
        # One chained expression assigned to the graph variable — fold all wiring into it.
        new_expr: cst.BaseExpression = cst.parse_expression(_chain_source(ir, tail))
        insert: list[cst.BaseStatement] = []
    else:
        # Statement / arrow: the construction stays bare (`g = Graph(S)`) and the wiring becomes
        # separate lines inserted right after it — matching how the author wrote it.
        new_expr = cst.parse_expression(_base_source(ir, tail))
        texts = (
            _statement_wiring(ir, assign.name)
            if style == "statement"
            else _arrow_wiring(ir, assign.name)
        )
        insert = [cst.parse_statement(f"{text}\n") for text in texts]

    rewriter = _GraphRewriter(anchor.value, anchor.line, wiring_lines, new_expr, insert)
    rewritten = module.visit(rewriter)

    # Arrow style renders conditionals as `>> Router(...)`; ensure the name is bound if a canvas
    # edit introduced the graph's first conditional into an otherwise Router-free arrow graph.
    if style == "arrow" and any(kind == "cond" for kind, _, _ in _wiring_items(ir)):
        rewritten = _ensure_import(rewritten, "Router", "from tensorsketch import Router\n")

    # Node-creation: any node the IR references but the source never defined is a canvas-created
    # stub. Generate its `class X(Node)` (typed ports + a `Hole` body) and splice it in.
    new_classes = _new_node_classes(module, ir)
    return _insert_defs(rewritten, new_classes).code


def _detect_style(module: cst.Module, gvar: str, chain: cst.Call) -> str:
    """Classify how the source authored its wiring: ``"fluent"``, ``"statement"``, or ``"arrow"``.

    Arrow wins if any `>>`/`.nodes(...)` appears; statement wins if wiring lives in separate
    `g.add(...)`/`g.edge(...)` statements (and isn't already chained onto the construction); else
    the graph is a single fluent chain.
    """
    _, methods = _unwind(chain)
    chain_wiring = any(name in _WIRING_METHODS for name, _ in methods)

    collector = _StatementWiring(gvar)
    module.visit(collector)
    has_rshift = any(kind in ("rshift", "nodes") for kind, _, _ in collector.ops)
    has_stmt_method = any(
        kind == "method" and name in _WIRING_METHODS for kind, name, _ in collector.ops
    )

    if has_rshift:
        return "arrow"
    if has_stmt_method and not chain_wiring:
        return "statement"
    return "fluent"


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
    """Rewrites the graph definition in place.

    Swaps the anchor statement's value for the new construction/chain, drops the folded wiring
    lines, and (for statement/arrow style) inserts the regenerated wiring lines right after the
    anchor. Removal + insertion happen at the *block* level so a builder nested in a function
    body rewrites at its own indentation.
    """

    def __init__(
        self,
        anchor: cst.Assign | cst.AnnAssign | cst.Expr,
        anchor_line: cst.SimpleStatementLine,
        wiring_lines: list[cst.SimpleStatementLine],
        new_expr: cst.BaseExpression,
        insert_lines: list[cst.BaseStatement],
    ) -> None:
        self.anchor = anchor
        self.anchor_line = anchor_line
        self.wiring = {id(line) for line in wiring_lines}
        self.new_expr = new_expr
        self.insert_lines = insert_lines

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

    def _rebuild(
        self, original_body: object, updated_body: object
    ) -> list[cst.BaseStatement] | None:
        """Drop folded wiring lines and splice the new ones in after the anchor line.

        Returns None when this block touches neither (so the block is left byte-identical).
        """
        assert isinstance(original_body, (list, tuple)) and isinstance(updated_body, (list, tuple))
        out: list[cst.BaseStatement] = []
        changed = False
        for orig, upd in zip(original_body, updated_body, strict=False):
            if id(orig) in self.wiring:
                changed = True
                continue
            out.append(upd)
            if orig is self.anchor_line and self.insert_lines:
                out.extend(self.insert_lines)
                changed = True
        return out if changed else None

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        body = self._rebuild(original_node.body, updated_node.body)
        return updated_node if body is None else updated_node.with_changes(body=body)

    def leave_IndentedBlock(
        self, original_node: cst.IndentedBlock, updated_node: cst.IndentedBlock
    ) -> cst.IndentedBlock:
        body = self._rebuild(original_node.body, updated_node.body)
        return updated_node if body is None else updated_node.with_changes(body=body)


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


def _wiring_items(ir: GraphIR) -> list[tuple[str, str, object]]:
    """Walk `ir.edges` once into ordered items every style renders from — the shared normal form.

    Yields ``("seq", source, target)`` for a plain edge and ``("cond", source, group)`` for a run
    of conditional edges sharing a source (emitted at its first edge, the rest skipped). Because
    all styles consume this same ordered list, they emit edges in identical order — which is what
    keeps the round-trip a list equality.
    """
    items: list[tuple[str, str, object]] = []
    emitted: set[str] = set()
    for edge in ir.edges:
        if edge.kind == "conditional":
            if edge.source in emitted:
                continue
            emitted.add(edge.source)
            group = [e for e in ir.edges if e.kind == "conditional" and e.source == edge.source]
            items.append(("cond", edge.source, group))
        else:
            items.append(("seq", edge.source, str(edge.target)))
    return items


def _chain_source(ir: GraphIR, tail: list[str]) -> str:
    parts = [f"Graph({ir.state})"]
    parts.extend(f".add({name})" for name in ir.added)
    for kind, source, extra in _wiring_items(ir):
        if kind == "seq":
            parts.append(f".edge({_endpoint(source)}, {_endpoint(str(extra))})")
        else:
            parts.append(f".{_conditional_call(source, _as_group(extra))}")
    parts.extend(tail)
    # Indentation is *relative*: libcst prepends the enclosing block's indent when it renders
    # this expression, so a fixed 4-space continuation lands correctly at any nesting depth.
    return "(\n    " + "\n    ".join(parts) + "\n)"


def _base_source(ir: GraphIR, tail: list[str]) -> str:
    """The bare construction `Graph(State)` (+ any preserved tail) for statement/arrow style."""
    return f"Graph({ir.state})" + "".join(tail)


def _statement_wiring(ir: GraphIR, gvar: str) -> list[str]:
    """One statement per wiring op: `g.add(A)`, `g.edge(x, y)`, `g.conditional(...)`."""
    lines = [f"{gvar}.add({name})" for name in ir.added]
    for kind, source, extra in _wiring_items(ir):
        if kind == "seq":
            lines.append(f"{gvar}.edge({_endpoint(source)}, {_endpoint(str(extra))})")
        else:
            lines.append(f"{gvar}.{_conditional_call(source, _as_group(extra))}")
    return lines


def _arrow_wiring(ir: GraphIR, gvar: str) -> list[str]:
    """Arrow style: `a, b = g.nodes(A, B)` then `>>` statements.

    Consecutive sequential edges that chain (one edge's target is the next's source) are merged
    into a single `a >> b >> c` spine. Only *adjacent* items are merged, so the emitted edge order
    still equals `ir.edges` order — the round-trip stays a list equality.
    """
    handles = _handles(ir.added)

    def ref(name: str) -> str:
        if name == START:
            return "START"
        if name == END:
            return "END"
        return handles.get(name, name)

    lines: list[str] = []
    if ir.added:
        targets = ", ".join(handles[name] for name in ir.added)
        classes = ", ".join(ir.added)
        lines.append(f"{targets} = {gvar}.nodes({classes})")

    chain: list[str] = []

    def flush() -> None:
        if chain:
            lines.append(" >> ".join(chain))
            chain.clear()

    for kind, source, extra in _wiring_items(ir):
        if kind == "seq":
            left, right = ref(source), ref(str(extra))
            if chain and chain[-1] == left:
                chain.append(right)
            else:
                flush()
                chain.extend((left, right))
        else:
            flush()
            lines.append(f"{ref(source)} >> {_router_call(_as_group(extra), ref)}")
    flush()
    return lines


def _handles(added: list[str]) -> dict[str, str]:
    """A distinct lowercase-ish handle variable per added node (`Classify` → `classify`)."""
    used: set[str] = set()
    handles: dict[str, str] = {}
    for name in added:
        base = _handle_base(name)
        handle, i = base, 2
        while handle in used:
            handle, i = f"{base}{i}", i + 1
        used.add(handle)
        handles[name] = handle
    return handles


def _handle_base(name: str) -> str:
    """A valid, readable identifier derived from a node name (lowercased first char)."""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "node"
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"n_{cleaned}"
    handle = cleaned[0].lower() + cleaned[1:]
    return f"{handle}_" if keyword.iskeyword(handle) else handle


def _as_group(extra: object) -> list[EdgeIR]:
    assert isinstance(extra, list)
    return extra


def _conditional_call(source: str, group: list[EdgeIR]) -> str:
    """A `conditional(src, fn, {...})` call (no leading dot) — shared by fluent + statement."""
    condition = group[0].condition or "..."
    mapping = {e.key: e.target for e in group if e.key is not None and e.target is not None}
    if mapping:
        pairs = ", ".join(f'"{key}": {_endpoint(target)}' for key, target in mapping.items())
        return f"conditional({_endpoint(source)}, {condition}, {{{pairs}}})"
    return f"conditional({_endpoint(source)}, {condition})"


def _router_call(group: list[EdgeIR], ref: object) -> str:
    """A `Router(fn, {...})` call for arrow style, resolving targets to handle variables."""
    assert callable(ref)
    condition = group[0].condition or "..."
    mapping = {e.key: e.target for e in group if e.key is not None and e.target is not None}
    if mapping:
        pairs = ", ".join(f'"{key}": {ref(target)}' for key, target in mapping.items())
        return f"Router({condition}, {{{pairs}}})"
    return f"Router({condition})"


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
    module = module.with_changes(body=body)
    return _ensure_import(module, "Hole", "from tensorsketch import Hole\n")


def _ensure_import(module: cst.Module, name: str, statement: str) -> cst.Module:
    """Add `statement` (a `from … import name` line) unless `name` is already bound."""
    if _binds_name(module, name):
        return module
    body = list(module.body)
    body.insert(_import_insert_index(module), cst.parse_statement(statement))
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
