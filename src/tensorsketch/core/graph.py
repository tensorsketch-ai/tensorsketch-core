"""The `Graph` — typed nodes wired over typed state, compiled to the BSP runtime.

`Graph` is the authoring surface for Phase 0. You give it a **state Schema** (whose fields
become channels), `add` typed nodes, and wire them with `edge` (sequential) and `conditional`
(routing). `compile()` validates the whole thing — every port maps to a real state field of a
compatible type, every edge points somewhere real — and lowers it to `runtime` processes. The
compiled graph is then `invoke`-able.

State-as-ports model: a node's `In` fields are the state channels it reads; its `Out` fields
are the channels it writes. This keeps the type story honest — an edge is legal because the
node's declared ports line up with the shared typed state, checked before anything runs.

The ergonomic `>>` / `Router.on(...)` wiring surface from the design doc, plus the code⇄canvas
extractor, build on top of this explicit API in Phase 1; this is the foundation they compile to.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ..runtime.checkpoint import Backend, Checkpoint, save_checkpoint
from ..runtime.checkpoint import apply as apply_snapshot
from ..runtime.engine import Process, execute
from ..runtime.streaming import EmitFn, Event, event_stream
from .channels import Channel, LastValue, Reducer, Topic, channel_for_field
from .context import Context
from .errors import GraphError, NodeError
from .node import Node
from .schema import Schema
from .send import Send

if TYPE_CHECKING:
    from ..observability.tracing import Tracer
    from .wiring import NodeHandle

#: Wire an entry with `edge(START, node)`; mark a terminal branch with `edge(node, END)`.
START = "__start__"
END = "__end__"

#: The graph's state type. Generic so `compile()` and `invoke()` preserve the concrete state,
#: giving callers full type inference on results (`out.answer` instead of `Schema`).
StateT = TypeVar("StateT", bound=Schema)


@dataclass(frozen=True)
class _Branch:
    """A conditional edge. `path(state)` returns a node name, a `Send`, or a list mixing the two
    (plus `END` to stop) — names route sequentially, `Send`s fan out with per-instance payloads."""

    path: Callable[[Any], Any]
    mapping: dict[str, str] | None


@dataclass
class _Prep:
    """The prepared starting point for a run (fresh or resumed), shared by invoke and stream."""

    channels: dict[str, Channel[Any, Any]]
    active: set[str]
    sends: list[tuple[str, dict[str, Any]]]
    start_step: int
    parent_id: str | None
    run_id: str
    thread_id: str


class Graph(Generic[StateT]):
    """A builder for a typed, compilable agent graph.

    Every mutating method returns `self`, so wiring reads fluently::

        g = (Graph(State)
             .add(Classify).add(Billing).add(Tech)
             .edge(START, "Classify")
             .conditional("Classify", route, {"billing": "Billing", "tech": "Tech"}))
        app = g.compile()
    """

    def __init__(self, state: type[StateT]) -> None:
        self.state: type[StateT] = state
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, set[str]] = defaultdict(set)
        self._branches: dict[str, _Branch] = {}
        self._entry: str | None = None

    # -- authoring -----------------------------------------------------------------------

    def add(self, node: type[Node] | Node, name: str | None = None) -> Graph[StateT]:
        """Add a node (a `Node` subclass or instance). Name defaults to the class name."""
        instance = node() if isinstance(node, type) else node
        node_name = name or instance.name
        if node_name in self._nodes:
            raise GraphError(f"a node named {node_name!r} is already in the graph")
        self._nodes[node_name] = instance
        return self

    def nodes(self, *node_types: type[Node] | Node) -> tuple[NodeHandle, ...]:
        """Add several nodes and return a `NodeHandle` for each — the entry to `>>` wiring::

            classify, billing = g.nodes(Classify, Billing)
            START >> classify >> Router(route, billing=billing)

        Each handle is bound to this graph; wiring through `>>` calls the same `.edge`/
        `.conditional` methods, so the compiled graph is identical to the fluent form.
        """
        from .wiring import NodeHandle

        handles: list[NodeHandle] = []
        for node in node_types:
            instance = node() if isinstance(node, type) else node
            self.add(instance)
            handles.append(NodeHandle(self, instance.name))
        return tuple(handles)

    def __getitem__(self, name: str) -> NodeHandle:
        """A `NodeHandle` for an already-added node, so you can wire it with `>>`."""
        from .wiring import NodeHandle

        if name not in self._nodes:
            raise GraphError(f"no node named {name!r} in the graph")
        return NodeHandle(self, name)

    def entry(self, name: str) -> Graph[StateT]:
        """Set the graph's entry node (equivalent to `edge(START, name)`)."""
        self._entry = name
        return self

    def edge(self, src: str, dst: str) -> Graph[StateT]:
        """Add a sequential edge. `edge(START, x)` sets the entry; `edge(x, END)` terminates x."""
        if src == START:
            self._entry = dst
            return self
        self._edges[src].add(dst)
        return self

    def conditional(
        self,
        src: str,
        path: Callable[[Any], str | list[str]],
        mapping: dict[str, str] | None = None,
    ) -> Graph[StateT]:
        """Route out of `src` dynamically.

        `path` is called with the post-step state (as a state instance) and returns the next
        node name, a list of names (parallel fan-out), a `Send` / list of `Send`s (dynamic
        fan-out with per-instance payloads), or — when `mapping` is given — key(s) looked up in
        `mapping`. Return `END` to stop that branch.
        """
        self._branches[src] = _Branch(path, mapping)
        return self

    def router(
        self,
        src: str,
        path: Callable[[Any], Any],
        mapping: dict[str, str] | None = None,
    ) -> Graph[StateT]:
        """Route dynamically out of `src` — the intent-named form of `conditional`.

        Use it for one-of-N branching (return a name, or a key looked up in `mapping`) or for
        **dynamic fan-out**: return a list of `Send`s and the engine schedules one instance of
        each target with its own payload, all merging at the next barrier.

            g.router("split", lambda s: [Send("worker", {"item": x}) for x in s.items])
        """
        return self.conditional(src, path, mapping)

    def loop(
        self,
        node: str,
        until: Callable[[Any], bool],
        *,
        exit: str = END,
    ) -> Graph[StateT]:
        """Repeat `node` until `until(state)` holds, then continue to `exit` (default `END`).

        Sugar over a self-`conditional`: after `node` runs, route to `exit` if `until` is true,
        else back to `node`. Wire the entry separately (`edge(START, node)`); `loop` adds only the
        repeat/exit branch. The `invoke(max_steps=...)` recursion limit still bounds a runaway loop.
        """
        return self.conditional(node, lambda state: exit if until(state) else node)

    # -- compilation ---------------------------------------------------------------------

    def compile(self) -> CompiledGraph[StateT]:
        """Validate the graph and lower it to a runnable `CompiledGraph`.

        Raises `GraphError` for any structural problem: no entry, an edge to an unknown node, a
        port that names a missing state field, a port whose type is incompatible with its
        channel, or a node given both static and conditional successors.
        """
        self._validate()
        processes: dict[str, Process] = {}
        for node_name, node in self._nodes.items():
            reads = node.In.field_names()
            writes = node.Out.field_names()
            branch = self._branches.get(node_name)
            static = tuple(self._edges.get(node_name, ()))
            processes[node_name] = Process(
                name=node_name,
                reads=reads,
                run=_make_run(node_name, node, writes),
                successors=_make_successors(self.state, static, branch),
            )
        assert self._entry is not None  # guaranteed by _validate
        return CompiledGraph(self.state, processes, self._entry)

    def _validate(self) -> None:
        state_fields = self.state.model_fields
        if self._entry is None:
            raise GraphError("graph has no entry; call .entry(name) or .edge(START, name)")
        if self._entry not in self._nodes:
            raise GraphError(f"entry node {self._entry!r} was never added to the graph")

        for node_name, node in self._nodes.items():
            _check_ports(node_name, node, self.state, state_fields)

        for src, dsts in self._edges.items():
            if src not in self._nodes:
                raise GraphError(f"edge source {src!r} is not a node in the graph")
            if src in self._branches:
                raise GraphError(
                    f"node {src!r} has both static edges and a conditional edge; use one"
                )
            for dst in dsts:
                if dst != END and dst not in self._nodes:
                    raise GraphError(f"edge target {dst!r} (from {src!r}) is not a node")

        for src, branch in self._branches.items():
            if src not in self._nodes:
                raise GraphError(f"conditional source {src!r} is not a node in the graph")
            if branch.mapping:
                for target in branch.mapping.values():
                    if target != END and target not in self._nodes:
                        raise GraphError(
                            f"conditional target {target!r} (from {src!r}) is not a node"
                        )


class CompiledGraph(Generic[StateT]):
    """A validated, runnable graph. Create one via `Graph.compile()`."""

    def __init__(self, state: type[StateT], processes: Mapping[str, Process], entry: str) -> None:
        self._state = state
        self._processes = processes
        self._entry = entry

    async def invoke(
        self,
        input: Schema | Mapping[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        backend: Backend | None = None,
        run_id: str | None = None,
        max_steps: int = 25,
        tracer: Tracer | None = None,
    ) -> StateT:
        """Run the graph to completion and return the final state.

        Without a `backend`, this is a plain in-memory run: `input` seeds the state channels
        (defaults first, then your input), the graph runs to quiescence, and the final state is
        returned.

        With a `backend` and a `thread_id`, the run is **durable**. A checkpoint is written at
        every superstep barrier and each `ctx.step(...)` effect is journaled. If a checkpoint
        already exists for `thread_id`, the run **resumes** from it — effects already recorded
        are replayed from the journal rather than re-executed — and `input`, if given, is applied
        as additional writes before continuing.

        Raises:
            GraphError: A backend was supplied without a `thread_id`, or `input` names a field
                that isn't part of the state.
            GraphRecursionError: The graph did not settle within `max_steps`.
        """
        prep = self._prepare(input, thread_id, backend, run_id)
        await execute(
            self._processes,
            prep.channels,
            prep.active,
            run_id=prep.run_id,
            thread_id=prep.thread_id,
            backend=backend,
            start_step=prep.start_step,
            parent_id=prep.parent_id,
            sends=prep.sends,
            max_steps=max_steps,
            tracer=tracer,
        )
        return self._read_state(prep.channels)

    async def stream(
        self,
        input: Schema | Mapping[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        backend: Backend | None = None,
        run_id: str | None = None,
        max_steps: int = 25,
        buffer: int = 256,
        tracer: Tracer | None = None,
    ) -> AsyncIterator[Event]:
        """Run the graph and yield a live `Event` stream as it progresses.

        Same seeding/durability/resume semantics as `invoke`, but instead of only returning the
        final state, it yields namespaced events: `run_start`, `node_start`, `node_end`,
        `values` (merged state after each barrier), `run_end`, plus any custom events a node
        emits via `ctx.emit`. Each event carries a monotonic `seq` cursor. A slow consumer
        applies backpressure (bounded by `buffer`). With a `backend`, events are persisted for
        `replay`.
        """
        prep = self._prepare(input, thread_id, backend, run_id)

        async def runner(emit: EmitFn) -> None:
            await execute(
                self._processes,
                prep.channels,
                prep.active,
                run_id=prep.run_id,
                thread_id=prep.thread_id,
                backend=backend,
                start_step=prep.start_step,
                parent_id=prep.parent_id,
                sends=prep.sends,
                max_steps=max_steps,
                emit=emit,
                tracer=tracer,
            )

        append = backend.append_event if backend is not None else None
        async for event in event_stream(
            runner,
            run_id=prep.run_id,
            thread_id=prep.thread_id,
            append=append,
            buffer=buffer,
        ):
            yield event

    async def replay(
        self, thread_id: str, backend: Backend, *, since: int = 0
    ) -> AsyncIterator[Event]:
        """Replay a durable run's persisted events from a cursor (`seq >= since`).

        Lets a dropped consumer catch up on what a run has already emitted. Yields in order.
        """
        for event in backend.read_events(thread_id, since):
            yield event

    def get_state(self, thread_id: str, backend: Backend) -> StateT | None:
        """The latest checkpointed state for a thread, or None if it has never run."""
        checkpoint = backend.latest_checkpoint(thread_id)
        if checkpoint is None:
            return None
        channels = self._build_channels()
        apply_snapshot(channels, checkpoint.channel_values)
        return self._read_state(channels)

    def get_history(self, thread_id: str, backend: Backend) -> list[Checkpoint]:
        """Every checkpoint for a thread, oldest first (the resume/fork timeline)."""
        return backend.list_checkpoints(thread_id)

    async def fork(
        self,
        backend: Backend,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: str,
        input: Schema | Mapping[str, Any] | None = None,
        *,
        run_id: str | None = None,
        max_steps: int = 25,
    ) -> StateT:
        """Branch a new run from a past checkpoint and run it to completion.

        The new thread starts with the checkpoint's state (plus any `input`) and a *fresh*
        journal — so effects run anew down the new branch. Ideal for "what if I'd routed
        differently here" exploration.
        """
        source = backend.get_checkpoint(source_thread_id, checkpoint_id)
        if source is None:
            raise GraphError(
                f"checkpoint {checkpoint_id!r} not found for thread {source_thread_id!r}"
            )
        channels = self._build_channels()
        apply_snapshot(channels, source.channel_values)
        self._apply_input(channels, input)
        forked_sends = [(node, dict(payload)) for node, payload in source.sends]
        parent_id = save_checkpoint(
            backend, new_thread_id, source.id, source.step, channels, source.active, forked_sends
        )
        await execute(
            self._processes,
            channels,
            set(source.active),
            run_id=run_id or uuid.uuid4().hex,
            thread_id=new_thread_id,
            backend=backend,
            start_step=source.step,
            parent_id=parent_id,
            sends=list(forked_sends),
            max_steps=max_steps,
        )
        return self._read_state(channels)

    def _prepare(
        self,
        input: Schema | Mapping[str, Any] | None,
        thread_id: str | None,
        backend: Backend | None,
        run_id: str | None,
    ) -> _Prep:
        """Build channels and decide start state — a fresh run or a resume — for invoke/stream."""
        if backend is not None and thread_id is None:
            raise GraphError("pass a thread_id alongside a backend so the run can be resumed")
        tid = thread_id or ""
        channels = self._build_channels()
        resumed = backend.latest_checkpoint(tid) if backend is not None else None
        if resumed is None:
            self._seed_defaults(channels)
            self._apply_input(channels, input)
            active: set[str] = {self._entry}
            sends: list[tuple[str, dict[str, Any]]] = []
            start_step = 0
            parent_id: str | None = None
            if backend is not None:
                parent_id = save_checkpoint(backend, tid, None, 0, channels, active)
        else:
            apply_snapshot(channels, resumed.channel_values)
            self._apply_input(channels, input)
            active = set(resumed.active)
            sends = [(node, dict(payload)) for node, payload in resumed.sends]
            start_step = resumed.step
            parent_id = resumed.id
        return _Prep(
            channels=channels,
            active=active,
            sends=sends,
            start_step=start_step,
            parent_id=parent_id,
            run_id=run_id or uuid.uuid4().hex,
            thread_id=tid,
        )

    def _build_channels(self) -> dict[str, Channel[Any, Any]]:
        return {name: channel_for_field(f) for name, f in self._state.model_fields.items()}

    def _read_state(self, channels: Mapping[str, Channel[Any, Any]]) -> StateT:
        result = {name: ch.get() for name, ch in channels.items() if ch.is_set}
        return self._state(**result)

    def _seed_defaults(self, channels: Mapping[str, Channel[Any, Any]]) -> None:
        # Initialize LastValue channels from their field defaults so counters/optionals behave.
        for name, field in self._state.model_fields.items():
            if isinstance(channels[name], LastValue):
                has_default, value = _field_default(field)
                if has_default:
                    channels[name].update([value])

    def _apply_input(
        self,
        channels: Mapping[str, Channel[Any, Any]],
        input: Schema | Mapping[str, Any] | None,
    ) -> None:
        if input is None:
            return
        data = input.model_dump() if isinstance(input, Schema) else dict(input)
        for key, value in data.items():
            if key not in channels:
                raise GraphError(f"input field {key!r} is not a field of {self._state.__name__}")
            channels[key].update([value])


# -- process construction (module-level so closures don't capture the builder) --------------


def _make_run(node_name: str, node: Node, writes: tuple[str, ...]) -> Any:
    async def run(ctx: Context, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            inp = node.In(**inputs)
        except ValidationError as exc:
            raise NodeError(
                f"node {node_name!r} could not build its input from state: {exc}"
            ) from exc
        out = await node.run(ctx, inp)
        return {field: getattr(out, field) for field in writes}

    return run


def _make_successors(
    state: type[Schema], static: tuple[str, ...], branch: _Branch | None
) -> Callable[[Mapping[str, Any]], Iterable[str | Send]]:
    def successors(post: Mapping[str, Any]) -> Iterable[str | Send]:
        if branch is None:
            return [n for n in static if n != END]
        view = state.model_construct(**dict(post))
        result = branch.path(view)
        items: list[Any] = result if isinstance(result, list) else [result]
        out: list[str | Send] = []
        for item in items:
            if isinstance(item, Send):
                out.append(item)  # dynamic fan-out — one instance, its own payload
                continue
            name: str = branch.mapping.get(item, item) if branch.mapping is not None else item
            if name != END:
                out.append(name)
        return out

    return successors


# -- validation helpers ---------------------------------------------------------------------


def _check_ports(
    node_name: str, node: Node, state: type[Schema], state_fields: Mapping[str, FieldInfo]
) -> None:
    for field in node.In.field_names():
        if field not in state_fields:
            raise GraphError(
                f"node {node_name!r} reads state field {field!r}, which does not exist on "
                f"{state.__name__}"
            )
        if not _reducer_field(state_fields[field]) and not _assignable(
            state.field_type(field), node.In.field_type(field)
        ):
            raise GraphError(
                f"node {node_name!r} reads {field!r} as {node.In.field_type(field)!r}, but the "
                f"state channel holds {state.field_type(field)!r}"
            )
    for field in node.Out.field_names():
        if field not in state_fields:
            raise GraphError(
                f"node {node_name!r} writes state field {field!r}, which does not exist on "
                f"{state.__name__}"
            )
        if not _reducer_field(state_fields[field]) and not _assignable(
            node.Out.field_type(field), state.field_type(field)
        ):
            raise GraphError(
                f"node {node_name!r} writes {field!r} as {node.Out.field_type(field)!r}, but the "
                f"state channel holds {state.field_type(field)!r}"
            )


def _reducer_field(field: FieldInfo) -> bool:
    # Reducer/Topic channels have an update type that differs from their value type, so we skip
    # the plain assignability check for them in Phase 0 (richer port typing comes later).
    return any(isinstance(m, (Reducer, Topic)) for m in field.metadata)


def _assignable(src: Any, dst: Any) -> bool:
    if src is dst or src is Any or dst is Any:
        return True
    if isinstance(src, type) and isinstance(dst, type):
        return issubclass(src, dst)
    # Generics, unions, literals: don't block in Phase 0 — too easy to reject a valid graph.
    return True


def _field_default(field: FieldInfo) -> tuple[bool, Any]:
    if field.default_factory is not None:
        return True, field.default_factory()  # type: ignore[call-arg]
    if field.default is not PydanticUndefined:
        return True, field.default
    return False, None
