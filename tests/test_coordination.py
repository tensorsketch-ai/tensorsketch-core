"""Multi-agent coordination: agents-as-tools (supervisor/handoff) via `as_tool`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import pytest

from tensorsketch import (
    END,
    START,
    AgentState,
    Context,
    FakeProvider,
    Graph,
    InMemoryBackend,
    InMemoryTracer,
    Message,
    Node,
    Schema,
    Span,
    as_tool,
    create_agent,
    tool,
)
from tensorsketch.core.graph import CompiledGraph
from tensorsketch.messages import ToolCall, assistant

_Policy = Callable[[list[Message], "Sequence[object] | None"], Message]


def _specialist(answer: str, name: str) -> tuple[CompiledGraph[AgentState], FakeProvider]:
    provider = FakeProvider(script=[assistant(answer)])
    return create_agent(provider, name=name), provider


def _delegating_policy(target: str, args: dict[str, object]) -> _Policy:
    """A supervisor policy: delegate to `target` once, then echo its answer as the final reply."""

    def policy(convo: list[Message], tools: Sequence[object] | None) -> Message:
        if not any(m.role == "tool" for m in convo):
            return assistant(tool_calls=[ToolCall(id="c1", name=target, args=args)])
        answer = [m for m in convo if m.role == "tool"][-1].content
        return assistant(content=f"[via {target}] {answer}")

    return policy


async def test_supervisor_delegates_to_a_specialist() -> None:
    billing, billing_provider = _specialist("Refund issued: $42.", "billing")
    tech, _ = _specialist("Try restarting.", "tech")

    supervisor = create_agent(
        FakeProvider(policy=_delegating_policy("billing", {"request": "refund order 7"})),
        tools=[
            as_tool(billing, name="billing", description="Billing questions."),
            as_tool(tech, name="tech", description="Technical problems."),
        ],
        name="supervisor",
    )

    state = await supervisor.invoke({"query": "I want a refund for order 7"})
    assert state.output == "[via billing] Refund issued: $42."
    assert len(billing_provider.calls) == 1  # the billing specialist ran


def test_as_tool_advertises_a_single_string_arg() -> None:
    agent, _ = _specialist("hi", "helper")
    t = as_tool(agent, name="helper", description="A helper agent.")
    schema = t.json_schema()
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["request"]
    assert schema["properties"]["request"]["type"] == "string"
    assert t.description == "A helper agent."


async def test_specialist_runs_under_the_supervisor_trace() -> None:
    billing, _ = _specialist("Refund issued.", "billing")
    supervisor = create_agent(
        FakeProvider(policy=_delegating_policy("billing", {"request": "refund"})),
        tools=[as_tool(billing, name="billing", description="Billing.")],
        name="supervisor",
    )
    tracer = InMemoryTracer()
    await supervisor.invoke({"query": "refund please"}, tracer=tracer)

    spans = tracer.trace.spans
    by_id = {s.span_id: s for s in spans}

    def ancestors(span: Span) -> Iterator[Span]:
        current = by_id.get(span.parent_id) if span.parent_id else None
        while current is not None:
            yield current
            current = by_id.get(current.parent_id) if current.parent_id else None

    # The specialist's model call was recorded on the *same* trace, nested under the delegating
    # tool call — so one trace shows the whole team.
    assert any(s.kind == "tool" for s in spans)
    model_spans = [s for s in spans if s.kind == "model"]
    assert any(any(a.kind == "tool" for a in ancestors(m)) for m in model_spans)


async def test_delegated_agent_runs_once_across_a_crash() -> None:
    runs = {"n": 0}

    def specialist_policy(convo: list[Message], tools: Sequence[object] | None) -> Message:
        runs["n"] += 1
        return assistant(content="Refund issued.")

    billing = create_agent(FakeProvider(policy=specialist_policy), name="billing")

    final_attempts = {"n": 0}

    def supervisor_policy(convo: list[Message], tools: Sequence[object] | None) -> Message:
        if any(m.role == "tool" for m in convo):
            final_attempts["n"] += 1
            if final_attempts["n"] == 1:
                raise RuntimeError("crash before composing the final answer")
            return assistant(content="done")
        return assistant(tool_calls=[ToolCall(id="c1", name="billing", args={"request": "refund"})])

    backend = InMemoryBackend()
    supervisor = create_agent(
        FakeProvider(policy=supervisor_policy),
        tools=[as_tool(billing, name="billing", description="Billing.")],
        name="supervisor",
    )

    with pytest.raises(RuntimeError):
        await supervisor.invoke({"query": "refund"}, thread_id="s1", backend=backend)
    assert runs["n"] == 1  # the specialist ran once, before the crash

    out = await supervisor.invoke(thread_id="s1", backend=backend)  # resume
    assert out.output == "done"
    assert runs["n"] == 1  # still once — the delegated call was replayed from the journal


async def test_output_and_input_keys_can_be_overridden() -> None:
    class Doc(Schema):
        topic: str
        draft: str = ""

    class Writer(Node):
        class In(Schema):
            topic: str

        class Out(Schema):
            draft: str

        async def run(self, ctx: Context, inp: In) -> Out:
            return self.Out(draft=f"draft about {inp.topic}")

    writer = Graph(Doc).add(Writer).edge(START, "Writer").edge("Writer", END).compile()
    t = as_tool(writer, name="writer", description="Write.", input_key="topic", output_key="draft")

    ctx = Context(run_id="r", thread_id="", superstep=0, node="n")
    result = await t.run({"request": "cats"}, ctx)
    assert result == "draft about cats"


async def test_ctx_is_injected_into_a_tool_that_declares_it() -> None:
    @tool
    def uses_ctx(ctx: Context, x: int) -> str:
        """A tool that needs the run context."""
        return f"{ctx.node}:{x}"

    # `ctx` is never advertised to the model…
    assert list(uses_ctx.json_schema()["properties"]) == ["x"]
    # …but it's injected at call time.
    ctx = Context(run_id="r", thread_id="", superstep=0, node="here")
    assert await uses_ctx.run({"x": 7}, ctx) == "here:7"
