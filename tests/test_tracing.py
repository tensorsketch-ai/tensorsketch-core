"""Tracing — the native, vendor-neutral spine: span trees, cost/token aggregation, errors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from tensorsketch import (
    END,
    START,
    Context,
    FakeProvider,
    Graph,
    InMemoryTracer,
    Node,
    Schema,
    Usage,
    create_agent,
    estimate_cost,
    tool,
)
from tensorsketch.messages import Message, ToolCall, assistant
from tensorsketch.observability.tracing import MODEL_KIND, NODE_KIND, TOOL_KIND
from tensorsketch.providers.base import ChatProvider, Completion
from tensorsketch.tools import Tool


class CostedProvider(ChatProvider):
    """A provider that reports a model id and token usage, so cost can be computed."""

    _model = "gpt-4o"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        return Completion(
            message=assistant(content="done"),
            usage=Usage(input_tokens=1000, output_tokens=500),
            model=self._model,
        )


async def test_agent_trace_tree_and_kinds() -> None:
    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply."""
        return a * b

    provider = FakeProvider(
        [
            assistant(tool_calls=[ToolCall(id="c1", name="multiply", args={"a": 6, "b": 7})]),
            assistant(content="42"),
        ]
    )
    tracer = InMemoryTracer()
    await create_agent(provider, tools=[multiply]).invoke({"query": "6*7?"}, tracer=tracer)

    trace = tracer.trace
    # run → node(agent) → model/tool spans
    [run] = trace.roots()
    assert run.name == "run"
    [node] = trace.children(run)
    assert node.kind == NODE_KIND
    kinds = [c.kind for c in trace.children(node)]
    assert kinds == [MODEL_KIND, TOOL_KIND, MODEL_KIND]  # in execution order
    assert trace.summary()["model_calls"] == 2
    assert trace.summary()["tool_calls"] == 1


async def test_trace_aggregates_tokens_and_cost() -> None:
    tracer = InMemoryTracer()
    await create_agent(CostedProvider()).invoke({"query": "hi"}, tracer=tracer)

    trace = tracer.trace
    assert trace.input_tokens == 1000
    assert trace.output_tokens == 500
    # gpt-4o = $2.50 / $10.00 per 1M → 1000*2.5e-6 + 500*10e-6 = 0.0075
    assert trace.cost_usd == pytest.approx(0.0075)
    model_span = trace.of_kind(MODEL_KIND)[0]
    assert model_span.attributes["gen_ai.request.model"] == "gpt-4o"


async def test_trace_records_errors() -> None:
    @tool
    def boom(x: int) -> int:
        """Always fails."""
        raise RuntimeError("kaboom")

    provider = FakeProvider(
        [
            assistant(tool_calls=[ToolCall(id="c1", name="boom", args={"x": 1})]),
            assistant(content="done"),
        ]
    )
    tracer = InMemoryTracer()
    with pytest.raises(RuntimeError, match="kaboom"):
        await create_agent(provider, tools=[boom]).invoke({"query": "x"}, tracer=tracer)

    errored = {s.kind for s in tracer.trace.errors}
    assert TOOL_KIND in errored  # the failing tool span is marked error
    assert "run" in errored  # and the error propagated up the tree


async def test_ctx_span_custom_spans_nest_under_the_node() -> None:
    class S(Schema):
        x: int = 0

    class Marky(Node):
        class In(Schema):
            x: int

        class Out(Schema):
            x: int

        async def run(self, ctx: Context, inp: In) -> Out:
            with ctx.span("parse"), ctx.span("validate"):
                pass
            return self.Out(x=inp.x)

    app = Graph(S).add(Marky).edge(START, "Marky").edge("Marky", END).compile()
    tracer = InMemoryTracer()
    await app.invoke({"x": 1}, tracer=tracer)

    names = {s.name for s in tracer.trace.spans}
    assert {"run", "Marky", "parse", "validate"} <= names
    parse = next(s for s in tracer.trace.spans if s.name == "parse")
    node = next(s for s in tracer.trace.spans if s.name == "Marky")
    assert parse.parent_id == node.span_id


def test_estimate_cost_exact_prefix_and_unknown() -> None:
    assert estimate_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
    assert estimate_cost("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.50)  # prefix match
    assert estimate_cost("mystery-model", 100, 100) is None
    assert estimate_cost(None, 1, 1) is None
