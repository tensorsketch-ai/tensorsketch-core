"""Trace exporters — the JSON-lines file tracer and the OpenTelemetry bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorsketch import (
    FakeProvider,
    FileTracer,
    InMemoryTracer,
    MultiTracer,
    Usage,
    create_agent,
    tool,
)
from tensorsketch.messages import ToolCall, assistant
from tensorsketch.observability.tracing import Span


@tool
def echo(text: str) -> str:
    """Echo text."""
    return text


def _provider() -> FakeProvider:
    return FakeProvider(
        [
            assistant(tool_calls=[ToolCall(id="c1", name="echo", args={"text": "hi"})]),
            assistant(content="done"),
        ],
        model="gpt-4o-mini",
        usage=Usage(input_tokens=100, output_tokens=50),
    )


async def test_file_tracer_writes_one_json_span_per_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with FileTracer(path) as tracer:
        await create_agent(_provider(), tools=[echo]).invoke({"query": "x"}, tracer=tracer)

    lines = [json.loads(raw) for raw in path.read_text().splitlines()]
    names = {line["name"] for line in lines}
    assert {"run", "agent", "model_call", "tool_call"} <= names

    for line in lines:  # every span is a complete, valid record
        assert line["status"] == "ok"
        assert line["duration_ms"] >= 0
        assert "attributes" in line and "span_id" in line

    model = next(line for line in lines if line["name"] == "model_call")
    assert model["attributes"]["gen_ai.request.model"] == "gpt-4o-mini"
    assert model["attributes"]["gen_ai.usage.cost_usd"] > 0


async def test_multi_tracer_fans_out_to_every_sink(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    collector = InMemoryTracer()
    live: list[Span] = []  # a callable sink — the shape the Studio overlay plugs into

    with FileTracer(path) as file_tracer:
        tracer = MultiTracer(collector, file_tracer, live.append)
        await create_agent(_provider(), tools=[echo]).invoke({"query": "x"}, tracer=tracer)

    file_names = {json.loads(raw)["name"] for raw in path.read_text().splitlines()}
    mem_names = {s.name for s in collector.trace.spans}
    live_names = {s.name for s in live}

    # every sink saw the identical span tree, from one lifecycle
    assert {"run", "agent", "model_call", "tool_call"} <= file_names
    assert file_names == mem_names == live_names
    assert len(collector.trace.spans) == len(live) == len(path.read_text().splitlines())

    # the collected Trace is coherent: one shared trace_id, aggregates populated off the same spans
    assert collector.trace.trace_id == tracer.trace_id
    assert {s.trace_id for s in collector.trace.spans} == {tracer.trace_id}
    assert collector.trace.cost_usd > 0
    assert collector.trace.summary()["tool_calls"] == 1


async def test_otel_tracer_bridges_to_opentelemetry() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from tensorsketch.observability.otel import OTelTracer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider.get_tracer("test"))

    await create_agent(_provider(), tools=[echo]).invoke({"query": "x"}, tracer=tracer)

    finished = exporter.get_finished_spans()
    names = {s.name for s in finished}
    assert {"run", "agent", "model_call", "tool_call"} <= names

    model = next(s for s in finished if s.name == "model_call")
    assert model.attributes.get("gen_ai.request.model") == "gpt-4o-mini"
    assert model.attributes.get("tensorsketch.kind") == "model"

    # nesting is preserved through OTel's own context: model_call sits under the agent node span
    agent_span = next(s for s in finished if s.name == "agent")
    assert model.parent is not None
    assert model.parent.span_id == agent_span.context.span_id
