"""The Studio live overlay pipeline — an agent's spans reach the bridge's ephemeral trace buffer.

`http_span_sink` (in the agent's process) POSTs each finished span to the bridge; the bridge buffers
them in memory and the browser polls `GET /api/trace`. Here we drive the whole path minus the JS.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tensorsketch import FakeProvider, InMemoryTracer, MultiTracer, Usage, create_agent, tool
from tensorsketch.canvas.server import TraceBuffer, make_handler
from tensorsketch.messages import ToolCall, assistant
from tensorsketch.observability.export import http_span_sink


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


def test_trace_buffer_returns_only_fresh_records() -> None:
    buf = TraceBuffer()
    assert buf.add({"name": "a"}) == 1
    assert buf.add({"name": "b"}) == 2

    out = buf.since(0)
    assert out["cursor"] == 2
    assert [r["span"]["name"] for r in out["records"]] == ["a", "b"]
    # a cursor only ever yields newer records — that's how the browser tails without re-fetching
    assert [r["span"]["name"] for r in buf.since(1)["records"]] == ["b"]
    assert buf.since(2)["records"] == []


def test_trace_buffer_is_bounded() -> None:
    buf = TraceBuffer(limit=3)
    for i in range(5):
        buf.add({"name": str(i)})
    out = buf.since(0)
    assert out["cursor"] == 5  # the seq keeps counting
    assert [r["span"]["name"] for r in out["records"]] == ["2", "3", "4"]  # only the last 3 kept


async def test_live_overlay_pipeline_delivers_run_spans(tmp_path: Path) -> None:
    buffer = TraceBuffer()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path / "graph.py", buffer))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        sink_url = f"http://127.0.0.1:{port}/api/trace"

        # exactly how a user wires the overlay: fan the trace to memory *and* the Studio bridge
        tracer = MultiTracer(InMemoryTracer(), http_span_sink(sink_url))
        await create_agent(_provider(), tools=[echo]).invoke({"query": "x"}, tracer=tracer)

        # the sink delivers on a background thread — poll the buffer until the run lands
        deadline = time.time() + 5.0
        records: list[Any] = []
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/trace?since=0") as resp:
                records = json.load(resp)["records"]
            names = {r["span"]["name"] for r in records}
            if {"run", "agent", "model_call", "tool_call"} <= names:
                break
            time.sleep(0.05)

        names = {r["span"]["name"] for r in records}
        assert {"run", "agent", "model_call", "tool_call"} <= names

        # the node span carries the `tensorsketch.node` attribute the overlay keys off, and the
        # model span
        # carries cost — so the badge can show which node ran, how long, and for how much
        node = next(r["span"] for r in records if r["span"]["kind"] == "node")
        assert node["attributes"]["tensorsketch.node"] == "agent"
        model = next(r["span"] for r in records if r["span"]["kind"] == "model")
        assert model["attributes"]["gen_ai.usage.cost_usd"] > 0
    finally:
        server.shutdown()
