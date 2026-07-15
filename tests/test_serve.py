"""Serving layer: OpenAI-compatible, A2A (expose + consume), and AG-UI, over the real ASGI apps."""

from __future__ import annotations

import json

import httpx
from starlette.testclient import TestClient

from tensorsketch import create_agent
from tensorsketch.agents.agent import AgentState
from tensorsketch.core.graph import CompiledGraph
from tensorsketch.messages import Message, assistant
from tensorsketch.providers.fake import FakeProvider
from tensorsketch.serve import a2a_app, a2a_tool, agui_app, openai_app

REPLY = "hello from tensorsketch"


def make_agent(reply: str = REPLY) -> CompiledGraph[AgentState]:
    # A policy provider always returns the same reply, so an app can serve many requests.
    def policy(messages: list[Message], tools: object) -> Message:
        return assistant(reply)

    return create_agent(FakeProvider(policy=policy))


def sse_datas(text: str) -> list[str]:
    return [
        line[len("data: ") :]
        for block in text.strip().split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]


# --- OpenAI-compatible ----------------------------------------------------------------------


def test_openai_non_streaming() -> None:
    client = TestClient(openai_app(make_agent(), model="my-bot"))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "my-bot", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "my-bot"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": REPLY}
    assert body["choices"][0]["finish_reason"] == "stop"


def test_openai_streaming() -> None:
    client = TestClient(openai_app(make_agent()))
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    datas = sse_datas(resp.text)
    assert datas[-1] == "[DONE]"

    content = ""
    for data in datas[:-1]:
        chunk = json.loads(data)
        assert chunk["object"] == "chat.completion.chunk"
        content += chunk["choices"][0]["delta"].get("content", "")
    assert content == REPLY
    assert json.loads(datas[-2])["choices"][0]["finish_reason"] == "stop"


def test_openai_list_models() -> None:
    client = TestClient(openai_app(make_agent(), model="my-bot"))
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "my-bot"


# --- A2A: expose ----------------------------------------------------------------------------


def test_a2a_agent_card() -> None:
    client = TestClient(a2a_app(make_agent(), name="my-bot"))
    for path in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
        card = client.get(path).json()
        assert card["name"] == "my-bot"
        assert card["capabilities"]["streaming"] is True
        assert card["skills"]


def test_a2a_message_send() -> None:
    client = TestClient(a2a_app(make_agent()))
    body = client.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}},
        },
    ).json()
    assert body["id"] == "1"
    task = body["result"]
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"][0]["text"] == REPLY


def test_a2a_message_stream() -> None:
    client = TestClient(a2a_app(make_agent()))
    resp = client.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/stream",
            "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}},
        },
    )
    results = [json.loads(d)["result"] for d in sse_datas(resp.text)]
    kinds = [r["kind"] for r in results]
    assert kinds == ["status-update", "artifact-update", "status-update"]
    assert results[0]["status"]["state"] == "working"
    assert results[1]["artifact"]["parts"][0]["text"] == REPLY
    assert results[-1]["status"]["state"] == "completed" and results[-1]["final"] is True


def test_a2a_unknown_method() -> None:
    client = TestClient(a2a_app(make_agent()))
    body = client.post("/", json={"jsonrpc": "2.0", "id": "1", "method": "bogus"}).json()
    assert body["error"]["code"] == -32601


# --- A2A: consume (round-trip against the expose side, in-process) ---------------------------


async def test_a2a_tool_round_trip() -> None:
    server = a2a_app(make_agent(reply="pong"))
    transport = httpx.ASGITransport(app=server)
    async with httpx.AsyncClient(transport=transport, base_url="http://a2a") as client:
        tool = a2a_tool("http://a2a/", client=client)
        result = await tool.run({"message": "ping"})
    assert result == "pong"


# --- AG-UI ----------------------------------------------------------------------------------


def test_agui_event_stream() -> None:
    client = TestClient(agui_app(make_agent()))
    resp = client.post(
        "/",
        json={"threadId": "t1", "runId": "r1", "messages": [{"role": "user", "content": "hi"}]},
    )
    events = [json.loads(d) for d in sse_datas(resp.text)]
    types = [e["type"] for e in events]

    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert "STATE_SNAPSHOT" in types
    assert events[0]["threadId"] == "t1" and events[0]["runId"] == "r1"

    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == REPLY
    snapshot = next(e["snapshot"] for e in events if e["type"] == "STATE_SNAPSHOT")
    assert snapshot["output"] == REPLY
