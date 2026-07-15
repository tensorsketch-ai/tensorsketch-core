"""A2A (Agent2Agent) interop — expose a TensorSketch agent, and consume remote ones.

`a2a_app(agent)` returns a Starlette app that publishes an **Agent Card** (capability discovery at
`/.well-known/agent.json`) and answers A2A's JSON-RPC methods `message/send` (one-shot) and
`message/stream` (SSE task updates). `a2a_tool(url)` is the other direction: a TensorSketch `Tool`
that
sends a message to a remote A2A agent and returns its reply, so one agent can call another across
frameworks.

This is a pragmatic, current-shaped subset of the protocol — agent card + message send/stream with
a completed-task result — not the full task store / push-notification surface. See the serving
guide for what's deferred.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..core.graph import CompiledGraph
from ..messages import user
from ..tools import Tool
from ._adapter import ChatAdapter, ToInput, ToReply, last_user_query, reply_text
from ._sse import sse_event, sse_response

PROTOCOL_VERSION = "0.2.5"


@dataclass
class AgentCard:
    """The A2A capability descriptor served at `/.well-known/agent.json`."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    url: str = "/"
    skills: list[dict[str, Any]] | None = None
    streaming: bool = True

    def to_dict(self) -> dict[str, Any]:
        skills = self.skills or [
            {
                "id": "chat",
                "name": self.name,
                "description": self.description or self.name,
                "tags": ["chat"],
            }
        ]
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "capabilities": {
                "streaming": self.streaming,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": skills,
        }


def _text_from_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))


def _message_text(params: dict[str, Any]) -> str:
    return _text_from_parts(params.get("message", {}).get("parts", []))


def _text_part(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def _completed_task(reply: str) -> dict[str, Any]:
    return {
        "id": uuid4().hex,
        "contextId": uuid4().hex,
        "kind": "task",
        "status": {"state": "completed"},
        "artifacts": [{"artifactId": uuid4().hex, "name": "reply", "parts": [_text_part(reply)]}],
        "history": [],
    }


def _rpc_result(rpc_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _rpc_error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def a2a_app(
    graph: CompiledGraph[Any],
    *,
    card: AgentCard | None = None,
    name: str = "tensorsketch-agent",
    to_input: ToInput | None = None,
    to_reply: ToReply | None = None,
) -> Starlette:
    """Build an A2A ASGI app for `graph`. Pass an `AgentCard` to customize discovery metadata."""
    adapter = ChatAdapter(
        graph, to_input=to_input or last_user_query, to_reply=to_reply or reply_text
    )
    agent_card = card or AgentCard(name=name)

    async def get_card(request: Request) -> Response:
        return JSONResponse(agent_card.to_dict())

    async def rpc(request: Request) -> Response:
        body = await request.json()
        rpc_id, method = body.get("id"), body.get("method")
        params = body.get("params", {})
        if method == "message/send":
            reply = await adapter.reply([user(_message_text(params))])
            return JSONResponse(_rpc_result(rpc_id, _completed_task(reply)))
        if method == "message/stream":
            return _stream(adapter, rpc_id, _message_text(params))
        return JSONResponse(_rpc_error(rpc_id, -32601, f"method not found: {method}"))

    return Starlette(
        routes=[
            Route("/.well-known/agent.json", get_card, methods=["GET"]),
            Route("/.well-known/agent-card.json", get_card, methods=["GET"]),
            Route("/", rpc, methods=["POST"]),
        ]
    )


def _stream(adapter: ChatAdapter, rpc_id: Any, text_in: str) -> Response:
    task_id, ctx_id = uuid4().hex, uuid4().hex

    def status(state: str, *, final: bool) -> str:
        event = {
            "taskId": task_id,
            "contextId": ctx_id,
            "kind": "status-update",
            "status": {"state": state},
            "final": final,
        }
        return sse_event(_rpc_result(rpc_id, event))

    async def frames() -> AsyncIterator[str]:
        yield status("working", final=False)
        reply = await adapter.reply([user(text_in)])
        artifact = {
            "taskId": task_id,
            "contextId": ctx_id,
            "kind": "artifact-update",
            "artifact": {"artifactId": uuid4().hex, "name": "reply", "parts": [_text_part(reply)]},
        }
        yield sse_event(_rpc_result(rpc_id, artifact))
        yield status("completed", final=True)

    return sse_response(frames())


def _reply_from_result(result: dict[str, Any]) -> str:
    # A `message/send` result is a Task (text in artifacts) or a Message (text in parts).
    artifact_parts: list[Any] = []
    for artifact in result.get("artifacts", []) or []:
        artifact_parts.extend(artifact.get("parts", []))
    return _text_from_parts(artifact_parts) or _text_from_parts(result.get("parts", []))


def a2a_tool(
    url: str,
    *,
    name: str = "a2a_agent",
    description: str = "Send a message to a remote A2A agent and return its reply.",
    client: httpx.AsyncClient | None = None,
) -> Tool:
    """A TensorSketch `Tool` that calls a remote A2A agent — so a graph can delegate to another
    agent.

    Pass a preconfigured `client` (auth headers, timeouts, a test transport) to reuse it; otherwise
    a short-lived client is created per call.
    """

    async def call(message: str) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": uuid4().hex,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": uuid4().hex,
                    "parts": [_text_part(message)],
                }
            },
        }
        http = client or httpx.AsyncClient()
        try:
            response = await http.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
        finally:
            if client is None:
                await http.aclose()
        if "error" in data:
            return f"error: {data['error'].get('message', 'A2A call failed')}"
        return _reply_from_result(data.get("result", {}))

    return Tool(
        call,
        name=name,
        description=description,
        json_schema={
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to send."}},
            "required": ["message"],
        },
    )
