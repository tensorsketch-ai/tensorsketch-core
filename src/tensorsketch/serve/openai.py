"""Serve a TensorSketch agent behind an OpenAI-compatible Chat Completions API.

`openai_app(agent)` returns a Starlette ASGI app exposing `POST /v1/chat/completions` (streaming
and non-streaming) and `GET /v1/models`, so any OpenAI client or SDK — pointed at your `base_url` —
can talk to a TensorSketch graph. Run it with any ASGI server (`uvicorn module:app`) or mount it
under an
existing app.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..core.graph import CompiledGraph
from ..messages import Message
from ._adapter import (
    ChatAdapter,
    ToInput,
    ToReply,
    last_user_query,
    messages_from_dicts,
    reply_text,
    text_chunks,
)
from ._sse import sse_event, sse_response


def openai_app(
    graph: CompiledGraph[Any],
    *,
    model: str = "tensorsketch",
    to_input: ToInput | None = None,
    to_reply: ToReply | None = None,
) -> Starlette:
    """Build an OpenAI-compatible ASGI app for `graph`.

    Args:
        graph: The agent/graph to serve (an agent-shaped `CompiledGraph`).
        model: The model id this endpoint advertises and echoes back.
        to_input / to_reply: Override the request↔state mapping (see `ChatAdapter`).
    """
    adapter = ChatAdapter(
        graph,
        to_input=to_input or last_user_query,
        to_reply=to_reply or reply_text,
    )

    def _completion_id() -> str:
        return f"chatcmpl-{uuid.uuid4().hex}"

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        messages = messages_from_dicts(body.get("messages", []))
        served_model = body.get("model") or model
        if body.get("stream"):
            return _stream(adapter, messages, served_model, _completion_id())
        text = await adapter.reply(messages)
        return JSONResponse(
            {
                "id": _completion_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                # Usage lives on the trace, not the response envelope; report zeros for the shape.
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    async def list_models(request: Request) -> Response:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "created": 0, "owned_by": "tensorsketch"}
                ],
            }
        )

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", list_models, methods=["GET"]),
        ]
    )


def _stream(adapter: ChatAdapter, messages: list[Message], model: str, cid: str) -> Response:
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None) -> str:
        return sse_event(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    async def frames() -> AsyncIterator[str]:
        yield chunk({"role": "assistant"}, None)
        text = await adapter.reply(messages)
        for piece in text_chunks(text):
            yield chunk({"content": piece}, None)
        yield chunk({}, "stop")
        yield sse_event("[DONE]")

    return sse_response(frames())
