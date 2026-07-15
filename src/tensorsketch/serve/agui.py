"""AG-UI interop — stream a TensorSketch run to a frontend as standard agent↔UI events.

`agui_app(agent)` returns a Starlette app with a single `POST /` endpoint that accepts an AG-UI
`RunAgentInput` (`threadId`, `runId`, `messages`, ...) and returns an SSE stream of AG-UI events:
`RUN_STARTED`, `TEXT_MESSAGE_START` / `_CONTENT` / `_END`, a `STATE_SNAPSHOT` of the final state,
and `RUN_FINISHED` (or `RUN_ERROR`). A CopilotKit / AG-UI client can render the run directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ..core.graph import CompiledGraph
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


def _state_snapshot(state: Any) -> dict[str, Any]:
    dump = getattr(state, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    return dict(state) if isinstance(state, dict) else {}


def agui_app(
    graph: CompiledGraph[Any],
    *,
    to_input: ToInput | None = None,
    to_reply: ToReply | None = None,
) -> Starlette:
    """Build an AG-UI ASGI app for `graph`."""
    adapter = ChatAdapter(
        graph, to_input=to_input or last_user_query, to_reply=to_reply or reply_text
    )

    async def run(request: Request) -> Response:
        body = await request.json()
        thread_id = body.get("threadId") or uuid4().hex
        run_id = body.get("runId") or uuid4().hex
        messages = messages_from_dicts(body.get("messages", []))

        async def frames() -> AsyncIterator[str]:
            yield sse_event({"type": "RUN_STARTED", "threadId": thread_id, "runId": run_id})
            try:
                state = await adapter.run_state(messages)
                text = adapter.to_reply(state)
                message_id = uuid4().hex
                yield sse_event(
                    {"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"}
                )
                for piece in text_chunks(text):
                    yield sse_event(
                        {"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": piece}
                    )
                yield sse_event({"type": "TEXT_MESSAGE_END", "messageId": message_id})
                yield sse_event({"type": "STATE_SNAPSHOT", "snapshot": _state_snapshot(state)})
                yield sse_event({"type": "RUN_FINISHED", "threadId": thread_id, "runId": run_id})
            except Exception as exc:  # surface a failed run as a protocol event, not a 500
                yield sse_event({"type": "RUN_ERROR", "message": str(exc)})

        return sse_response(frames())

    return Starlette(routes=[Route("/", run, methods=["POST"])])
