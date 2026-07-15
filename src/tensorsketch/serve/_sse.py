"""Server-Sent-Events framing, shared by every serving protocol.

All three protocols stream results the same way — a `text/event-stream` of `data:` frames — so
the framing lives here once. Each protocol builds its own async iterator of frames (via
`sse_event`) and wraps it in `sse_response`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from starlette.responses import StreamingResponse

# Disable proxy/browser buffering so tokens reach the client as they're produced.
SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def sse_event(data: Any, *, event: str | None = None) -> str:
    """Frame one SSE message. `data` is JSON-encoded unless it's already a string (e.g. `[DONE]`);
    `event` sets the optional `event:` name line that A2A/AG-UI use to type each frame."""
    payload = data if isinstance(data, str) else json.dumps(data)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def sse_response(frames: AsyncIterator[str]) -> StreamingResponse:
    """A `text/event-stream` response over an async iterator of pre-framed SSE strings."""
    return StreamingResponse(frames, media_type="text/event-stream", headers=SSE_HEADERS)
