"""Serve a TensorSketch agent over standard protocols — optional `tensorsketch-core[serve]`.

Each factory turns an agent-shaped `CompiledGraph` into a mountable **ASGI app** (Starlette); run
it with any ASGI server (`uvicorn module:app`) or mount it under an existing app:

    from tensorsketch.serve import openai_app, a2a_app, agui_app

    app = openai_app(agent, model="my-bot")   # OpenAI-compatible /v1/chat/completions
    app = a2a_app(agent, name="my-bot")        # A2A agent card + message send/stream
    app = agui_app(agent)                       # AG-UI event stream for a frontend

Consume other agents from inside a graph with `a2a_tool(url)`. Starlette and httpx are imported
only here, so `import tensorsketch` still pulls in no web framework. `ChatAdapter` (and `to_input` /
`to_reply`) is the seam if your graph isn't shaped like `create_agent`.
"""

from __future__ import annotations

from ._adapter import ChatAdapter, ToInput, ToReply, last_user_query, reply_text
from .a2a import AgentCard, a2a_app, a2a_tool
from .agui import agui_app
from .openai import openai_app

__all__ = [
    "AgentCard",
    "ChatAdapter",
    "ToInput",
    "ToReply",
    "a2a_app",
    "a2a_tool",
    "agui_app",
    "last_user_query",
    "openai_app",
    "reply_text",
]
