"""The one seam between a TensorSketch graph and a chat protocol.

Every serving protocol (OpenAI, A2A, AG-UI) turns an inbound message list into a graph input, runs
the graph, and turns the final state back into an assistant reply. `ChatAdapter` captures exactly
that, with defaults that match `create_agent` (a `query` in, an `output` out) so serving a prebuilt
agent needs no configuration. Point it at a differently-shaped graph by overriding `to_input` /
`to_reply`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.graph import CompiledGraph
from ..messages import Message
from ..runtime.streaming import Event

#: Map the request's messages to the graph's input mapping.
ToInput = Callable[[list[Message]], Mapping[str, Any]]
#: Extract the assistant's reply text from the graph's final state.
ToReply = Callable[[Any], str]


def messages_from_dicts(raw: list[dict[str, Any]]) -> list[Message]:
    """Parse inbound role/content dicts (OpenAI / AG-UI shape) into TensorSketch `Message`s."""
    out: list[Message] = []
    for item in raw:
        role = item.get("role", "user")
        content = item.get("content") or ""
        if isinstance(content, list):  # OpenAI content-parts form → join the text parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role in ("system", "user", "assistant", "tool"):
            out.append(Message(role=role, content=str(content)))
    return out


def last_user_query(messages: list[Message]) -> dict[str, Any]:
    """Default request→input: the last user turn becomes `{"query": ...}` (`create_agent` shape)."""
    for message in reversed(messages):
        if message.role == "user":
            return {"query": message.content}
    return {"query": messages[-1].content if messages else ""}


def text_chunks(text: str, size: int = 24) -> list[str]:
    """Split reply text into small pieces for streaming deltas.

    Providers don't stream tokens yet (that flows through `ctx.emit` in a later phase), so a
    streamed reply is the completed text sliced into `size`-character pieces — real SSE framing
    and client behavior, with token-level streaming a drop-in once it lands.
    """
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


def reply_text(state: Any) -> str:
    """Default state→reply: the agent's `output`, else the last assistant message's content."""
    output = getattr(state, "output", None)
    if output:
        return str(output)
    messages = getattr(state, "messages", None) or []
    for message in reversed(messages):
        if getattr(message, "role", None) == "assistant" and getattr(message, "content", ""):
            return str(message.content)
    return ""


@dataclass
class ChatAdapter:
    """Runs an agent-shaped `CompiledGraph` as a chat endpoint.

    Attributes:
        graph: The compiled agent/graph to serve.
        to_input: Maps the request messages to the graph input (default: last user turn → `query`).
        to_reply: Extracts the reply text from the final state (default: `output` / last assistant).
    """

    graph: CompiledGraph[Any]
    to_input: ToInput = field(default=last_user_query)
    to_reply: ToReply = field(default=reply_text)

    async def run_state(self, messages: list[Message], **run: Any) -> Any:
        """Run to completion and return the final graph state."""
        return await self.graph.invoke(self.to_input(messages), **run)

    async def reply(self, messages: list[Message], **run: Any) -> str:
        """Run to completion and return the assistant reply text."""
        return self.to_reply(await self.run_state(messages, **run))

    def events(self, messages: list[Message], **run: Any) -> AsyncIterator[Event]:
        """Stream the run's live `Event`s (node/values/custom) for the given messages."""
        return self.graph.stream(self.to_input(messages), **run)
