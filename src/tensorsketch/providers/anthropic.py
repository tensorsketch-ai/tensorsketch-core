"""The Anthropic provider — an optional install (`pip install tensorsketch-core[anthropic]`).

The `anthropic` SDK is imported lazily inside the constructor, so importing TensorSketch (or even
this
module) never requires it. This is the pattern every real provider follows: the core stays
SDK-free; each backend is an opt-in dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.schema import Schema
from ..messages import Message, ToolCall
from ..tools import Tool
from .base import ChatProvider, Completion, Usage


class AnthropicProvider(ChatProvider):
    """A `ChatProvider` backed by the Anthropic Messages API.

    Args:
        model: The model id (default a balanced Sonnet; pass an Opus id for the hardest tasks).
        api_key: Overrides the `ANTHROPIC_API_KEY` environment variable.
        client: Inject a pre-built (or fake) async client; if given, the `anthropic` package is
            not imported.
        **defaults: Default request options (e.g. `temperature`) applied to every call.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        client: Any = None,
        **defaults: Any,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' package. "
                    "Install it with:  pip install tensorsketch-core[anthropic]"
                ) from exc
            client = anthropic.AsyncAnthropic(api_key=api_key)
        self._client = client
        self._model = model
        self._defaults = defaults

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        system_text, convo = _to_anthropic(messages)
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": convo,
            **self._defaults,
            **options,
        }
        if system_text:
            request["system"] = system_text
        if output_schema is not None:
            # Force the model to answer through a single "respond" tool whose schema is the
            # requested output type — a reliable way to get structured output.
            request["tools"] = [
                {
                    "name": "respond",
                    "description": "Return the final structured answer.",
                    "input_schema": output_schema.json_schema(),
                }
            ]
            request["tool_choice"] = {"type": "tool", "name": "respond"}
        elif tools:
            request["tools"] = [_tool_spec(t) for t in tools]

        response = await self._client.messages.create(**request)
        return _from_anthropic(response, output_schema)


def _tool_spec(tool: Tool) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.json_schema()}


def _to_anthropic(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert TensorSketch messages to (system_text, Anthropic message list)."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            out.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
                )
            out.append({"role": "assistant", "content": blocks or msg.content})
        elif msg.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
            }
            # Anthropic carries tool results inside a following user turn; merge consecutive ones.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return "\n\n".join(p for p in system_parts if p), out


def _from_anthropic(response: Any, output_schema: type[Schema] | None) -> Completion:
    """Convert an Anthropic response into a TensorSketch `Completion`."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    parsed: Schema | None = None
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            if output_schema is not None and block.name == "respond":
                parsed = output_schema.model_validate(block.input)
            else:
                tool_calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))
    usage = Usage(
        input_tokens=getattr(response.usage, "input_tokens", 0),
        output_tokens=getattr(response.usage, "output_tokens", 0),
    )
    message = Message(role="assistant", content="".join(text_parts), tool_calls=tool_calls)
    return Completion(
        message=message, parsed=parsed, usage=usage, model=getattr(response, "model", None)
    )
