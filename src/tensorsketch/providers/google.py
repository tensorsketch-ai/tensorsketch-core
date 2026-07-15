"""The Google (Gemini) provider — an optional install (`pip install tensorsketch-core[google]`).

Uses the `google-genai` SDK, imported lazily. Gemini's message shape differs from the others
(roles are `user`/`model`, the system prompt is separate, tool results ride in a user turn), and
this module translates to and from TensorSketch's uniform `Message`/`Tool` model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.schema import Schema
from ..messages import Message, ToolCall
from ..tools import Tool
from .base import ChatProvider, Completion, Usage


class GoogleProvider(ChatProvider):
    """A `ChatProvider` backed by the Gemini API.

    Args:
        model: Model id (default `gemini-2.5-flash`).
        api_key: Overrides `GOOGLE_API_KEY`.
        client: Inject a pre-built (or fake) client; if given, `google-genai` is not imported.
        **defaults: Default generation options applied to every call.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        client: Any = None,
        **defaults: Any,
    ) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "GoogleProvider requires the 'google-genai' package. "
                    "Install:  pip install tensorsketch-core[google]"
                ) from exc
            client = genai.Client(api_key=api_key)
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
        system_text, contents = _to_google(messages)
        config: dict[str, Any] = {"max_output_tokens": max_tokens, **self._defaults, **options}
        if system_text:
            config["system_instruction"] = system_text
        if output_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = output_schema.json_schema()
        elif tools:
            config["tools"] = [{"function_declarations": [_tool_spec(t) for t in tools]}]

        response = await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        return _from_google(response, output_schema)


def _tool_spec(tool: Tool) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "parameters": tool.json_schema()}


def _to_google(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Convert TensorSketch messages to (system_instruction, Gemini contents)."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == "assistant":
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for call in msg.tool_calls:
                parts.append({"function_call": {"name": call.name, "args": call.args}})
            contents.append({"role": "model", "parts": parts})
        elif msg.role == "tool":
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": msg.name or "tool",
                                "response": {"result": msg.content},
                            }
                        }
                    ],
                }
            )
    return "\n\n".join(p for p in system_parts if p), contents


def _from_google(response: Any, output_schema: type[Schema] | None) -> Completion:
    parts = response.candidates[0].content.parts
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, part in enumerate(parts):
        call = getattr(part, "function_call", None)
        text = getattr(part, "text", None)
        if call is not None:
            tool_calls.append(
                ToolCall(id=f"{call.name}-{index}", name=call.name, args=dict(call.args or {}))
            )
        elif text:
            text_parts.append(text)
    content = "".join(text_parts)
    parsed = output_schema.model_validate_json(content) if output_schema and content else None
    metadata = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(metadata, "prompt_token_count", 0),
        output_tokens=getattr(metadata, "candidates_token_count", 0),
    )
    return Completion(
        message=Message(role="assistant", content=content, tool_calls=tool_calls),
        parsed=parsed,
        usage=usage,
        model=getattr(response, "model_version", None),
    )
