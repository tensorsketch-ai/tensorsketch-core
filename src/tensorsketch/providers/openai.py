"""The OpenAI provider — an optional install (`pip install tensorsketch-core[openai]`).

Uses the Chat Completions API, so it also works with any **OpenAI-compatible** endpoint (Together,
Groq, local vLLM/Ollama, …) by passing `base_url`. The `openai` SDK is imported lazily, so the
core never depends on it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..core.schema import Schema
from ..messages import Message, ToolCall
from ..tools import Tool
from .base import ChatProvider, Completion, Usage


class OpenAIProvider(ChatProvider):
    """A `ChatProvider` backed by the OpenAI Chat Completions API (or a compatible endpoint).

    Args:
        model: Model id (default `gpt-4o`).
        api_key: Overrides `OPENAI_API_KEY`.
        base_url: Point at any OpenAI-compatible server.
        client: Inject a pre-built (or fake) async client; if given, `openai` is not imported.
        **defaults: Default request options applied to every call.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        **defaults: Any,
    ) -> None:
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "OpenAIProvider requires the 'openai' package. "
                    "Install:  pip install tensorsketch-core[openai]"
                ) from exc
            client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
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
        request: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai(messages),
            "max_tokens": max_tokens,
            **self._defaults,
            **options,
        }
        if output_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.json_schema(),
                    "strict": True,
                },
            }
        elif tools:
            request["tools"] = [_tool_spec(t) for t in tools]

        response = await self._client.chat.completions.create(**request)
        return _from_openai(response, output_schema)


def _tool_spec(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema(),
        },
    }


def _to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.args)},
                        }
                        for call in msg.tool_calls
                    ],
                }
            )
        elif msg.role == "tool":
            out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
        else:
            out.append({"role": msg.role, "content": msg.content})
    return out


def _from_openai(response: Any, output_schema: type[Schema] | None) -> Completion:
    choice = response.choices[0].message
    tool_calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            args=json.loads(call.function.arguments or "{}"),
        )
        for call in (choice.tool_calls or [])
    ]
    content = choice.content or ""
    parsed = output_schema.model_validate_json(content) if output_schema and content else None
    usage = Usage(
        input_tokens=getattr(response.usage, "prompt_tokens", 0),
        output_tokens=getattr(response.usage, "completion_tokens", 0),
    )
    return Completion(
        message=Message(role="assistant", content=content, tool_calls=tool_calls),
        parsed=parsed,
        usage=usage,
        model=getattr(response, "model", None),
    )
