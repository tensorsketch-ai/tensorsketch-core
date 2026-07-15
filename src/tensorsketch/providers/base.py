"""The provider abstraction: one interface every model backend implements.

`ChatProvider` is the single seam between TensorSketch and a model API. The core defines *only* this
interface and its data types — it depends on **no** provider SDK. Real providers (Anthropic,
OpenAI, ...) are separate, optional installs that implement `complete`. Swapping models is
swapping a provider; nothing else in a graph changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.schema import Schema
from ..messages import Message
from ..tools import Tool


@dataclass
class Usage:
    """Token accounting for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Completion:
    """The result of a `ChatProvider.complete` call.

    Attributes:
        message: The assistant's reply (may carry `tool_calls`).
        parsed: The validated structured output, when an `output_schema` was requested.
        usage: Token usage for the call.
        model: The model id that produced this reply (for tracing and cost). Providers set it
            from the response; None when unknown.
    """

    message: Message
    parsed: Schema | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None


class ChatProvider(ABC):
    """Interface a model backend implements. Providers are stateless with respect to a run."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        """Generate the next assistant message.

        Args:
            messages: The conversation so far.
            tools: Tools the model may call; a reply may request calls via `message.tool_calls`.
            output_schema: If given, the model is asked to produce this structured type and the
                result is validated into `Completion.parsed`.
            max_tokens: Generation cap.
            **options: Provider-specific knobs (temperature, etc.).
        """
