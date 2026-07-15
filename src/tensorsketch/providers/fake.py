"""A scripted provider for tests and examples — no API key, fully deterministic.

Drive it either with a fixed `script` of assistant replies (returned in order) or a `policy`
function that inspects the conversation and decides the reply. It records every call it receives
on `.calls`, which makes it easy to assert what an agent sent the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..core.schema import Schema
from ..messages import Message
from ..tools import Tool
from .base import ChatProvider, Completion, Usage

Policy = Callable[[list[Message], Sequence[Tool] | None], Message]


class FakeProvider(ChatProvider):
    """A `ChatProvider` that returns canned replies. Give it a `script` or a `policy`."""

    def __init__(
        self,
        script: Sequence[Message] | None = None,
        *,
        policy: Policy | None = None,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        if script is None and policy is None:
            raise ValueError("FakeProvider needs either a script or a policy")
        self._script = list(script) if script is not None else None
        self._policy = policy
        self._model = model
        self._usage = usage
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options: Any,
    ) -> Completion:
        convo = list(messages)
        self.calls.append(convo)
        if self._policy is not None:
            reply = self._policy(convo, tools)
        else:
            assert self._script is not None
            if not self._script:
                raise RuntimeError("FakeProvider script exhausted")
            reply = self._script.pop(0)
        parsed = (
            output_schema.model_validate_json(reply.content) if output_schema is not None else None
        )
        usage = self._usage if self._usage is not None else Usage()
        return Completion(message=reply, parsed=parsed, usage=usage, model=self._model)
