"""A single model call as a node, plus a helper for typed structured output.

`Llm` is the simplest compute node: one prompt in, one text answer out. `generate_structured`
asks the model for a specific `Schema` and returns a validated instance — usable directly inside
any node body (pass `ctx` to make the call durable).
"""

from __future__ import annotations

from typing import TypeVar, cast

from pydantic import ValidationError

from ..core.context import Context
from ..core.node import Node
from ..core.schema import Schema
from ..messages import Message, system, user
from ..providers.base import ChatProvider, Completion

S = TypeVar("S", bound=Schema)


def _conversation(system_prompt: str, query: str) -> list[Message]:
    convo: list[Message] = []
    if system_prompt:
        convo.append(system(system_prompt))
    convo.append(user(query))
    return convo


class Llm(Node):
    """One model call. Reads `query`, writes the reply text to `output`."""

    class In(Schema):
        query: str

    class Out(Schema):
        output: str

    def __init__(
        self,
        provider: ChatProvider,
        *,
        system: str = "",
        max_tokens: int = 1024,
        name: str = "Llm",
    ) -> None:
        self._provider = provider
        self._system = system
        self._max_tokens = max_tokens
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def run(self, ctx: Context, inp: In) -> Out:
        convo = _conversation(self._system, inp.query)
        completion = await ctx.step(
            "model",
            lambda: self._provider.complete(convo, max_tokens=self._max_tokens),
        )
        return self.Out(output=completion.message.content)


async def generate_structured(
    provider: ChatProvider,
    output: type[S],
    query: str,
    *,
    system: str = "",
    ctx: Context | None = None,
    max_repairs: int = 2,
) -> S:
    """Ask the model for a `Schema` and return a validated instance, repairing if needed.

    If the model's reply doesn't match the schema, the error is fed back and the model is asked
    again, up to `max_repairs` times (the validate-and-repair loop). Pass `ctx` to journal each
    attempt (durable, exactly-once). Raises `ValueError` if every attempt fails.
    """
    convo = _conversation(system, query)
    last_error = "no structured output returned"
    for attempt in range(max_repairs + 1):
        try:
            completion = await _complete_structured(provider, convo, output, ctx, attempt)
            if completion.parsed is not None:
                return cast(S, completion.parsed)
        except ValidationError as exc:
            last_error = str(exc)
        convo.append(
            user(
                f"Your previous response did not match the required schema ({last_error}). "
                f"Reply with only valid JSON that matches it."
            )
        )
    raise ValueError(f"structured output failed after {max_repairs + 1} attempt(s): {last_error}")


async def _complete_structured(
    provider: ChatProvider,
    convo: list[Message],
    output: type[Schema],
    ctx: Context | None,
    attempt: int,
) -> Completion:
    if ctx is not None:

        async def call() -> Completion:
            return await provider.complete(convo, output_schema=output)

        return await ctx.step(f"structured:{attempt}", call)
    return await provider.complete(convo, output_schema=output)
