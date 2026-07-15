"""Messages: the conversation currency between agents and models.

A `Message` is one turn in a conversation — a system instruction, a user query, an assistant
reply (possibly requesting tool calls), or a tool result. Messages are `Schema`s, so they
validate, serialize, and checkpoint like everything else in TensorSketch, and a list of them is a
natural typed state channel (with the `add_messages` reducer to append across steps).
"""

from __future__ import annotations

from typing import Any, Literal

from .core.schema import Schema

#: Who a message is from.
Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(Schema):
    """A model's request to call a tool: a stable `id`, the tool `name`, and its arguments."""

    id: str
    name: str
    args: dict[str, Any] = {}


class Message(Schema):
    """One conversation turn.

    `content` is the text. An assistant turn may carry `tool_calls`; a `tool` turn carries the
    `tool_call_id` it answers (and the tool `name`).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    name: str | None = None


def system(content: str) -> Message:
    """A system instruction message."""
    return Message(role="system", content=content)


def user(content: str) -> Message:
    """A user message."""
    return Message(role="user", content=content)


def assistant(content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
    """An assistant message, optionally requesting tool calls."""
    return Message(role="assistant", content=content, tool_calls=tool_calls or [])


def tool_result(tool_call_id: str, content: str, name: str | None = None) -> Message:
    """A tool-result message answering a specific `tool_call_id`."""
    return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)


def add_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Reducer for a messages channel: append `right` onto `left`.

    Use as `messages: Annotated[list[Message], Reducer(add_messages)]`. Kept as a named function
    (rather than `operator.add`) so it can grow smarter later — e.g. de-duplicating by a message
    id — without changing how state is declared.
    """
    return [*left, *right]
