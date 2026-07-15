"""AnthropicProvider mapping tests, driven by an injected fake client (no SDK, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tensorsketch import Schema, tool
from tensorsketch.messages import ToolCall, assistant, system, tool_result, user
from tensorsketch.providers.anthropic import AnthropicProvider, _to_anthropic


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class _FakeMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.received: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.received = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = _FakeMessages(response)


def _response(blocks: list[Any], *, in_tokens: int = 1, out_tokens: int = 1) -> Any:
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
    )


def _text(value: str) -> Any:
    return SimpleNamespace(type="text", text=value)


def _tool_use(call_id: str, name: str, value: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=value)


async def test_text_completion_and_system_param() -> None:
    client = _FakeClient(_response([_text("hi!")], in_tokens=7))
    provider = AnthropicProvider(model="claude-sonnet-4-6", client=client)
    completion = await provider.complete([system("be terse"), user("hello")])

    assert completion.message.content == "hi!"
    assert completion.usage.input_tokens == 7
    assert client.messages.received["system"] == "be terse"
    assert client.messages.received["messages"] == [{"role": "user", "content": "hello"}]


async def test_tool_calls_are_mapped() -> None:
    client = _FakeClient(_response([_tool_use("t1", "add", {"a": 1, "b": 2})]))
    provider = AnthropicProvider(client=client)
    completion = await provider.complete([user("add 1 and 2")], tools=[add])

    assert completion.message.tool_calls[0].name == "add"
    assert completion.message.tool_calls[0].args == {"a": 1, "b": 2}
    assert client.messages.received["tools"][0]["name"] == "add"


async def test_structured_output_forces_respond_tool() -> None:
    class Person(Schema):
        name: str
        age: int

    client = _FakeClient(_response([_tool_use("r", "respond", {"name": "Ada", "age": 36})]))
    provider = AnthropicProvider(client=client)
    completion = await provider.complete([user("who?")], output_schema=Person)

    assert completion.parsed == Person(name="Ada", age=36)
    assert client.messages.received["tool_choice"] == {"type": "tool", "name": "respond"}


def test_conversation_conversion_merges_tool_results() -> None:
    convo = [
        system("s"),
        user("q"),
        assistant(tool_calls=[ToolCall(id="t1", name="add", args={"a": 1})]),
        tool_result("t1", "2"),
    ]
    system_text, out = _to_anthropic(convo)

    assert system_text == "s"
    assert out[0] == {"role": "user", "content": "q"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"][0]["type"] == "tool_use"
    assert out[2]["role"] == "user"  # the tool result rides in a following user turn
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "t1"
