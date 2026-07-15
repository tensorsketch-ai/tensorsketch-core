"""OpenAI + Google provider mapping tests, driven by injected fake clients (no SDK, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tensorsketch import Schema, tool
from tensorsketch.messages import system, tool_result, user
from tensorsketch.providers.google import GoogleProvider, _to_google
from tensorsketch.providers.openai import OpenAIProvider


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Person(Schema):
    name: str
    age: int


# -- OpenAI -----------------------------------------------------------------------------------


class _OpenAICompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.received: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.received = kwargs
        return self._response


class _OpenAIClient:
    def __init__(self, response: Any) -> None:
        self.chat = SimpleNamespace(completions=_OpenAICompletions(response))


def _openai_response(*, content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )


def _openai_tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


async def test_openai_text_and_messages() -> None:
    client = _OpenAIClient(_openai_response(content="hi!"))
    provider = OpenAIProvider(client=client)
    completion = await provider.complete([system("be terse"), user("hello")])

    assert completion.message.content == "hi!"
    assert completion.usage.input_tokens == 3
    messages = client.chat.completions.received["messages"]
    assert messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]


async def test_openai_tool_calls() -> None:
    client = _OpenAIClient(
        _openai_response(tool_calls=[_openai_tool_call("t1", "add", '{"a": 1, "b": 2}')])
    )
    provider = OpenAIProvider(client=client)
    completion = await provider.complete([user("add")], tools=[add])

    assert completion.message.tool_calls[0].name == "add"
    assert completion.message.tool_calls[0].args == {"a": 1, "b": 2}
    assert client.chat.completions.received["tools"][0]["function"]["name"] == "add"


async def test_openai_structured_output() -> None:
    client = _OpenAIClient(_openai_response(content='{"name": "Ada", "age": 36}'))
    provider = OpenAIProvider(client=client)
    completion = await provider.complete([user("who?")], output_schema=Person)

    assert completion.parsed == Person(name="Ada", age=36)
    assert client.chat.completions.received["response_format"]["type"] == "json_schema"


# -- Google -----------------------------------------------------------------------------------


class _GoogleModels:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.received: dict[str, Any] = {}

    async def generate_content(self, **kwargs: Any) -> Any:
        self.received = kwargs
        return self._response


class _GoogleClient:
    def __init__(self, response: Any) -> None:
        self.aio = SimpleNamespace(models=_GoogleModels(response))


def _google_response(parts: list[Any]) -> Any:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=6),
    )


def _google_text(value: str) -> Any:
    return SimpleNamespace(text=value, function_call=None)


def _google_call(name: str, args: dict[str, Any]) -> Any:
    return SimpleNamespace(text=None, function_call=SimpleNamespace(name=name, args=args))


async def test_google_text_and_system() -> None:
    client = _GoogleClient(_google_response([_google_text("hi!")]))
    provider = GoogleProvider(client=client)
    completion = await provider.complete([system("be terse"), user("hello")])

    assert completion.message.content == "hi!"
    assert completion.usage.output_tokens == 6
    received = client.aio.models.received
    assert received["config"]["system_instruction"] == "be terse"
    assert received["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


async def test_google_tool_calls() -> None:
    client = _GoogleClient(_google_response([_google_call("add", {"a": 1, "b": 2})]))
    provider = GoogleProvider(client=client)
    completion = await provider.complete([user("add")], tools=[add])

    assert completion.message.tool_calls[0].name == "add"
    assert completion.message.tool_calls[0].args == {"a": 1, "b": 2}
    tools = client.aio.models.received["config"]["tools"]
    assert tools[0]["function_declarations"][0]["name"] == "add"


async def test_google_structured_output() -> None:
    client = _GoogleClient(_google_response([_google_text('{"name": "Ada", "age": 36}')]))
    provider = GoogleProvider(client=client)
    completion = await provider.complete([user("who?")], output_schema=Person)

    assert completion.parsed == Person(name="Ada", age=36)
    assert client.aio.models.received["config"]["response_mime_type"] == "application/json"


def test_google_conversion_merges_system_and_tool_results() -> None:
    convo = [system("s"), user("q"), tool_result("t1", "42", name="add")]
    system_text, contents = _to_google(convo)

    assert system_text == "s"
    assert contents[0] == {"role": "user", "parts": [{"text": "q"}]}
    assert contents[1]["parts"][0]["function_response"]["name"] == "add"
