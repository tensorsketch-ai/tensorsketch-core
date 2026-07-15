"""Agent primitives: tools with auto-schema, single calls, the agent loop, and durability."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from tensorsketch import (
    END,
    START,
    FakeProvider,
    Graph,
    InMemoryBackend,
    Llm,
    Message,
    Schema,
    create_agent,
    generate_structured,
    tool,
)
from tensorsketch.messages import ToolCall, assistant
from tensorsketch.tools import Tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# --------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------


def test_tool_auto_schema_from_signature() -> None:
    schema = add.json_schema()
    assert set(schema["properties"]) == {"a", "b"}
    assert schema["properties"]["a"]["type"] == "integer"
    assert add.name == "add"
    assert add.description == "Add two numbers."


async def test_tool_validates_and_runs() -> None:
    assert await add.run({"a": 2, "b": 3}) == 5
    with pytest.raises(ValidationError):
        await add.run({"a": "not-an-int", "b": 3})


async def test_async_tool_is_awaited() -> None:
    @tool
    async def fetch(url: str) -> str:
        """Pretend to fetch a URL."""
        return f"body of {url}"

    assert await fetch.run({"url": "x"}) == "body of x"


# --------------------------------------------------------------------------------------------
# Single model call
# --------------------------------------------------------------------------------------------


async def test_llm_node_single_call() -> None:
    class S(Schema):
        query: str
        output: str = ""

    provider = FakeProvider([assistant("hello there")])
    app = Graph(S).add(Llm(provider)).edge(START, "Llm").edge("Llm", END).compile()
    out = await app.invoke({"query": "hi"})
    assert out.output == "hello there"


async def test_structured_output_is_validated() -> None:
    class Person(Schema):
        name: str
        age: int

    provider = FakeProvider([assistant('{"name": "Ada", "age": 36}')])
    person = await generate_structured(provider, Person, "who invented the analytical engine?")
    assert person.name == "Ada"
    assert person.age == 36


# --------------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------------


def _calc_script() -> list[Message]:
    return [
        assistant(tool_calls=[ToolCall(id="c1", name="add", args={"a": 2, "b": 3})]),
        assistant(content="The answer is 5."),
    ]


async def test_agent_runs_tool_loop() -> None:
    provider = FakeProvider(_calc_script())
    app = create_agent(provider, tools=[add], system="You are a calculator.")
    out = await app.invoke({"query": "what is 2 + 3?"})

    assert out.output == "The answer is 5."
    assert [m.role for m in out.messages] == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = out.messages[3]
    assert tool_msg.content == "5"  # the add tool actually ran and its result was fed back
    assert len(provider.calls) == 2  # one call to request the tool, one to answer


async def test_agent_without_tools_answers_directly() -> None:
    provider = FakeProvider([assistant("42")])
    app = create_agent(provider)
    out = await app.invoke({"query": "the meaning of life?"})
    assert out.output == "42"


async def test_agent_respects_iteration_budget() -> None:
    # A provider that always asks for a tool would loop forever; the budget stops it.
    def always_tool(messages: list[Message], tools: Sequence[Tool] | None) -> Message:
        return assistant(tool_calls=[ToolCall(id="c", name="add", args={"a": 1, "b": 1})])

    provider = FakeProvider(policy=always_tool)
    app = create_agent(provider, tools=[add], max_iterations=3)
    await app.invoke({"query": "loop"})
    assert len(provider.calls) == 3  # capped at the budget, no error


# --------------------------------------------------------------------------------------------
# Durable agent: model + tool calls run exactly once across a crash
# --------------------------------------------------------------------------------------------


async def test_durable_agent_calls_each_effect_once_across_crash() -> None:
    tool_runs = {"n": 0}

    @tool
    def counting_add(a: int, b: int) -> int:
        """Add, counting how many times it really runs."""
        tool_runs["n"] += 1
        return a + b

    final_attempts = {"n": 0}

    def policy(messages: list[Message], tools: Sequence[Tool] | None) -> Message:
        if messages[-1].role == "tool":
            final_attempts["n"] += 1
            if final_attempts["n"] == 1:
                raise RuntimeError("crash before the final answer")
            return assistant(content="done")
        call = ToolCall(id="c1", name="counting_add", args={"a": 10, "b": 5})
        return assistant(tool_calls=[call])

    backend = InMemoryBackend()
    provider = FakeProvider(policy=policy)
    app = create_agent(provider, tools=[counting_add])

    with pytest.raises(RuntimeError):
        await app.invoke({"query": "add 10 and 5"}, thread_id="a", backend=backend)
    assert tool_runs["n"] == 1  # the tool ran once, before the crash

    out = await app.invoke(thread_id="a", backend=backend)  # resume
    assert out.output == "done"
    assert tool_runs["n"] == 1  # still once — the tool call was replayed from the journal
