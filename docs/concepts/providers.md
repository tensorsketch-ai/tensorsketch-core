# Providers

A **provider** is the one seam between TensorSketch and a model API. The core defines only the
interface — it depends on **no** model SDK. Real providers are optional installs; swapping
models means swapping a provider, and nothing else in your graph changes.

## The interface

```python
class ChatProvider(ABC):
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        output_schema: type[Schema] | None = None,
        max_tokens: int = 1024,
        **options,
    ) -> Completion: ...
```

A `Completion` carries the assistant `message` (which may include `tool_calls`), the validated
`parsed` structured output (when an `output_schema` was requested), and token `usage`.

## FakeProvider — for tests and offline runs

`FakeProvider` returns canned replies, so you can build and test agents with no API key and
full determinism. Drive it with a fixed script or a policy function, and inspect `.calls` to
assert what the model was sent.

```python
from tensorsketch import FakeProvider
from tensorsketch.messages import assistant

provider = FakeProvider([assistant("hello there")])          # scripted
provider = FakeProvider(policy=lambda msgs, tools: assistant("..."))  # dynamic
```

## Built-in providers

Three real providers ship with TensorSketch, each an optional install imported lazily (so importing
TensorSketch never pulls in an SDK). They're interchangeable — same interface, same agents:

| Provider | Install | Import |
|---|---|---|
| **Anthropic** | `pip install tensorsketch-core[anthropic]` | `from tensorsketch.providers.anthropic import AnthropicProvider` |
| **OpenAI** | `pip install tensorsketch-core[openai]` | `from tensorsketch.providers.openai import OpenAIProvider` |
| **Google (Gemini)** | `pip install tensorsketch-core[google]` | `from tensorsketch.providers.google import GoogleProvider` |

```python
from tensorsketch.providers.openai import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o")                      # reads OPENAI_API_KEY
provider = OpenAIProvider(base_url="http://localhost:11434/v1")  # any OpenAI-compatible endpoint
agent = create_agent(provider, tools=[...])
```

`OpenAIProvider` speaks the Chat Completions API, so it also drives OpenAI-compatible servers
(Together, Groq, vLLM, Ollama, …) via `base_url` — one provider, many backends.

## Structured output

Ask a provider for a typed result by passing `output_schema`; the validated instance comes back
on `completion.parsed`. The [`generate_structured`](agents.md#structured-output) helper wraps
this into a one-liner. Each provider implements it the way its API allows (OpenAI/Google use a
JSON-schema response format; Anthropic forces a "respond" tool) — the interface is identical.

## Writing a custom provider

Any other backend is a small `ChatProvider`. Implement `complete`: map the conversation to your
API, call it, map the reply back to a `Completion`.

```python
from collections.abc import Sequence
from tensorsketch import ChatProvider, Completion, Message, Schema, Tool
from tensorsketch.messages import Message as Msg

class EchoProvider(ChatProvider):
    async def complete(self, messages, *, tools=None, output_schema=None, max_tokens=1024, **opts):
        last_user = next(m.content for m in reversed(messages) if m.role == "user")
        return Completion(message=Msg(role="assistant", content=f"You said: {last_user}"))
```

That's the whole contract. It works with every agent, graph, and pattern unchanged. The built-in
[Anthropic](../../src/tensorsketch/providers/anthropic.py), [OpenAI](../../src/tensorsketch/providers/openai.py),
and [Google](../../src/tensorsketch/providers/google.py) providers are compact real references.

> **Note:** the built-in providers' request/response mappings are covered by unit tests using
> injected fake clients; verify against the live APIs before relying on them in production (see
> the [decisions log](../design/decisions.md)).
