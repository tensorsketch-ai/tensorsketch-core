# Serving a TensorSketch agent

A TensorSketch agent is a graph you can run in-process — but often you want to put it *behind an endpoint*
so other software can call it. TensorSketch serves an agent over three standard protocols, each as a
mountable **ASGI app**:

| Protocol | Who calls it | Factory |
|---|---|---|
| **OpenAI-compatible** | any OpenAI client/SDK pointed at your `base_url` | `openai_app(agent)` |
| **A2A** (Agent2Agent) | other agents, across frameworks | `a2a_app(agent)` |
| **AG-UI** | a frontend (CopilotKit / AG-UI client) | `agui_app(agent)` |

It's an optional install — the web stack is never pulled into the core:

```bash
pip install tensorsketch-core[serve]
```

Each factory returns a [Starlette](https://www.starlette.io/) app (itself an ASGI app), so you run
it with any ASGI server or mount it under an existing one:

```python
from tensorsketch import create_agent
from tensorsketch.serve import openai_app

agent = create_agent(provider, tools=[...])
app = openai_app(agent, model="my-bot")   # run:  uvicorn mymodule:app
```

`import tensorsketch` still imports no web framework — Starlette and httpx load only inside `tensorsketch.serve`.

## OpenAI-compatible

`openai_app` exposes `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`.
Point the OpenAI SDK at it and nothing else in your code changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(model="my-bot", messages=[{"role": "user", "content": "hi"}])
```

The last user message becomes the agent's input; the agent's answer comes back as the assistant
message. Streaming uses real SSE framing (`chat.completion.chunk` deltas ending in `[DONE]`).

## A2A (Agent2Agent)

`a2a_app` publishes an **Agent Card** for capability discovery at `/.well-known/agent.json` and
answers A2A's JSON-RPC methods `message/send` (one-shot) and `message/stream` (SSE task updates):

```python
from tensorsketch.serve import a2a_app, AgentCard

app = a2a_app(agent, card=AgentCard(name="research-bot", description="Searches and summarizes."))
```

The other direction — **calling** a remote A2A agent from inside a graph — is a tool:

```python
from tensorsketch.serve import a2a_tool

delegate = a2a_tool("https://other-agent.example/", name="ask_specialist")
agent = create_agent(provider, tools=[delegate])   # your agent can now hand off to theirs
```

So TensorSketch is on both sides of A2A: expose your agent to the ecosystem, and consume anyone else's.

## AG-UI

`agui_app` exposes a single `POST /` that accepts an AG-UI `RunAgentInput` and streams the run back
as typed UI events — `RUN_STARTED`, `TEXT_MESSAGE_START` / `_CONTENT` / `_END`, a `STATE_SNAPSHOT`
of the final state, and `RUN_FINISHED` (or `RUN_ERROR`) — which a CopilotKit/AG-UI frontend renders
directly.

## Serving a non-agent graph

The factories default to the `create_agent` shape — a `query` in, an `output` out. For a graph with
a different state, pass `to_input` / `to_reply` (see `ChatAdapter`) to map the request messages to
your input and pull the reply out of your state:

```python
app = openai_app(
    my_graph,
    to_input=lambda messages: {"prompt": messages[-1].content},
    to_reply=lambda state: state.answer,
)
```

## What's deferred

These are pragmatic, current-shaped implementations, not the entire surface of each spec:

- **Token-level streaming.** Providers don't stream tokens yet (that will flow through `ctx.emit`);
  today a streamed reply is the completed text sliced into SSE deltas — real framing, so it becomes
  true token streaming with no client change.
- **A2A** covers the agent card + `message/send` / `message/stream` with a completed-task result;
  the full task store, `tasks/get` / `tasks/cancel`, and push notifications are not implemented.
- **Multi-turn history and inbound tools** (the request carrying prior turns or tool defs) aren't
  wired into the default agent, which builds its own conversation from a single query.

See [`examples/serving.py`](../../examples/serving.py) for all three running offline in-process.

## Relationship to the other seams

Serving *exposes* an agent; [MCP interop](mcp.md) *connects tools*; [tracing](tracing.md) records
what a served run did. AG-UI is the UI-facing cousin of [streaming](streaming.md) — it's the same
run events, re-encoded in a protocol a frontend understands.
