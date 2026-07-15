"""The agent: an autonomous model+tools loop, encapsulated as one durable node.

`Agent` runs the familiar loop — call the model, and if it asks for tools, run them and feed the
results back, until the model answers or a budget is hit. What makes TensorSketch's version
different is
that **every model and tool call is wrapped in `ctx.step`**, so the whole loop is durable: if the
process dies at iteration 5, resuming replays iterations 0-4 from the journal (no repeated API
calls, no reasoning drift) and continues. The agent is a normal `Node`, so it composes into any
graph; `create_agent` wraps one into a ready-to-run graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from ..core.channels import Reducer
from ..core.context import Context
from ..core.graph import END, START, CompiledGraph, Graph
from ..core.node import Node
from ..core.schema import Schema
from ..messages import Message, add_messages, system, tool_result, user
from ..middleware import (
    Middleware,
    ModelRequest,
    ToolRequest,
    compose_model,
    compose_tool,
)
from ..observability.cost import estimate_cost
from ..observability.tracing import (
    COST_USD,
    INPUT_TOKENS,
    MODEL,
    MODEL_KIND,
    OUTPUT_TOKENS,
    TOOL_ARGS,
    TOOL_KIND,
    TOOL_NAME,
    TOOL_RESULT,
    Span,
)
from ..providers.base import ChatProvider, Completion
from ..tools import Tool


class AgentState(Schema):
    """The state of a standalone agent run: a query in, an answer plus transcript out."""

    query: str
    output: str = ""
    messages: Annotated[list[Message], Reducer(add_messages)] = []


class Agent(Node):
    """A model+tools loop as one node. Reads `query`; writes `output` and the `messages` log."""

    class In(Schema):
        query: str

    class Out(Schema):
        output: str
        messages: list[Message]

    def __init__(
        self,
        provider: ChatProvider,
        *,
        tools: Sequence[Tool] = (),
        system: str = "",
        max_iterations: int = 8,
        middleware: Sequence[Middleware] = (),
        name: str = "Agent",
    ) -> None:
        self._provider = provider
        self._tools = {t.name: t for t in tools}
        self._system = system
        self._max_iterations = max_iterations
        self._middleware = tuple(middleware)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def run(self, ctx: Context, inp: In) -> Out:
        convo: list[Message] = []
        if self._system:
            convo.append(system(self._system))
        convo.append(user(inp.query))
        tool_list = list(self._tools.values())
        final = ""

        # The middleware stack wraps the raw model/tool calls; the whole wrapped call is then
        # journaled by ctx.step, so retries/logging never re-run on resume.
        model_call = compose_model(self._middleware, self._base_model)
        tool_call = compose_tool(self._middleware, self._base_tool)

        for i in range(self._max_iterations):
            request = ModelRequest(messages=convo, tools=tool_list, ctx=ctx, node=self._name)

            async def call_model(request: ModelRequest = request) -> Completion:
                with ctx.span("model_call", kind=MODEL_KIND) as span:
                    completion = await model_call(request)
                    _record_model(span, completion)
                    return completion

            completion = await ctx.step(f"model:{i}", call_model)
            reply = completion.message
            convo.append(reply)

            if not reply.tool_calls:
                final = reply.content
                break

            for call in reply.tool_calls:
                treq = ToolRequest(
                    call=call, ctx=ctx, node=self._name, tool=self._tools.get(call.name)
                )

                async def run_tool(treq: ToolRequest = treq) -> object:
                    attrs = {TOOL_NAME: treq.call.name, TOOL_ARGS: dict(treq.call.args)}
                    with ctx.span("tool_call", kind=TOOL_KIND, **attrs) as span:
                        result = await tool_call(treq)
                        span.set(**{TOOL_RESULT: str(result)})
                        return result

                result = await ctx.step(f"tool:{i}:{call.id}", run_tool)
                convo.append(tool_result(call.id, str(result), name=call.name))

        return self.Out(output=final, messages=convo)

    async def _base_model(self, request: ModelRequest) -> Completion:
        return await self._provider.complete(
            request.messages,
            tools=request.tools or None,
            output_schema=request.output_schema,
            **request.options,
        )

    async def _base_tool(self, request: ToolRequest) -> object:
        if request.tool is None:
            return f"error: unknown tool {request.call.name!r}"
        return await request.tool.run(request.call.args, request.ctx)


def _record_model(span: Span, completion: Completion) -> None:
    """Record model, token usage, and estimated cost on a model-call span (for cost eval)."""
    usage = completion.usage
    span.set(**{INPUT_TOKENS: usage.input_tokens, OUTPUT_TOKENS: usage.output_tokens})
    if completion.model:
        span.set(**{MODEL: completion.model})
        cost = estimate_cost(completion.model, usage.input_tokens, usage.output_tokens)
        if cost is not None:
            span.set(**{COST_USD: cost})


def create_agent(
    provider: ChatProvider,
    *,
    tools: Sequence[Tool] = (),
    system: str = "",
    max_iterations: int = 8,
    middleware: Sequence[Middleware] = (),
    name: str = "agent",
) -> CompiledGraph[AgentState]:
    """Build a ready-to-run agent graph. Invoke it with `{"query": "..."}`.

    `middleware` intercepts every model and tool call (retries, tracing, guardrails, …); see
    `tensorsketch.middleware`.
    """
    agent = Agent(
        provider,
        tools=tools,
        system=system,
        max_iterations=max_iterations,
        middleware=middleware,
        name=name,
    )
    return Graph(AgentState).add(agent, name=name).edge(START, name).edge(name, END).compile()
