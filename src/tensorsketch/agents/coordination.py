"""Multi-agent coordination: expose an agent (or any graph) as a tool another agent can call.

This is the **supervisor / agents-as-tools** pattern. A coordinator agent is handed specialist
agents as `Tool`s; it delegates by *calling* one, reads its answer, and composes a final result —
the very same ReAct loop, one level up. Nothing new in the runtime: a delegated call is an
ordinary tool call, so it inherits the agent's durability (journaled — a specialist isn't re-run
on resume) and its tracing (the specialist's spans nest under the delegating tool call, so a
single trace shows the whole team, with per-specialist cost).

    supervisor = create_agent(provider, tools=[
        as_tool(billing_agent, name="billing", description="Answer billing questions."),
        as_tool(tech_agent, name="tech", description="Debug technical problems."),
    ])
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import Field, create_model

from ..core.context import Context
from ..core.graph import CompiledGraph
from ..core.schema import Schema
from ..tools import Tool


def as_tool(
    graph: CompiledGraph[Any],
    *,
    name: str,
    description: str,
    input_key: str = "query",
    output_key: str = "output",
    arg: str = "request",
    arg_description: str = "The task or question to hand to this agent.",
) -> Tool:
    """Wrap a compiled agent/graph as a `Tool` a supervisor agent can call.

    The tool advertises one string argument (`arg`); calling it runs `graph` with
    `{input_key: value}` and returns the graph's `output_key` field as text. The defaults match
    `create_agent` (a `query` in, an `output` out) — override the keys to wrap a graph whose state
    is shaped differently.

    The call runs under the caller's trace (so the sub-agent's spans nest under the delegating
    tool call) and, because it's a normal tool call, is journaled by the calling agent — a
    specialist is never re-run when the supervisor resumes.
    """
    fields: dict[str, Any] = {arg: (str, Field(..., description=arg_description))}
    args_schema = cast(type[Schema], create_model(f"{name}_Args", __base__=Schema, **fields))

    async def call(ctx: Context, **kwargs: Any) -> str:
        tracer = ctx.tracer if ctx is not None else None
        result = await graph.invoke({input_key: kwargs[arg]}, tracer=tracer)
        output = getattr(result, output_key, "")
        return output if isinstance(output, str) else str(output)

    return Tool(call, name=name, description=description, args_schema=args_schema)
