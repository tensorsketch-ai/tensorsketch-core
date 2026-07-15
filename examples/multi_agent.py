"""Multi-agent coordination: a supervisor delegating to specialist agents (agents-as-tools).

`as_tool` wraps a compiled agent as a `Tool`, so a coordinator agent can *call* a specialist the
same way it calls any tool — the supervisor/handoff pattern with nothing new in the runtime. Each
delegated call is a normal tool call, so it's journaled (a specialist isn't re-run on resume) and
traced under the delegating call (one trace shows the whole team).

Runs offline with `FakeProvider`; the comments show the one-line swap to real models.

Run:  uv run python examples/multi_agent.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from tensorsketch import FakeProvider, Message, as_tool, create_agent
from tensorsketch.messages import ToolCall, assistant


def _canned(answer: str, name: str) -> object:
    """A specialist agent that just answers (a real one would have its own tools + model)."""
    return create_agent(FakeProvider(script=[assistant(answer)]), name=name)


def _triage(convo: Sequence[Message], tools: Sequence[object] | None) -> Message:
    """Supervisor policy: route the first query to a specialist, then relay its answer.

    A real supervisor would use a model here (`create_agent(AnthropicProvider(...), tools=[...])`);
    this rule-based policy keeps the example deterministic and offline.
    """
    if not any(m.role == "tool" for m in convo):
        query = next(m.content for m in convo if m.role == "user").lower()
        target = "billing" if any(w in query for w in ("refund", "charge", "invoice")) else "tech"
        return assistant(tool_calls=[ToolCall(id="c1", name=target, args={"request": query})])
    answer = [m for m in convo if m.role == "tool"][-1]
    return assistant(content=f"[routed to {answer.name}] {answer.content}")


async def main() -> None:
    billing = _canned(
        "I've issued a full refund for order #7 — you'll see it in 3-5 days.", "billing"
    )
    tech = _canned("Clear the app cache and restart; that resolves the login loop.", "tech")

    supervisor = create_agent(
        FakeProvider(policy=_triage),
        tools=[
            as_tool(billing, name="billing", description="Handles refunds, charges, and invoices."),
            as_tool(tech, name="tech", description="Debugs technical and login problems."),
        ],
        system="You are a support router. Delegate to the right specialist, then relay the answer.",
        name="supervisor",
    )
    # For real models, swap each FakeProvider for e.g. AnthropicProvider(model="claude-sonnet-4-6").

    for query in ["I need a refund for order #7", "the app is stuck on the login screen"]:
        result = await supervisor.invoke({"query": query})
        print(f"user:       {query}")
        print(f"supervisor: {result.output}\n")


if __name__ == "__main__":
    asyncio.run(main())
