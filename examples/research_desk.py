"""A multi-agent research desk — authored as a graph, so it draws on the canvas.

This is a fuller system than the single-purpose examples: it shows how the pieces compose into
something sophisticated *and* stays a plain, inspectable graph.

  * **Orchestration is an explicit `Graph`** — Plan → Research → Write → Review, with a
    **revision loop** (Review routes back to Revise until the draft is approved). Because the
    pipeline is real code with typed edges, you can open it on the canvas:

        python -m tensorsketch.canvas examples/research_desk.py

  * **Each stage delegates to a real agent.** Research runs a *tool-using* sub-agent (a
    model→`web_search`→model loop) once per sub-question; Write and Review run their own agents.
    Sub-agents are invoked with `tracer=ctx.tracer`, so the whole hierarchy is **one trace**.

  * **Durable by construction.** Every search is wrapped in `ctx.step(...)`, so a crash mid-run
    resumes without re-searching (see `durable_resume.py` for the crash/resume proof).

  * **Observable and gradable.** The run prints a nested span tree (timing · tokens · cost), and
    the same run is scored by the eval harness with *trajectory* graders — did the desk actually
    plan, search, and revise?

Everything runs offline with `FakeProvider` (deterministic policies stand in for models); each is a
one-line swap to a real provider, e.g. `AnthropicProvider(model="claude-sonnet-4-6")`.

Run:  uv run python examples/research_desk.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from tensorsketch import (
    END,
    START,
    Context,
    FakeProvider,
    Graph,
    InMemoryTracer,
    Node,
    Schema,
    create_agent,
    tool,
)
from tensorsketch.eval import (
    Case,
    Suite,
    ToolCalled,
    ToolSequence,
    Trial,
    evaluate,
    grader,
)
from tensorsketch.messages import Message, ToolCall, assistant

# --- a tiny offline "knowledge base" the research sub-agent searches --------------------------

KB = {
    "durable": "Durable execution journals each side effect so a resumed run never repeats it.",
    "bsp": "A BSP runtime advances in supersteps: parallel fan-out, then a reducer barrier.",
    "canvas": "Because the graph is code, it renders as a diagram and edits round-trip losslessly.",
    "plugin": "Providers, tools, storage, and protocols are plugins behind small interfaces.",
}
MAX_REVISIONS = 3


@tool
def web_search(query: str) -> str:
    """Search the knowledge base for a fact relevant to the query (offline stub)."""
    hits = [fact for key, fact in KB.items() if key in query.lower()]
    return hits[0] if hits else "No direct hit; recommend consulting a primary source."


# --- the specialist agents (real model→tool loops; here driven by offline policies) ----------


def _researcher() -> object:
    """A tool-using agent: search, then report the finding. (Swap the provider for a real model.)"""

    def policy(convo: Sequence[Message], tools: object) -> Message:
        if any(m.role == "tool" for m in convo):
            fact = [m for m in convo if m.role == "tool"][-1].content
            return assistant(f"Finding: {fact}")
        question = next(m.content for m in convo if m.role == "user")
        return assistant(
            "", tool_calls=[ToolCall(id="s1", name="web_search", args={"query": question})]
        )

    return create_agent(FakeProvider(policy=policy), tools=[web_search], name="researcher")


def _writer() -> object:
    """Composes a first draft from the collected findings."""

    def policy(convo: Sequence[Message], tools: object) -> Message:
        brief = next(m.content for m in convo if m.role == "user")
        return assistant(f"# Draft\n\n{brief}")

    return create_agent(FakeProvider(policy=policy), name="writer")


def _critic() -> object:
    """Reviews a draft: approve it, or ask for a specific revision."""

    def policy(convo: Sequence[Message], tools: object) -> Message:
        draft = next(m.content for m in convo if m.role == "user")
        if "Sources:" in draft:
            return assistant("APPROVE — claims are sourced and on-topic.")
        return assistant("REVISE — add a Sources section citing the findings.")

    return create_agent(FakeProvider(policy=policy), name="critic")


RESEARCHER = _researcher()
WRITER = _writer()
CRITIC = _critic()


# --- shared, typed state (each field is a channel) -------------------------------------------


class Desk(Schema):
    topic: str
    plan: list[str] = []
    findings: list[str] = []
    sources: list[str] = []
    draft: str = ""
    critique: str = ""
    revisions: int = 0
    approved: bool = False
    report: str = ""


# --- the pipeline stages, each a node that delegates to an agent ------------------------------


class Plan(Node):
    """Decompose the topic into sub-questions to research."""

    class In(Schema):
        topic: str

    class Out(Schema):
        plan: list[str]

    async def run(self, ctx: Context, inp: In) -> Out:
        angles = ["durable", "bsp", "canvas", "plugin"]
        return self.Out(plan=[f"{inp.topic}: {angle}" for angle in angles])


class Research(Node):
    """Run the research sub-agent on each sub-question (each search is a durable step)."""

    class In(Schema):
        plan: list[str]

    class Out(Schema):
        findings: list[str]
        sources: list[str]

    async def run(self, ctx: Context, inp: In) -> Out:
        findings: list[str] = []
        sources: list[str] = []
        for question in inp.plan:

            async def _investigate(q: str = question) -> str:
                state = await RESEARCHER.invoke({"query": q}, tracer=ctx.tracer)
                return str(state.output)

            # Journaled: on resume, a completed search is replayed, not re-run.
            findings.append(await ctx.step(f"research:{question}", _investigate))
            sources.append(f"kb://{question.split(':')[-1].strip()}")
        return self.Out(findings=findings, sources=sources)


class Write(Node):
    """Have the writer agent compose a first draft from the findings."""

    class In(Schema):
        topic: str
        findings: list[str]

    class Out(Schema):
        draft: str

    async def run(self, ctx: Context, inp: In) -> Out:
        brief = f"Topic: {inp.topic}\n\n" + "\n".join(f"- {f}" for f in inp.findings)
        state = await WRITER.invoke({"query": brief}, tracer=ctx.tracer)
        return self.Out(draft=str(state.output))


class Review(Node):
    """Have the critic agent review the current draft and decide approve/revise."""

    class In(Schema):
        draft: str

    class Out(Schema):
        critique: str
        approved: bool

    async def run(self, ctx: Context, inp: In) -> Out:
        state = await CRITIC.invoke({"query": inp.draft}, tracer=ctx.tracer)
        verdict = str(state.output)
        return self.Out(critique=verdict, approved=verdict.startswith("APPROVE"))


class Revise(Node):
    """Apply the critique — here, attach the Sources section the critic asked for."""

    class In(Schema):
        draft: str
        sources: list[str]
        revisions: int

    class Out(Schema):
        draft: str
        revisions: int

    async def run(self, ctx: Context, inp: In) -> Out:
        revised = inp.draft + "\n\nSources: " + "; ".join(inp.sources)
        return self.Out(draft=revised, revisions=inp.revisions + 1)


class Publish(Node):
    """Finalize the approved draft into the report."""

    class In(Schema):
        topic: str
        draft: str
        revisions: int

    class Out(Schema):
        report: str

    async def run(self, ctx: Context, inp: In) -> Out:
        header = f"RESEARCH BRIEF — {inp.topic}  (after {inp.revisions} revision(s))\n"
        return self.Out(report=header + "=" * len(header) + "\n" + inp.draft)


def review_route(state: Desk) -> str:
    """Approved (or out of revision budget) → publish; otherwise loop back to revise."""
    if state.approved or state.revisions >= MAX_REVISIONS:
        return "Publish"
    return "Revise"


# --- graders: judge the whole *trajectory*, not just the final text --------------------------


@grader
def publishes_brief(trial: Trial) -> bool:
    """The pipeline reached Publish and produced a titled brief."""
    return "RESEARCH BRIEF" in getattr(trial.output, "report", "")


@grader
def incorporated_revision(trial: Trial) -> bool:
    """The Sources section exists only after Review→Revise ran — proof the loop fired."""
    return "Sources:" in getattr(trial.output, "report", "")


# The orchestration graph — this is what the canvas renders (note the Review ⇄ Revise loop).
desk = (
    Graph(Desk)
    .add(Plan)
    .add(Research)
    .add(Write)
    .add(Review)
    .add(Revise)
    .add(Publish)
    .edge(START, "Plan")
    .edge("Plan", "Research")
    .edge("Research", "Write")
    .edge("Write", "Review")
    .conditional("Review", review_route, {"Publish": "Publish", "Revise": "Revise"})
    .edge("Revise", "Review")
    .edge("Publish", END)
).compile()


async def main() -> None:
    tracer = InMemoryTracer()
    result = await desk.invoke({"topic": "What makes TensorSketch different"}, tracer=tracer)

    print(result.report)
    print("\n" + "-" * 78)
    print("ONE TRACE FOR THE WHOLE HIERARCHY (desk → sub-agents → tools):\n")
    print(tracer.trace.render())
    print(f"\nsummary: {tracer.trace.summary()}")

    # Grade the run the way you'd gate a real agent in CI: check the trajectory (did it search
    # every sub-question? did the revision loop run?), not just the final wording. Same graders you
    # would point at a real model — the trace they read spans the whole agent hierarchy.
    print("\n" + "-" * 78)
    print("EVALUATING THE DESK — trajectory graders over 3 trials:\n")
    suite = Suite(
        "research-desk",
        [
            Case(
                "differentiators",
                {"topic": "What makes TensorSketch different"},
                graders=[
                    publishes_brief,
                    incorporated_revision,
                    ToolCalled("web_search", min_times=4),  # searched every sub-question
                    ToolSequence(["web_search"]),
                ],
                trials=3,
            ),
        ],
    )
    report = await evaluate(desk, suite)
    print(report.render())
    print("per-grader pass rate:", report.grader_breakdown())


if __name__ == "__main__":
    asyncio.run(main())
