"""Serve one TensorSketch agent over three standard protocols — OpenAI, A2A, AG-UI.

Each factory returns a Starlette ASGI app you'd normally run with `uvicorn module:app`. To keep the
example self-contained and offline, it drives the apps in-process with httpx's ASGI transport
instead of binding a port. Requires the serve extra:  uv pip install "tensorsketch-core[serve]".

Run:  uv run python examples/serving.py
"""

from __future__ import annotations

import asyncio
import json

import httpx

from tensorsketch import create_agent
from tensorsketch.messages import Message, assistant
from tensorsketch.providers.fake import FakeProvider
from tensorsketch.serve import a2a_app, a2a_tool, agui_app, openai_app


def demo_agent() -> object:
    # Offline provider so the example runs without an API key; a real one drops in unchanged.
    def policy(messages: list[Message], tools: object) -> Message:
        return assistant("TensorSketch serves this over any protocol.")

    return create_agent(FakeProvider(policy=policy), system="You are a helpful assistant.")


def sse_lines(text: str) -> list[str]:
    return [ln[6:] for ln in text.splitlines() if ln.startswith("data: ")]


async def main() -> None:
    agent = demo_agent()

    # --- OpenAI-compatible: any OpenAI client can POST /v1/chat/completions -----------------
    transport = httpx.ASGITransport(app=openai_app(agent, model="tensorsketch-demo"))
    async with httpx.AsyncClient(transport=transport, base_url="http://openai") as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "tensorsketch-demo", "messages": [{"role": "user", "content": "hi"}]},
        )
        print("OpenAI :", r.json()["choices"][0]["message"]["content"])

    # --- A2A: discover the agent card, then send a message ---------------------------------
    transport = httpx.ASGITransport(app=a2a_app(agent, name="tensorsketch-demo"))
    async with httpx.AsyncClient(transport=transport, base_url="http://a2a") as client:
        card = (await client.get("/.well-known/agent.json")).json()
        print("A2A    : card =", card["name"], "| streaming =", card["capabilities"]["streaming"])
        # ...and consume it from "another agent" via the client-side tool:
        tool = a2a_tool("http://a2a/", client=client)
        print("A2A    :", await tool.run({"message": "hi"}))

    # --- AG-UI: stream a run to a frontend as typed UI events ------------------------------
    transport = httpx.ASGITransport(app=agui_app(agent))
    async with httpx.AsyncClient(transport=transport, base_url="http://agui") as client:
        r = await client.post("/", json={"messages": [{"role": "user", "content": "hi"}]})
        events = [json.loads(d) for d in sse_lines(r.text)]
        print("AG-UI  : events =", [e["type"] for e in events])


if __name__ == "__main__":
    asyncio.run(main())
