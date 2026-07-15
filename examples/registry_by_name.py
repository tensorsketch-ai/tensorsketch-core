"""Pick your model and your database by name — the way a config file would.

`create_provider` / `create_backend` build a built-in from a *string*, so the choice can live in
config or a CLI flag instead of an `import`. Names resolve lazily: nothing optional is imported
until you actually construct it. You can also register your own name in-process (or, in a package,
via a `tensorsketch.providers` entry point) and it's selectable exactly the same way.

Run:  uv run python examples/registry_by_name.py
"""

from __future__ import annotations

import asyncio

from tensorsketch import create_backend, create_provider, register_provider
from tensorsketch.messages import assistant, user
from tensorsketch.providers.base import ChatProvider, Completion
from tensorsketch.registry import backends, providers

# Imagine this came from config.yaml / env / --flags, not source code.
CONFIG = {"provider": "fake", "backend": "sqlite"}


class EchoProvider(ChatProvider):
    """A trivial custom provider, registered under a name of our choosing."""

    async def complete(self, messages, **kwargs) -> Completion:  # type: ignore[no-untyped-def]
        last = messages[-1].content if messages else ""
        return Completion(message=assistant(f"echo: {last}"), model="echo-1")


async def main() -> None:
    # A package would declare `[project.entry-points."tensorsketch.providers"]`; in-process it's one
    # call.
    register_provider("echo", EchoProvider)

    print("providers:", providers.names())  # built-ins + 'echo', none imported to list them
    print("backends: ", backends.names())

    # Build the stack from the config strings — swap the strings, swap the stack.
    provider = create_provider(CONFIG["provider"], script=[assistant("hi from the fake model")])
    store = create_backend(CONFIG["backend"], ":memory:")
    print("\nbuilt:", type(provider).__name__, "+", type(store).__name__)

    reply = await provider.complete([user("anything")])
    print("fake  ->", reply.message.content)

    custom = create_provider("echo")
    reply = await custom.complete([user("hello")])
    print("echo  ->", reply.message.content, f"({reply.model})")


if __name__ == "__main__":
    asyncio.run(main())
