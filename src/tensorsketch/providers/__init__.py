"""Model providers. The core ships the interface and a fake; real providers are optional installs.

Built-in real providers (import from their submodule; each needs its extra):

    from tensorsketch.providers.anthropic import AnthropicProvider   # pip install
    tensorsketch-core[anthropic]
    from tensorsketch.providers.openai import OpenAIProvider         # pip install
    tensorsketch-core[openai]
    from tensorsketch.providers.google import GoogleProvider         # pip install
    tensorsketch-core[google]

Any other backend is a ~30-line custom `ChatProvider` — see the providers guide.
"""

from .base import ChatProvider, Completion, Usage
from .fake import FakeProvider

__all__ = ["ChatProvider", "Completion", "FakeProvider", "Usage"]
