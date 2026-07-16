"""TensorSketch — a code-first, visually-editable, durable agentic framework.

Phase 0 exposes the runtime & type spine: the `Schema` abstraction, typed state `channels`
with reducers, typed `Node`s, and a `Graph` that compiles to a BSP superstep runtime. Import
the common surface straight from the top level::

    from tensorsketch import Schema, Node, Graph, START, END

Agent primitives, providers, the code⇄canvas engine, and durability land in later phases on
top of these foundations.
"""

from .agents import Agent, AgentState, Llm, as_tool, create_agent, generate_structured
from .core import (
    END,
    START,
    BinaryOperatorAggregate,
    Channel,
    ChannelError,
    CompiledGraph,
    Context,
    EmptyChannelError,
    Graph,
    GraphError,
    GraphRecursionError,
    Hole,
    InvalidUpdateError,
    LastValue,
    Node,
    NodeError,
    NodeHandle,
    Reducer,
    Router,
    Schema,
    Send,
    TensorSketchError,
    Topic,
)
from .messages import Message, ToolCall, add_messages
from .middleware import (
    Middleware,
    ModelRequest,
    ObservabilityMiddleware,
    RetryMiddleware,
    ToolRequest,
)
from .observability import (
    FileTracer,
    InMemoryTracer,
    MultiTracer,
    NoopTracer,
    Span,
    Trace,
    Tracer,
    estimate_cost,
)
from .patterns import gather_map, parallel, run_subgraph
from .providers import ChatProvider, Completion, FakeProvider, Usage
from .registry import (
    Registry,
    create_backend,
    create_provider,
    register_backend,
    register_provider,
)
from .runtime import (
    Backend,
    Checkpoint,
    Event,
    InMemoryBackend,
    PickleSerializer,
    Serializer,
    SqliteBackend,
)
from .tools import Tool, tool

__version__ = "0.1.0"

__all__ = [
    "END",
    "START",
    "Agent",
    "AgentState",
    "Backend",
    "BinaryOperatorAggregate",
    "Channel",
    "ChannelError",
    "ChatProvider",
    "Checkpoint",
    "CompiledGraph",
    "Completion",
    "Context",
    "EmptyChannelError",
    "Event",
    "FakeProvider",
    "FileTracer",
    "Graph",
    "GraphError",
    "GraphRecursionError",
    "Hole",
    "InMemoryBackend",
    "InMemoryTracer",
    "InvalidUpdateError",
    "LastValue",
    "Llm",
    "Message",
    "Middleware",
    "ModelRequest",
    "MultiTracer",
    "Node",
    "NodeError",
    "NodeHandle",
    "NoopTracer",
    "ObservabilityMiddleware",
    "PickleSerializer",
    "Reducer",
    "Registry",
    "RetryMiddleware",
    "Router",
    "Schema",
    "Send",
    "Serializer",
    "Span",
    "SqliteBackend",
    "TensorSketchError",
    "Tool",
    "ToolCall",
    "ToolRequest",
    "Topic",
    "Trace",
    "Tracer",
    "Usage",
    "__version__",
    "add_messages",
    "as_tool",
    "create_agent",
    "create_backend",
    "create_provider",
    "estimate_cost",
    "gather_map",
    "generate_structured",
    "parallel",
    "register_backend",
    "register_provider",
    "run_subgraph",
    "tool",
]
