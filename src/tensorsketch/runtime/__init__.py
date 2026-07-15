"""L0 — the runtime core.

The BSP/Pregel superstep engine that drives execution. It knows nothing about LLMs, tools, or
the typed `Node`/`Graph` authoring layer — it schedules opaque *processes* over channels. The
typed layer in `tensorsketch.core.graph` compiles down to what lives here, and later phases add the
durable journal, message bus, and streaming behind the same seam.
"""

from .backends import SqliteBackend
from .checkpoint import (
    Backend,
    Checkpoint,
    InMemoryBackend,
    apply,
    capture,
    save_checkpoint,
)
from .engine import Process, execute
from .serialization import PickleSerializer, Serializer
from .streaming import Event, event_stream

__all__ = [
    "Backend",
    "Checkpoint",
    "Event",
    "InMemoryBackend",
    "PickleSerializer",
    "Process",
    "Serializer",
    "SqliteBackend",
    "apply",
    "capture",
    "event_stream",
    "execute",
    "save_checkpoint",
]
