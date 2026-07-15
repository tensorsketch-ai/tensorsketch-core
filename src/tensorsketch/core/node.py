"""The `Node` — the unit of work you author.

A node declares a **typed interface** (`In`/`Out` Schemas) and an **opaque body** (`run`). The
interface is what the canvas draws, what the compiler type-checks against the graph's state,
and what natural-language generation targets. The body is arbitrary async code — an LLM call, a
tool invocation, parsing, anything — and TensorSketch never introspects it. That split (typed edges,
opaque bodies) is what lets code and canvas stay in sync without trying to reverse-engineer
control flow from a Turing-complete program.

Author a node by subclassing and declaring nested `In`/`Out` Schemas::

    class Classify(Node):
        class In(Schema):  query: str
        class Out(Schema): intent: Literal["billing", "tech", "other"]

        async def run(self, ctx, inp):
            ...  # opaque: return self.Out(intent=...)

`In` fields name the state channels the node reads; `Out` fields name the channels it writes.
In Phase 0 a node's ports are exactly slices of the graph's typed state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .context import Context
from .schema import Schema


class Node(ABC):
    """Base class for every unit of work in a graph.

    Subclasses declare `In` and `Out` as nested `Schema` classes and implement `run`. The class
    name is the node's default name in a graph (override via `Graph.add(node, name=...)`).
    """

    #: The node's input ports. Set by declaring `class In(Schema): ...` in the subclass.
    In: ClassVar[type[Schema]]
    #: The node's output ports. Set by declaring `class Out(Schema): ...` in the subclass.
    Out: ClassVar[type[Schema]]

    @property
    def name(self) -> str:
        """The node's default name (its class name)."""
        return type(self).__name__

    @abstractmethod
    async def run(self, ctx: Context, inp: Any) -> Any:
        """Execute the node's work.

        Receives the run `Context` and a validated `In` instance; returns an `Out` instance.
        The body is opaque to TensorSketch — do whatever the node needs to do. Raise `Hole(...)` to
        mark
        the node as "needs code" while still declaring its interface.
        """
        raise NotImplementedError
