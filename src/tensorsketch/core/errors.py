"""The TensorSketch exception hierarchy.

Every error TensorSketch raises is a `TensorSketchError`, so an embedding application can catch the
whole
family with one `except`. Errors are meant to *teach*: messages should say what was expected,
what was found, and (where possible) what to do about it.
"""

from __future__ import annotations


class TensorSketchError(Exception):
    """Base class for every error TensorSketch raises."""


class GraphError(TensorSketchError):
    """The graph is structurally invalid.

    Raised at build/compile time for problems like an unknown node in an edge, a node that
    reads or writes a state field that does not exist, or a port whose type is incompatible
    with the state channel it maps to. These are author mistakes caught before any run.
    """


class GraphRecursionError(TensorSketchError):
    """The scheduler ran more supersteps than its budget without halting.

    Almost always an unguarded loop (an edge cycles forever because its exit condition is
    never met). Raise the limit with `invoke(..., max_steps=...)` only once you are sure the
    loop terminates.
    """


class ChannelError(TensorSketchError):
    """An illegal operation on a state channel."""


class EmptyChannelError(ChannelError):
    """Read from a channel that has never been written."""


class InvalidUpdateError(ChannelError):
    """A channel received an update it cannot reduce.

    The common case: two nodes wrote the same `LastValue` channel in a single superstep, so
    the result would be order-dependent. Give that field a reducer (e.g. `Reducer(add)`) if
    concurrent writes are intended.
    """


class NodeError(TensorSketchError):
    """A node's `run` raised while executing."""


class Hole(TensorSketchError):
    """Raised by an unimplemented node body — "this node still needs code".

    A node's typed interface (`In`/`Out`) can be fully declared while its body is left as a
    stub that raises `Hole`. The interface round-trips to the canvas and the graph type-checks;
    the tooling greps for `Hole(...)` to surface "N nodes need code". Later, a natural-language
    description attached here can be compiled to a real body against the same typed contract.

    Example::

        class BillingAgent(Node):
            class In(Schema):  query: str
            class Out(Schema): answer: str
            async def run(self, ctx, inp):
                raise Hole("Answer billing questions using the KB tool")
    """

    def __init__(self, spec: str = "") -> None:
        self.spec = spec
        super().__init__(spec or "this node needs code")
