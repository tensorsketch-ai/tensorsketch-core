"""Pluggable serialization for durable state (checkpoints, effect results, events).

A `Backend` has to turn arbitrary run values — channel snapshots, tool results, Pydantic models —
into bytes for its store. That codec is a **seam**, not a fixed choice: the default is
`PickleSerializer` (round-trips any Python object), but a deployment can supply its own — a
signed pickle, a JSON codec for a portable/inspectable store, MessagePack, etc. — by passing a
`serializer=` to any backend.

Pickle executes arbitrary code on load, so only ever point a pickle-backed store at data you
trust. Swapping in a restricted codec is exactly what this seam is for.
"""

from __future__ import annotations

import pickle
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Serializer(Protocol):
    """Turns a run value into bytes and back. Implement this to change how a backend stores data."""

    def dumps(self, obj: Any) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


class PickleSerializer:
    """The default codec — round-trips any picklable Python object (Pydantic models included).

    `protocol` defaults to the highest available. Only load stores you trust: unpickling runs
    code.
    """

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        self._protocol = protocol

    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self._protocol)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)
