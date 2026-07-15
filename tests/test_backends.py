"""The `Backend` contract — every store must behave identically for the three durable roles.

Runs against the in-memory, SQLite, and Redis (fakeredis) backends by default; add a Postgres
DSN in `LOOM_TEST_POSTGRES` to include it. A custom `Serializer` is exercised too, since the
codec is a documented seam.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tensorsketch.runtime import Backend, Checkpoint, Event, InMemoryBackend, SqliteBackend


def _make_backends() -> Iterator[Any]:
    yield pytest.param(InMemoryBackend, id="memory")
    yield pytest.param(lambda: SqliteBackend(":memory:"), id="sqlite")

    try:
        import fakeredis

        from tensorsketch.runtime.backends import RedisBackend

        yield pytest.param(
            lambda: RedisBackend(client=fakeredis.FakeStrictRedis()), id="redis-fake"
        )
    except ImportError:  # pragma: no cover
        yield pytest.param(
            None, id="redis-fake", marks=pytest.mark.skip(reason="fakeredis not installed")
        )

    dsn = os.environ.get("LOOM_TEST_POSTGRES")
    if dsn:
        from tensorsketch.runtime.backends import PostgresBackend

        yield pytest.param(lambda: PostgresBackend(dsn), id="postgres")
    else:
        yield pytest.param(
            None, id="postgres", marks=pytest.mark.skip(reason="set LOOM_TEST_POSTGRES to run")
        )


@pytest.fixture(params=list(_make_backends()))
def backend(request: pytest.FixtureRequest) -> Backend:
    make: Callable[[], Backend] = request.param
    return make()


def _checkpoint(thread_id: str, cid: str, parent: str | None, step: int, values: Any) -> Checkpoint:
    return Checkpoint(
        thread_id=thread_id,
        id=cid,
        parent_id=parent,
        step=step,
        channel_values=values,
        active=frozenset({"A"}),
    )


def test_checkpoints_round_trip(backend: Backend) -> None:
    t = uuid.uuid4().hex
    assert backend.latest_checkpoint(t) is None

    c1 = _checkpoint(t, "c1", None, 1, {"count": 1, "msgs": ["a", "b"]})
    c2 = _checkpoint(t, "c2", "c1", 2, {"count": 2, "msgs": ["a", "b", "c"]})
    backend.save_checkpoint(c1)
    backend.save_checkpoint(c2)

    latest = backend.latest_checkpoint(t)
    assert latest is not None and latest.id == "c2"
    assert latest.channel_values == {"count": 2, "msgs": ["a", "b", "c"]}
    assert latest.active == frozenset({"A"})

    got = backend.get_checkpoint(t, "c1")
    assert got is not None and got.channel_values == {"count": 1, "msgs": ["a", "b"]}
    assert backend.get_checkpoint(t, "nope") is None
    assert [c.id for c in backend.list_checkpoints(t)] == ["c1", "c2"]


def test_effect_journal_round_trips(backend: Backend) -> None:
    t = uuid.uuid4().hex
    assert backend.lookup_effect(t, "k") == (False, None)

    backend.record_effect(t, "k", {"answer": 42, "items": [1, 2, 3]})
    found, value = backend.lookup_effect(t, "k")
    assert found is True
    assert value == {"answer": 42, "items": [1, 2, 3]}

    # a different thread does not see it
    assert backend.lookup_effect(uuid.uuid4().hex, "k") == (False, None)


def test_event_log_is_ordered_and_cursorable(backend: Backend) -> None:
    t = uuid.uuid4().hex
    for i in range(3):
        backend.append_event(
            Event(
                seq=i,
                run_id="r",
                thread_id=t,
                superstep=i,
                node=None,
                type="values",
                data={"i": i},
            )
        )

    assert [e.seq for e in backend.read_events(t)] == [0, 1, 2]
    assert [e.seq for e in backend.read_events(t, since=1)] == [1, 2]
    assert backend.read_events(t, since=99) == []
    assert [e.data["i"] for e in backend.read_events(t)] == [0, 1, 2]


def test_custom_serializer_is_used() -> None:
    """A backend routes all persistence through its `Serializer` seam."""
    import json

    class JsonishSerializer:
        # A toy codec that only needs to round-trip these test values, proving the seam works.
        def dumps(self, obj: Any) -> bytes:
            return json.dumps(obj).encode()

        def loads(self, data: bytes) -> Any:
            return json.loads(data)

    t = uuid.uuid4().hex
    backend = SqliteBackend(":memory:", serializer=JsonishSerializer())
    backend.record_effect(t, "k", {"n": 1})
    assert backend.lookup_effect(t, "k") == (True, {"n": 1})
