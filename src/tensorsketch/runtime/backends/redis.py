"""Redis backend (redis-py) — fast, shared durable state for distributed runs.

`pip install tensorsketch-core[redis]`. redis is imported lazily, so the core never depends on it.
Point it at
a URL (or hand it an existing client to share a pool):

    from tensorsketch.runtime.backends import RedisBackend

    backend = RedisBackend("redis://localhost:6379/0")
    await app.invoke({...}, thread_id="run-1", backend=backend)

Keys are namespaced `{prefix}:{thread_id}:{checkpoints|effects|events}` — a list of checkpoints,
a hash for the effect journal, and a hash of `seq → event` for the replayable log.
"""

from __future__ import annotations

from typing import Any

from ..checkpoint import Backend, Checkpoint
from ..serialization import PickleSerializer, Serializer
from ..streaming import Event


class RedisBackend(Backend):
    """A `Backend` on Redis via redis-py."""

    def __init__(
        self,
        url: str | None = None,
        *,
        client: Any | None = None,
        prefix: str = "tensorsketch",
        serializer: Serializer | None = None,
    ) -> None:
        if client is None:
            import redis  # lazy: only needed when this backend is actually used

            client = redis.Redis.from_url(url or "redis://localhost:6379/0")
        # Typed Any: redis-py returns bytes by default, but its stubs union bytes|str for the
        # decode_responses case; we always store/read bytes, so we normalize with bytes(...).
        self._r: Any = client
        self._prefix = prefix
        self._ser: Serializer = serializer or PickleSerializer()

    def _key(self, thread_id: str, suffix: str) -> str:
        return f"{self._prefix}:{thread_id}:{suffix}"

    # -- checkpoints ---------------------------------------------------------------------

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._r.rpush(self._key(checkpoint.thread_id, "checkpoints"), self._ser.dumps(checkpoint))

    def latest_checkpoint(self, thread_id: str) -> Checkpoint | None:
        raw = self._r.lindex(self._key(thread_id, "checkpoints"), -1)
        return self._ser.loads(bytes(raw)) if raw is not None else None

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None:
        for checkpoint in self.list_checkpoints(thread_id):
            if checkpoint.id == checkpoint_id:
                return checkpoint
        return None

    def list_checkpoints(self, thread_id: str) -> list[Checkpoint]:
        raws = self._r.lrange(self._key(thread_id, "checkpoints"), 0, -1)
        return [self._ser.loads(bytes(raw)) for raw in raws]

    # -- effect journal ------------------------------------------------------------------

    def record_effect(self, thread_id: str, key: str, result: Any) -> None:
        self._r.hset(self._key(thread_id, "effects"), key, self._ser.dumps(result))

    def lookup_effect(self, thread_id: str, key: str) -> tuple[bool, Any]:
        raw = self._r.hget(self._key(thread_id, "effects"), key)
        if raw is None:
            return False, None
        return True, self._ser.loads(bytes(raw))

    # -- event log -----------------------------------------------------------------------

    def append_event(self, event: Event) -> None:
        self._r.hset(self._key(event.thread_id, "events"), str(event.seq), self._ser.dumps(event))

    def read_events(self, thread_id: str, since: int = 0) -> list[Event]:
        raw = self._r.hgetall(self._key(thread_id, "events"))
        ordered = sorted((int(field), blob) for field, blob in raw.items())
        return [self._ser.loads(bytes(blob)) for seq, blob in ordered if seq >= since]
