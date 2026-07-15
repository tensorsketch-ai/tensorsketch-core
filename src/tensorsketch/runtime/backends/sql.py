"""SQL backends — one implementation over any DB-API connection, two thin dialects.

`SqlBackend` implements the whole `Backend` contract (checkpoints, effect journal, event log) in
terms of a DB-API 2.0 connection (`.execute(sql, params)` / `.commit()` / cursors). SQLite and
Postgres differ only in a few tokens — the value placeholder, the auto-increment column, the blob
type, and the upsert clause — so each is a handful of overrides, not a reimplementation.

Three tables per store, all keyed by `thread_id`:

* ``checkpoints(seq, thread_id, id, blob)`` — append-only; `seq` orders them, `id` finds one.
* ``effects(thread_id, key, blob)`` — the exactly-once journal; ``PRIMARY KEY (thread_id, key)``.
* ``events(thread_id, seq, blob)`` — the replayable event log; ``PRIMARY KEY (thread_id, seq)``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..checkpoint import Backend, Checkpoint
from ..serialization import PickleSerializer, Serializer
from ..streaming import Event


class SqlBackend(Backend):
    """A `Backend` over a DB-API connection. Subclasses set the dialect tokens and connect."""

    #: Value placeholder in prepared statements (`?` for SQLite, `%s` for Postgres).
    _ph = "?"
    #: Column definition for the auto-incrementing checkpoint ordering key.
    _autopk = "INTEGER PRIMARY KEY AUTOINCREMENT"
    #: Binary column type.
    _blob = "BLOB"

    _conn: Any

    def __init__(self, serializer: Serializer | None = None) -> None:
        self._ser: Serializer = serializer or PickleSerializer()
        self._init_schema()

    # -- dialect hooks -------------------------------------------------------------------

    def _upsert(self, table: str, columns: tuple[str, ...], conflict: str) -> str:
        """An INSERT that overwrites the blob on key conflict (dialect-specific)."""
        raise NotImplementedError

    # -- schema --------------------------------------------------------------------------

    def _init_schema(self) -> None:
        cur = self._conn
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS checkpoints "
            f"(seq {self._autopk}, thread_id TEXT, id TEXT, blob {self._blob})"
        )
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS effects "
            f"(thread_id TEXT, key TEXT, blob {self._blob}, PRIMARY KEY (thread_id, key))"
        )
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS events "
            f"(thread_id TEXT, seq BIGINT, blob {self._blob}, PRIMARY KEY (thread_id, seq))"
        )
        self._conn.commit()

    # -- checkpoints ---------------------------------------------------------------------

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        p = self._ph
        self._conn.execute(
            f"INSERT INTO checkpoints (thread_id, id, blob) VALUES ({p}, {p}, {p})",
            (checkpoint.thread_id, checkpoint.id, self._ser.dumps(checkpoint)),
        )
        self._conn.commit()

    def latest_checkpoint(self, thread_id: str) -> Checkpoint | None:
        row = self._conn.execute(
            f"SELECT blob FROM checkpoints WHERE thread_id = {self._ph} ORDER BY seq DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        return self._load_checkpoint(row)

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> Checkpoint | None:
        p = self._ph
        row = self._conn.execute(
            f"SELECT blob FROM checkpoints WHERE thread_id = {p} AND id = {p} LIMIT 1",
            (thread_id, checkpoint_id),
        ).fetchone()
        return self._load_checkpoint(row)

    def list_checkpoints(self, thread_id: str) -> list[Checkpoint]:
        rows = self._conn.execute(
            f"SELECT blob FROM checkpoints WHERE thread_id = {self._ph} ORDER BY seq ASC",
            (thread_id,),
        ).fetchall()
        return [self._ser.loads(bytes(row[0])) for row in rows]

    # -- effect journal ------------------------------------------------------------------

    def record_effect(self, thread_id: str, key: str, result: Any) -> None:
        self._conn.execute(
            self._upsert("effects", ("thread_id", "key", "blob"), "thread_id, key"),
            (thread_id, key, self._ser.dumps(result)),
        )
        self._conn.commit()

    def lookup_effect(self, thread_id: str, key: str) -> tuple[bool, Any]:
        p = self._ph
        row = self._conn.execute(
            f"SELECT blob FROM effects WHERE thread_id = {p} AND key = {p}",
            (thread_id, key),
        ).fetchone()
        if row is None:
            return False, None
        return True, self._ser.loads(bytes(row[0]))

    # -- event log -----------------------------------------------------------------------

    def append_event(self, event: Event) -> None:
        self._conn.execute(
            self._upsert("events", ("thread_id", "seq", "blob"), "thread_id, seq"),
            (event.thread_id, event.seq, self._ser.dumps(event)),
        )
        self._conn.commit()

    def read_events(self, thread_id: str, since: int = 0) -> list[Event]:
        p = self._ph
        rows = self._conn.execute(
            f"SELECT blob FROM events WHERE thread_id = {p} AND seq >= {p} ORDER BY seq ASC",
            (thread_id, since),
        ).fetchall()
        return [self._ser.loads(bytes(row[0])) for row in rows]

    # -- lifecycle -----------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def _load_checkpoint(self, row: tuple[Any, ...] | None) -> Checkpoint | None:
        if row is None:
            return None
        loaded: Checkpoint = self._ser.loads(bytes(row[0]))
        return loaded


class SqliteBackend(SqlBackend):
    """Persistent backend backed by SQLite. `path=":memory:"` for an ephemeral DB.

    Good for local persistence and single-writer use. For multi-writer production, reach for
    `PostgresBackend`. Values are stored via the `serializer` (pickle by default) — only open a
    database you trust.
    """

    _ph = "?"
    _autopk = "INTEGER PRIMARY KEY AUTOINCREMENT"
    _blob = "BLOB"

    def __init__(self, path: str = ":memory:", *, serializer: Serializer | None = None) -> None:
        self._conn = sqlite3.connect(path)
        super().__init__(serializer)

    def _upsert(self, table: str, columns: tuple[str, ...], conflict: str) -> str:
        cols = ", ".join(columns)
        phs = ", ".join([self._ph] * len(columns))
        return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({phs})"
