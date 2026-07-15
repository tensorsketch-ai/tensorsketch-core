"""Postgres backend (psycopg 3) — durable, multi-writer state for production runs.

`pip install tensorsketch-core[postgres]`. psycopg is imported lazily, so the core never depends on
it. Point
it at a DSN (or hand it an existing connection to share a pool / transaction):

    from tensorsketch.runtime.backends import PostgresBackend

    backend = PostgresBackend("postgresql://user:pass@host/db")
    await app.invoke({...}, thread_id="run-1", backend=backend)

The schema (three `thread_id`-keyed tables) is created on first use. Everything else is inherited
from `SqlBackend`; only the dialect tokens and the connection differ.
"""

from __future__ import annotations

from typing import Any

from ..serialization import Serializer
from .sql import SqlBackend


class PostgresBackend(SqlBackend):
    """A `Backend` on Postgres via psycopg 3."""

    _ph = "%s"
    _autopk = "BIGSERIAL PRIMARY KEY"
    _blob = "BYTEA"

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection: Any | None = None,
        serializer: Serializer | None = None,
    ) -> None:
        if connection is None:
            import psycopg  # lazy: only needed when this backend is actually used

            if dsn is None:
                raise ValueError("PostgresBackend needs a dsn or an existing connection")
            connection = psycopg.connect(dsn, autocommit=False)
        self._conn = connection
        super().__init__(serializer)

    def _upsert(self, table: str, columns: tuple[str, ...], conflict: str) -> str:
        cols = ", ".join(columns)
        phs = ", ".join([self._ph] * len(columns))
        return (
            f"INSERT INTO {table} ({cols}) VALUES ({phs}) "
            f"ON CONFLICT ({conflict}) DO UPDATE SET blob = EXCLUDED.blob"
        )
