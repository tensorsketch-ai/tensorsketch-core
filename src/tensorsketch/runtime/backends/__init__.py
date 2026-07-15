"""Bring-your-own-database backends.

TensorSketch is stateless: nothing durable lives in the framework — checkpoints, the effect journal,
and
the event log all live in a `Backend` you point at your store. `InMemoryBackend` (dev) and
`SqliteBackend` (local file) ship in the core; the connectors here target real databases and are
optional installs, imported lazily so the core never depends on a driver:

    pip install tensorsketch-core[postgres]   # PostgresBackend  (psycopg 3)
    pip install tensorsketch-core[redis]      # RedisBackend     (redis-py)

They all implement the one `Backend` ABC, so switching stores is a one-line change — and writing
a connector for a store we don't ship is just implementing that ABC (see the docs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .sql import SqlBackend, SqliteBackend

if TYPE_CHECKING:
    from .postgres import PostgresBackend
    from .redis import RedisBackend

__all__ = ["PostgresBackend", "RedisBackend", "SqlBackend", "SqliteBackend"]


def __getattr__(name: str) -> object:
    # Lazy-load the driver-backed connectors so importing tensorsketch never imports psycopg/redis.
    if name == "PostgresBackend":
        from .postgres import PostgresBackend

        return PostgresBackend
    if name == "RedisBackend":
        from .redis import RedisBackend

        return RedisBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
