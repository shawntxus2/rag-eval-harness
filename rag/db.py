"""Postgres connection helper (psycopg3) with pgvector type registration."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import psycopg
from pgvector.psycopg import register_vector

from rag.config import settings


def dsn() -> str:
    return (
        f"host={settings.db_host} port={settings.db_port} dbname={settings.db_name} "
        f"user={settings.db_user} password={settings.db_password}"
    )


@contextlib.contextmanager
def connect(autocommit: bool = True) -> Iterator[psycopg.Connection]:
    """Yield a connection with the pgvector adapter registered.

    Fails loudly if the server is unreachable or the `vector` extension is
    missing — we never want a silent fallback that hides an empty datastore.
    """
    conn = psycopg.connect(dsn(), autocommit=autocommit)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
