#!/usr/bin/env python3
"""Apply db/schema.sql (idempotent) and print the resulting schema.

Usage:  python scripts/init_db.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg

from rag.db import dsn

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def main() -> int:
    sql = SCHEMA.read_text()
    try:
        with psycopg.connect(dsn(), autocommit=True) as conn:
            conn.execute(sql)
            print(f"Applied {SCHEMA}")
            _report(conn)
    except psycopg.OperationalError as e:
        print(f"ERROR: could not connect to Postgres ({e}).", file=sys.stderr)
        print("Is the container up?  docker compose up -d", file=sys.stderr)
        return 1
    return 0


def _report(conn: psycopg.Connection) -> None:
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name"
        ).fetchall()
    ]
    print(f"\nTables ({len(tables)}): {', '.join(tables)}\n")
    for t in tables:
        print(f"── {t}")
        cols = conn.execute(
            "SELECT column_name, data_type, udt_name, is_nullable "
            "FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position",
            (t,),
        ).fetchall()
        for name, dtype, udt, nullable in cols:
            shown = udt if dtype == "USER-DEFINED" else dtype
            null = "" if nullable == "YES" else " NOT NULL"
            print(f"     {name:<20} {shown}{null}")
        idx = conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename=%s ORDER BY indexname", (t,)
        ).fetchall()
        for (iname,) in idx:
            print(f"     idx: {iname}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
