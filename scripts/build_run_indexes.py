#!/usr/bin/env python3
"""Build per-(ingestion_run × model) partial HNSW indexes.

The schema's indexes span all runs, so a within-run query filters a shared index
and its latency depends on run selectivity (a confound — the 1000-token run holds
14% of chunks and would be slowest under iterative scan, for reasons unrelated to
chunk size). One partial index per run per model lets each query hit an index
holding only its own run, so iterative scan can be dropped. Six indexes over
~17.5k rows build in seconds. Run after scripts/embed.py.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.config import EMBEDDING_MODELS
from rag.db import connect


def main() -> int:
    with connect() as conn:
        runs = conn.execute("SELECT id, chunk_size FROM ingestion_runs ORDER BY chunk_size").fetchall()
        for rid, size in runs:
            for spec in EMBEDDING_MODELS.values():
                col = spec["column"]
                tag = col.replace("embedding_", "")
                name = f"chunks_{tag}_hnsw_run{size}"
                t = time.perf_counter()
                # rid is our own integer; inlining it makes the index predicate a
                # literal the planner can match against a literal-filtered query.
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {name} ON chunks USING hnsw "
                    f"({col} vector_cosine_ops) "
                    f"WHERE ingestion_run_id = {int(rid)} AND {col} IS NOT NULL"
                )
                dt = time.perf_counter() - t
                n = conn.execute(
                    f"SELECT count(*) FROM chunks WHERE ingestion_run_id={int(rid)} AND {col} IS NOT NULL"
                ).fetchone()[0]
                print(f"  {name:30} rows={n:<6} built in {dt:.2f}s")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
