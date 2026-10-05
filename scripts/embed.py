#!/usr/bin/env python3
"""Phase 2 — embed every chunk with each supported model, store normalized.

For each model in EMBEDDING_MODELS, embeds all chunks whose column is still NULL
(so the script is resumable), writing L2-normalized vectors into that model's
column. Precision is settings.torch_dtype (fp16). After writing, VERIFIES the
normalization actually happened by sampling stored vectors from each column and
confirming the L2 norm is 1.0 within tolerance — a silent normalization failure
would degrade every cosine result with no error.

Passages are embedded with no instruction prefix; bge-v1.5's query instruction is
a Phase 3 (retrieval) concern, applied to queries only.

Usage:  python scripts/embed.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rag.config import EMBEDDING_MODELS, settings
from rag.db import connect
from rag.embed import embed, load_model, resolve_device

WRITE_BATCH = 1000   # DB rows per round-trip
NORM_TOL = 1e-3
NORM_SAMPLE = 300


def _embed_model(conn, model_name: str, column: str) -> int:
    rows = conn.execute(
        f"SELECT id, text FROM chunks WHERE {column} IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        print(f"  {model_name}: nothing to do ({column} already populated)")
        return 0
    model = load_model(model_name)
    done = 0
    for i in range(0, len(rows), WRITE_BATCH):
        batch = rows[i:i + WRITE_BATCH]
        vecs = embed(model, [t for _, t in batch], settings.embed_batch_size)
        with conn.cursor() as cur:
            cur.executemany(
                f"UPDATE chunks SET {column}=%s WHERE id=%s",
                [(v.astype(np.float32), cid) for (cid, _), v in zip(batch, vecs)],
            )
        done += len(batch)
        print(f"  {model_name}: {done}/{len(rows)}", end="\r", flush=True)
    print(f"  {model_name}: embedded {done} chunks into {column}" + " " * 20)
    return done


def _verify_norms(conn, model_name: str, column: str) -> bool:
    rows = conn.execute(
        f"SELECT id, {column} FROM chunks WHERE {column} IS NOT NULL "
        f"ORDER BY id LIMIT {NORM_SAMPLE}"
    ).fetchall()
    if not rows:
        print(f"  {model_name}: NO vectors to verify")
        return False
    norms = [float(np.linalg.norm(np.asarray(v, dtype=np.float32))) for _, v in rows]
    arr = np.array(norms)
    worst = max(abs(arr - 1.0))
    ok = worst <= NORM_TOL
    print(f"  {model_name}: {len(rows)} sampled | L2 norm min={arr.min():.6f} "
          f"max={arr.max():.6f} | max|norm-1|={worst:.2e} | {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print(f"Phase 2 embedding | device={resolve_device()} | dtype={settings.torch_dtype} "
          f"| batch={settings.embed_batch_size}\n")
    with connect() as conn:
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"chunks in corpus: {total}\n")
        for model_name, spec in EMBEDDING_MODELS.items():
            _embed_model(conn, model_name, spec["column"])
        print("\nNORMALIZATION VERIFICATION (sampled from stored vectors):")
        all_ok = True
        for model_name, spec in EMBEDDING_MODELS.items():
            all_ok &= _verify_norms(conn, model_name, spec["column"])
        # mark runs embedded
        conn.execute("UPDATE ingestion_runs SET status='ready'")
        print("\nALL NORMS OK" if all_ok else "\nNORM CHECK FAILED — investigate before Phase 3")
        return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
