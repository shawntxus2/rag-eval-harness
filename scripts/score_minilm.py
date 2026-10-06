#!/usr/bin/env python3
"""Scoped MiniLM vs bge-base comparison — two winning configs only.

Answers "is 384-dim good enough?" without re-running the full matrix (which would
add ~60 tests to the multiple-comparisons pile). Scores all-MiniLM-L6-v2 at exactly:
  - cited, vector+rerank, 250-token chunks   (bge's cited winner, .73)
  - descriptive, vector, 500-token chunks     (bge's descriptive winner, .41)
and compares to bge-base at the same two cells, paired McNemar on hit@5.

Usage:  python scripts/score_minilm.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from psycopg.types.json import Json
from scipy.stats import binomtest

from rag.config import settings
from rag.db import connect
from rag.retrieval import Retriever

BGE = "BAAI/bge-base-en-v1.5"
MINI = "sentence-transformers/all-MiniLM-L6-v2"
CONFIGS = [  # (question_type, mode, chunk_size)
    ("cited", "vector_rerank", 250),
    ("descriptive", "vector", 500),
]


def hits_for(r, conn, qs, mode, rid):
    out = []
    for qid, q, gdoc, gs, ge in qs:
        ranked = getattr(r, mode)(conn, q, rid)
        ids = [h.chunk_id for h in ranked]
        meta = {cid: (d, cs, ce) for cid, d, cs, ce in conn.execute(
            "SELECT id, document_id, char_start, char_end FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()} if ids else {}
        rank = next((i for i, cid in enumerate(ids, 1)
                     if meta.get(cid) and meta[cid][0] == gdoc and meta[cid][1] <= gs and meta[cid][2] >= ge), None)
        out.append((qid, rank))
    return out


def main() -> int:
    bge, mini = Retriever(BGE), Retriever(MINI)
    with connect() as conn:
        bge.configure_session(conn)
        runs = {cs: rid for rid, cs in conn.execute("SELECT id, chunk_size FROM ingestion_runs").fetchall()}
        print("MiniLM (384) vs bge-base (768) at the two winning configs; recall as hits/N\n")
        for qtype, mode, size in CONFIGS:
            qs = conn.execute(
                "SELECT id, question, source_document_id, gold_char_start, gold_char_end "
                "FROM eval_questions WHERE generated_by='openai' AND status='candidate' AND question_type=%s ORDER BY id",
                (qtype,)).fetchall()
            n = len(qs)
            rid = runs[size]
            res = {}
            for tag, r in (("bge", bge), ("minilm", mini)):
                ranks = hits_for(r, conn, qs, mode, rid)
                h5 = [1 if (rk and rk <= 5) else 0 for _, rk in ranks]
                h10 = [1 if (rk and rk <= 10) else 0 for _, rk in ranks]
                mrr = sum(1.0 / rk for _, rk in ranks if rk) / n
                res[tag] = (h5, h10, mrr)
                if tag == "minilm":
                    conn.execute(
                        """INSERT INTO eval_runs (retrieval_mode, chunk_size, ingestion_run_id, embedding_model,
                               rerank_model, rrf_k, top_k, correctness_rule, dtype, question_type, num_questions,
                               recall_at_5, recall_at_10, mrr, params)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (mode, size, rid, MINI, settings.rerank_model if "rerank" in mode else None,
                         settings.rrf_k, settings.top_k, settings.correctness_rule, settings.torch_dtype, qtype, n,
                         sum(h5)/n, sum(h10)/n, mrr, Json({"scoped_minilm": True})))
            (b5, b10, bmrr), (m5, m10, mmrr) = res["bge"], res["minilm"]
            bb = sum(1 for x, y in zip(b5, m5) if x == 1 and y == 0)
            cc = sum(1 for x, y in zip(b5, m5) if x == 0 and y == 1)
            p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if (bb + cc) else 1.0
            print(f"{qtype} / {mode} @ {size}  (N={n})")
            print(f"  bge    recall@5 {sum(b5)/n:.2f} ({sum(b5)}/{n})  @10 {sum(b10)/n:.2f}  MRR {bmrr:.3f}")
            print(f"  minilm recall@5 {sum(m5)/n:.2f} ({sum(m5)}/{n})  @10 {sum(m10)/n:.2f}  MRR {mmrr:.3f}")
            print(f"  McNemar hit@5 bge vs minilm: b={bb} c={cc} p={p:.3f} {'(differ)' if p<0.05 else '(no sig. difference)'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
