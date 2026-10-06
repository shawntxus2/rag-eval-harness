#!/usr/bin/env python3
"""Weighted-RRF sweep — where does fusion stop losing to vector on descriptive?

Equal-weight RRF imports the weak lexical arm's errors. This sweeps the vector
weight (lexical fixed at 1.0) in score = Σ w_i/(k+rank_i) and reports hybrid
recall@5 at each weight against plain vector, per chunk size, on descriptive
questions. The crossover weight is a measured result, not an excuse.

Vector and lexical candidate lists are retrieved once per question; only the fusion
is recomputed per weight.

Usage:  python scripts/rrf_sweep.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.config import settings
from rag.db import connect
from rag.retrieval import Retriever, _rrf

WEIGHTS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]  # vector weight; lexical fixed at 1.0


def hit5(ranked, meta, gdoc, gs, ge):
    for h in ranked[:5]:
        m = meta.get(h.chunk_id)
        if m and m[0] == gdoc and m[1] <= gs and m[2] >= ge:
            return 1
    return 0


def main() -> int:
    r = Retriever()
    with connect() as conn:
        r.configure_session(conn)
        qs = conn.execute(
            "SELECT id, question, source_document_id, gold_char_start, gold_char_end "
            "FROM eval_questions WHERE generated_by='openai' AND status='candidate' "
            "AND question_type='descriptive' ORDER BY id").fetchall()
        runs = conn.execute("SELECT id, chunk_size FROM ingestion_runs ORDER BY chunk_size").fetchall()
        n = len(qs)
        print(f"WEIGHTED-RRF SWEEP on DESCRIPTIVE (N={n}); recall@5 as hits/N; lexical weight=1.0")
        print(f"{'size':>5} {'vector':>9} " + " ".join(f"w={w:<4}" for w in WEIGHTS))
        for rid, size in runs:
            vrec, per_w = 0, {w: 0 for w in WEIGHTS}
            for qid, q, gdoc, gs, ge in qs:
                vec = r.vector(conn, q, rid)
                lex = r.lexical(conn, q, rid)
                ids = {h.chunk_id for h in vec} | {h.chunk_id for h in lex}
                meta = {cid: (d, cs, ce) for cid, d, cs, ce in conn.execute(
                    "SELECT id, document_id, char_start, char_end FROM chunks WHERE id = ANY(%s)", (list(ids),)).fetchall()}
                vrec += hit5(vec, meta, gdoc, gs, ge)
                for w in WEIGHTS:
                    fused = _rrf([vec, lex], settings.rrf_k, [w, 1.0])
                    per_w[w] += hit5(fused, meta, gdoc, gs, ge)
            cells = []
            cross = None
            for w in WEIGHTS:
                mark = "*" if per_w[w] >= vrec else " "
                if per_w[w] >= vrec and cross is None:
                    cross = w
                cells.append(f"{per_w[w]/n:.2f}{mark}")
            print(f"{size:>5} {f'{vrec/n:.2f} ({vrec}/{n})':>9} " + " ".join(f"{c:<6}" for c in cells)
                  + f"  crossover(hybrid>=vector): {'w='+str(cross) if cross else 'none<=8'}")
        print("\n* = hybrid recall@5 at that vector weight meets or beats plain vector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
