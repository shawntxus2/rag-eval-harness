#!/usr/bin/env python3
"""Phase 4 scoring — the deliverable table, split by question_type, never pooled.

Retrieval is corpus-wide (all chunks in the run). A chunk HITS iff it is from the
gold document AND fully contains [gold_char_start, gold_char_end]. For every
embedding_model × question_type × chunk_size × retrieval_mode we record recall@5,
recall@10, MRR, mean tokens across the top-5 chunks (names the large-chunk bias),
and retrieval-step latency p50/p95 (query embedding timed separately). A per-run
containment CEILING (fraction of questions some chunk in the run can contain) bounds
recall. Significance via McNemar's exact test on paired hit@5. Hits/total are shown
with every rate.

Usage:  python scripts/score.py
"""
from __future__ import annotations

import pathlib
import statistics as st
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from psycopg.types.json import Json
from scipy.stats import binomtest

from rag.config import settings
from rag.db import connect
from rag.retrieval import Retriever

MODES = ["vector", "lexical", "hybrid", "hybrid_rerank"]
TYPES = ["cited", "descriptive"]
PAIRS = [("lexical", "vector"), ("hybrid", "vector"), ("hybrid", "lexical"), ("hybrid_rerank", "hybrid")]


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def main() -> int:
    r = Retriever()  # bge-base, reranker loaded once
    emb_model = r.embedding_model
    with connect() as conn:
        r.configure_session(conn)
        conn.execute("DELETE FROM eval_results")
        conn.execute("DELETE FROM eval_runs")
        runs = conn.execute("SELECT id, chunk_size FROM ingestion_runs ORDER BY chunk_size").fetchall()
        qbytype = {t: conn.execute(
            "SELECT id, question, source_document_id, gold_char_start, gold_char_end "
            "FROM eval_questions WHERE generated_by='openai' AND status='candidate' AND question_type=%s "
            "ORDER BY id", (t,)).fetchall() for t in TYPES}

        hit5 = {}   # (type,size,mode) -> list[0/1] aligned to question order
        table = []  # rows for printing
        embed_ms_all = []
        for t in TYPES:
            qs = qbytype[t]
            for rid, size in runs:
                # containment ceiling for this (type,size): any chunk in run+gold doc contains span
                ceil = 0
                for _, _, gdoc, gs, ge in qs:
                    if conn.execute("SELECT 1 FROM chunks WHERE ingestion_run_id=%s AND document_id=%s "
                                    "AND char_start<=%s AND char_end>=%s LIMIT 1", (rid, gdoc, gs, ge)).fetchone():
                        ceil += 1
                ceiling = ceil / len(qs)
                for mode in MODES:
                    fn = getattr(r, mode)
                    fn(conn, qs[0][1], rid)  # warmup (discarded)
                    hits_vec, rrs, h10, tok5, retr_ms = [], [], [], [], []
                    results = []
                    for qid, q, gdoc, gs, ge in qs:
                        t0 = time.perf_counter(); ranked = fn(conn, q, rid); total = time.perf_counter() - t0
                        if mode == "lexical":
                            emb = 0.0
                        else:
                            t1 = time.perf_counter(); r.embed_query(q); emb = time.perf_counter() - t1
                            embed_ms_all.append(emb * 1000)
                        retr_ms.append(max(0.0, (total - emb)) * 1000)
                        ids = [h.chunk_id for h in ranked]
                        meta = {}
                        if ids:
                            for cid, d, cs, ce, tc in conn.execute(
                                "SELECT id, document_id, char_start, char_end, token_count FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall():
                                meta[cid] = (d, cs, ce, tc)
                        rank = None
                        for i, cid in enumerate(ids, 1):
                            m = meta.get(cid)
                            if m and m[0] == gdoc and m[1] <= gs and m[2] >= ge:
                                rank = i; break
                        h5 = 1 if (rank and rank <= 5) else 0
                        hits_vec.append(h5)
                        h10.append(1 if (rank and rank <= 10) else 0)
                        rrs.append(1.0 / rank if rank else 0.0)
                        tok5.append(sum(meta.get(c, (0, 0, 0, 0))[3] for c in ids[:5]))
                        results.append((qid, rank, h5, (1 if (rank and rank <= 10) else 0), (1.0/rank if rank else 0.0), retr_ms[-1]))
                    recall5 = sum(hits_vec) / len(qs)
                    recall10 = sum(h10) / len(qs)
                    mrr = sum(rrs) / len(qs)
                    mt5 = st.mean(tok5)
                    p50, p95 = pctl(retr_ms, .5), pctl(retr_ms, .95)
                    hit5[(t, size, mode)] = hits_vec
                    table.append((t, size, mode, recall5, sum(hits_vec), recall10, sum(h10), mrr, mt5, p50, p95, ceiling, len(qs)))
                    rk = conn.execute(
                        """INSERT INTO eval_runs (retrieval_mode, chunk_size, ingestion_run_id, embedding_model,
                               rerank_model, rrf_k, top_k, correctness_rule, dtype, question_type, mean_tokens_at_5,
                               num_questions, recall_at_5, recall_at_10, mrr, latency_p50_ms, latency_p95_ms, params)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (mode, size, rid, emb_model, settings.rerank_model if mode == "hybrid_rerank" else None,
                         settings.rrf_k, settings.top_k, settings.correctness_rule, settings.torch_dtype, t, mt5,
                         len(qs), recall5, recall10, mrr, p50, p95,
                         Json({"depth": settings.retrieval_depth, "instruction_on": bool(r.query_instruction)}))).fetchone()[0]
                    with conn.cursor() as cur:
                        cur.executemany(
                            "INSERT INTO eval_results (eval_run_id, question_id, gold_rank, hit_at_5, hit_at_10, reciprocal_rank, retrieval_ms) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            [(rk, qid, rank, bool(h5), bool(h10v), rr, ms) for qid, rank, h5, h10v, rr, ms in results])

        _print_table(table)
        _print_mcnemar(hit5)
        print(f"\nquery-embed latency (separate, ~constant): mean {st.mean(embed_ms_all):.2f} ms "
              f"(fp16, bge). Retrieval-step latency above EXCLUDES it.")
    return 0


def _print_table(table) -> None:
    print("\nDELIVERABLE TABLE (embedding=bge-base, dtype=fp16). recall shown as rate (hits/N).")
    for t in TYPES:
        rows = [x for x in table if x[0] == t]
        n = rows[0][12]
        print(f"\n=== {t.upper()}  (N={n}) ===")
        print(f"{'size':>5} {'mode':<14}{'recall@5':>14}{'recall@10':>14}{'MRR':>7}{'tok@5':>8}{'p50ms':>8}{'p95ms':>8}{'ceil':>7}")
        for _, size, mode, r5, h5, r10, h10, mrr, mt5, p50, p95, ceil, _n in rows:
            print(f"{size:>5} {mode:<14}{f'{r5:.2f} ({h5}/{_n})':>14}{f'{r10:.2f} ({h10}/{_n})':>14}"
                  f"{mrr:>7.3f}{mt5:>8.0f}{p50:>8.2f}{p95:>8.2f}{ceil:>7.2f}")


def _print_mcnemar(hit5) -> None:
    print("\nMcNEMAR (exact, paired hit@5). b=A-hit/B-miss, c=A-miss/B-hit; p two-sided; * = p<0.05")
    for t in TYPES:
        for size in sorted({k[1] for k in hit5 if k[0] == t}):
            print(f"\n  {t} / chunk {size}:")
            for a, b in PAIRS:
                A, B = hit5.get((t, size, a)), hit5.get((t, size, b))
                if not A or not B:
                    continue
                bb = sum(1 for x, y in zip(A, B) if x == 1 and y == 0)
                cc = sum(1 for x, y in zip(A, B) if x == 0 and y == 1)
                p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if (bb + cc) else 1.0
                da, db = sum(A), sum(B)
                sig = "*" if p < 0.05 else " "
                print(f"    {a:<13} vs {b:<13}  {da:>2} vs {db:<2} hits  b={bb} c={cc}  p={p:.3f} {sig}")


if __name__ == "__main__":
    raise SystemExit(main())
