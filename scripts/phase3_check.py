#!/usr/bin/env python3
"""Phase 3 validation: symmetry asserts, self-retrieval, and a worked example.

Does NOT report p50/p95 — latency over the full question set is Phase 4. It does
demonstrate the timing discipline (warmup discarded; query-embed timed separately
from the retrieval step) on a single labelled query.

Usage:  python scripts/phase3_check.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.config import settings
from rag.db import connect
from rag.retrieval import Retriever

RUN_LABEL = "chunk-500-overlap-50"
N_SELF = 20


def _doc_page(conn, chunk_id):
    return conn.execute(
        "SELECT d.filename, c.page_number, left(c.text, 90) FROM chunks c "
        "JOIN documents d ON d.id=c.document_id WHERE c.id=%s",
        (chunk_id,),
    ).fetchone()


def main() -> int:
    r = Retriever()  # bge-base, reranker loaded once
    with connect() as conn:
        r.configure_session(conn)
        r.assert_symmetry(conn)
        print(f"symmetry OK: query model={r.embedding_model} dtype={settings.torch_dtype} "
              f"device={r.device} | lexical ts_config='{r.ts_config}' matches tsv | query vecs asserted ‖q‖≈1")

        run_id = conn.execute("SELECT id FROM ingestion_runs WHERE label=%s", (RUN_LABEL,)).fetchone()[0]

        # ---- 5. self-retrieval check ----
        sample = conn.execute(
            "SELECT id, text FROM chunks WHERE ingestion_run_id=%s ORDER BY random() LIMIT %s",
            (run_id, N_SELF),
        ).fetchall()
        rank1 = 0
        misses = []
        saved = r.query_instruction
        r.query_instruction = ""  # symmetric: a chunk's own text must retrieve itself (pure plumbing)
        for cid, text in sample:
            hits = r.vector(conn, text, run_id, depth=10)
            rank = next((i + 1 for i, h in enumerate(hits) if h.chunk_id == cid), None)
            if rank == 1:
                rank1 += 1
            else:
                misses.append((cid, rank))
        r.query_instruction = saved
        print(f"\nSELF-RETRIEVAL (vector-only, no instruction, {N_SELF} random chunks as their own query): "
              f"{rank1}/{N_SELF} returned themselves at rank 1")
        if misses:
            print("  misses (chunk_id, rank):", misses)

        # ---- worked example: where vector and lexical disagree pre-fusion ----
        query = "What valuation discount is applied to collateral based on its risk and liquidity?"
        # warmup (discarded), then time embed separately from the DB retrieval step
        r.vector(conn, query, run_id, depth=settings.retrieval_depth)
        t0 = time.perf_counter(); qv = r.embed_query(query); embed_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); vec = r.vector(conn, query, run_id); vec_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); lex = r.lexical(conn, query, run_id); lex_ms = (time.perf_counter() - t0) * 1000

        print(f"\nWORKED EXAMPLE  query={query!r}  (run={RUN_LABEL})")
        print(f"  [timing, single query, illustrative] embed={embed_ms:.1f}ms  "
              f"vector-search={vec_ms:.1f}ms  lexical-search={lex_ms:.1f}ms")
        for label, hits in (("VECTOR top-5", vec[:5]), ("LEXICAL top-5", lex[:5])):
            print(f"\n  {label}:")
            for h in hits:
                fn, pg, snip = _doc_page(conn, h.chunk_id)
                print(f"    id={h.chunk_id:<7} score={h.score:7.4f}  {fn} p{pg}")
                print(f"              {snip!r}")
        vset = {h.chunk_id for h in vec[:5]}
        lset = {h.chunk_id for h in lex[:5]}
        print(f"\n  overlap of top-5 (vector ∩ lexical): {len(vset & lset)} of 5  "
              f"— disagreement is what RRF fusion has to reconcile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
