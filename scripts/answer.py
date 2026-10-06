#!/usr/bin/env python3
"""Phase 5 — answer a question with required citations, over retrieved context.

Retrieves the top-k chunks (vector+rerank by default — the precision mode from the
Phase 4 results), passes their text to the generation provider (OpenAI, via the
Phase 0 seam in rag/generation.py), and prints the answer with its [n] citations
resolved to document + page. Generation is the ONLY step that calls an API;
retrieval and evaluation never do.

Usage:  python scripts/answer.py "your question" [mode] [chunk_size] [k]
        mode defaults to vector_rerank, chunk_size 500, k 5.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.db import connect
from rag.generation import generate
from rag.retrieval import Retriever


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: answer.py "question" [mode] [chunk_size] [k]', file=sys.stderr)
        return 2
    question = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "vector_rerank"
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    r = Retriever()
    with connect() as conn:
        r.configure_session(conn)
        run = conn.execute("SELECT id FROM ingestion_runs WHERE chunk_size=%s LIMIT 1", (size,)).fetchone()
        if not run:
            print(f"no ingestion run for chunk_size={size}", file=sys.stderr)
            return 1
        hits = getattr(r, mode)(conn, question, run[0])[:k]
        if not hits:
            print("no chunks retrieved — nothing to answer from", file=sys.stderr)
            return 1
        rows = {cid: (fn, pg, txt) for cid, fn, pg, txt in conn.execute(
            "SELECT c.id, d.filename, c.page_number, c.text FROM chunks c "
            "JOIN documents d ON d.id=c.document_id WHERE c.id = ANY(%s)",
            ([h.chunk_id for h in hits],)).fetchall()}
        context = [rows[h.chunk_id][2] for h in hits]

        gen = generate(question, context)  # raises loudly on missing/bad key

        print(f"Q: {question}\n")
        print(gen.text + "\n")
        print("Sources (passage [n] → document, page):")
        for i, h in enumerate(hits, 1):
            fn, pg, _ = rows[h.chunk_id]
            print(f"  [{i}] {fn}  p{pg}  (chunk {h.chunk_id})")
        print(f"\nmodel={gen.model}  tokens in={gen.input_tokens} out={gen.output_tokens}  "
              f"| retrieval={mode} @ {size}-token chunks, top {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
