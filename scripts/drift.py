#!/usr/bin/env python3
"""Domain-drift probe — where do the nine ambiguous terms live in embedding space?

Each term has an everyday meaning unrelated to its meaning here. We embed the BARE
term (no query instruction) with bge-base and show its nearest-neighbour chunks in
the corpus (500-token run). The question for each: does the model place the bare
word next to its DOMAIN sense or its GENERAL-LANGUAGE sense? (Classification is read
off these neighbours in the README.)

Usage:  python scripts/drift.py
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.db import connect
from rag.retrieval import Retriever

TERMS = ["haircut", "waterfall", "margin", "exposure", "netting",
         "position", "settlement", "clearing", "collateral"]


def main() -> int:
    r = Retriever()
    r.query_instruction = ""  # probe the bare term, not an s2p query
    with connect() as conn:
        r.configure_session(conn)
        rid = conn.execute("SELECT id FROM ingestion_runs WHERE chunk_size=500 LIMIT 1").fetchone()[0]
        for term in TERMS:
            hits = r.vector(conn, term, rid, depth=3)
            print(f"\n### {term}")
            for h in hits:
                fn, pg, txt = conn.execute(
                    "SELECT d.filename, c.page_number, c.text FROM chunks c JOIN documents d ON d.id=c.document_id "
                    "WHERE c.id=%s", (h.chunk_id,)).fetchone()
                snip = re.sub(r"\s+", " ", txt).strip()
                # center the snippet on the term if present
                m = re.search(re.escape(term), snip, re.I)
                s = max(0, m.start() - 60) if m else 0
                print(f"  {h.score:.3f} {fn[:34]:34} p{pg}  …{snip[s:s+150]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
