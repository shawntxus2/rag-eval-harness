#!/usr/bin/env python3
"""Phase 1 — ingest the corpus: extract → strip furniture → persist cleaned text
→ chunk at 250/500/1000 tokens → load documents / ingestion_runs / chunks.

Extraction is held constant (pypdf default, rag/extract). Furniture rules live in
rag/furniture. char_start/char_end on chunks are offsets into the CLEANED text,
which is persisted to corpus/cleaned/<stem>.txt so those offsets always resolve.

Prints the furniture-removal report for ALL documents (distinct lines stripped,
hit counts, % of content removed) and FLAGS anything above 15% without loading it.
Idempotent: dedups documents by sha256, upserts the three runs, and reloads chunks
for each (run, document) pair it processes.

Usage:  python scripts/ingest.py
"""
from __future__ import annotations

import glob
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from rag.chunk import chunk_text
from rag.config import settings
from rag.db import connect
from rag.extract import extract_pages
from rag.furniture import clean_document

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
CLEANED = CORPUS / "cleaned"
FLAG_PCT = 15.0

# The three chunk-size runs (10% token overlap each).
RUNS = [
    ("chunk-250-overlap-25", 250, 25),
    ("chunk-500-overlap-50", 500, 50),
    ("chunk-1000-overlap-100", 1000, 100),
]


def _upsert_document(conn, name, rel_path, sha, pages, size) -> int:
    row = conn.execute(
        """INSERT INTO documents (filename, rel_path, sha256, num_pages, size_bytes, parse_status)
           VALUES (%s,%s,%s,%s,%s,'ok')
           ON CONFLICT (sha256) DO UPDATE SET filename=EXCLUDED.filename,
             rel_path=EXCLUDED.rel_path, num_pages=EXCLUDED.num_pages,
             size_bytes=EXCLUDED.size_bytes, parse_status='ok'
           RETURNING id""",
        (name, rel_path, sha, pages, size),
    ).fetchone()
    return row[0]


def _upsert_run(conn, label, size, overlap, tokenizer) -> int:
    row = conn.execute(
        """INSERT INTO ingestion_runs (label, chunk_size, overlap, tokenizer, status)
           VALUES (%s,%s,%s,%s,'embedding')
           ON CONFLICT (label) DO UPDATE SET chunk_size=EXCLUDED.chunk_size,
             overlap=EXCLUDED.overlap, tokenizer=EXCLUDED.tokenizer
           RETURNING id""",
        (label, size, overlap, tokenizer),
    ).fetchone()
    return row[0]


def main() -> int:
    CLEANED.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(settings.embedding_model)
    tok.model_max_length = 10**9  # silence the >512 length warning; we chunk ourselves

    pdfs = sorted(glob.glob(str(CORPUS / "*.pdf")))
    print(f"Ingesting {len(pdfs)} documents | tokenizer={settings.embedding_model}\n")

    # ---- Pass 1: extract + clean + persist, collect furniture report ----
    cleaned: dict[str, dict] = {}
    report = []
    for f in pdfs:
        name = pathlib.Path(f).name
        pages = extract_pages(f)
        res = clean_document(pages)
        (CLEANED / (pathlib.Path(f).stem + ".txt")).write_text(res.text)
        data = pathlib.Path(f).read_bytes()
        cleaned[name] = {
            "res": res,
            "pages": len(pages),
            "sha": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "rel": f"corpus/{name}",
        }
        report.append((name, res))

    # ---- Furniture report (all docs) ----
    print("FURNITURE REMOVAL REPORT (all documents)")
    print(f"{'pct':>6} {'removed':>8}  document")
    flagged = []
    for name, res in sorted(report, key=lambda r: -r[1].pct_removed):
        mark = "  <<< FLAG >15%" if res.pct_removed > FLAG_PCT else ""
        if res.pct_removed > FLAG_PCT:
            flagged.append(name)
        print(f"{res.pct_removed:6.2f}% {res.removed_chars:9,}  {name}{mark}")
    if flagged:
        print(f"\nABORT: {len(flagged)} document(s) over {FLAG_PCT}% — not loading. "
              "Review before accepting:", ", ".join(flagged))
        return 2

    # ---- Pass 2: chunk + load ----
    with connect() as conn:
        run_ids = {}
        for label, size, overlap in RUNS:
            run_ids[label] = _upsert_run(conn, label, size, overlap, settings.embedding_model)

        for name, info in cleaned.items():
            doc_id = _upsert_document(conn, name, info["rel"], info["sha"], info["pages"], info["size"])
            res = info["res"]
            for label, size, overlap in RUNS:
                run_id = run_ids[label]
                conn.execute(
                    "DELETE FROM chunks WHERE ingestion_run_id=%s AND document_id=%s",
                    (run_id, doc_id),
                )
                chunks = chunk_text(res.text, res.page_spans, tok, size, overlap)
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO chunks (document_id, ingestion_run_id, chunk_index,
                               page_number, page_end, char_start, char_end, token_count, text)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        [(doc_id, run_id, c.chunk_index, c.page_number, c.page_end,
                          c.char_start, c.char_end, c.token_count, c.text) for c in chunks],
                    )

        # finalize run rows
        for label in run_ids:
            conn.execute(
                """UPDATE ingestion_runs SET num_chunks=(SELECT count(*) FROM chunks
                       WHERE ingestion_run_id=%s), status='ready' WHERE id=%s""",
                (run_ids[label], run_ids[label]),
            )

        _counts(conn)
    return 0


def _counts(conn) -> None:
    print("\nLOADED:")
    docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"  documents: {docs}  |  chunks total: {chunks}")
    for label, size, ov, nc in conn.execute(
        "SELECT label, chunk_size, overlap, num_chunks FROM ingestion_runs ORDER BY chunk_size"
    ).fetchall():
        print(f"  run {label:24} size={size:<5} overlap={ov:<4} chunks={nc}")


if __name__ == "__main__":
    raise SystemExit(main())
