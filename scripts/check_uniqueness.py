#!/usr/bin/env python3
"""Descriptive-question uniqueness check (don't invite eyeballing — measure it).

For each descriptive candidate: retrieve the top-20 chunks corpus-wide (vector),
take the chunks from documents OTHER than the gold document, and have the model
judge whether any of them, on its own, fully answers the question. A clear
conflict (another document fully answers) is dropped. Questions where another
document is merely topically related are kept and counted — that residual is a
named floor on measurable recall in a corpus with deliberate topic overlap.

Usage:  python scripts/check_uniqueness.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rag.config import settings
from rag.db import connect
from rag.retrieval import Retriever

RUN = "chunk-500-overlap-50"
TOPN = 20
MAX_OTHER = 6


def main() -> int:
    from openai import OpenAI
    client = OpenAI()
    r = Retriever()
    with connect() as conn:
        r.configure_session(conn)
        rid = conn.execute("SELECT id FROM ingestion_runs WHERE label=%s", (RUN,)).fetchone()[0]
        qs = conn.execute(
            """SELECT q.id, q.question, q.source_document_id, d.filename
               FROM eval_questions q JOIN documents d ON d.id=q.source_document_id
               WHERE q.generated_by='openai' AND q.status='candidate' AND q.question_type='descriptive'
               ORDER BY q.id""").fetchall()
        clear, residual = [], []
        for qid, question, gold_doc, gold_fn in qs:
            hits = r.vector(conn, question, rid, depth=TOPN)
            others = []
            for h in hits:
                row = conn.execute(
                    "SELECT c.document_id, d.filename, left(c.text,400) FROM chunks c "
                    "JOIN documents d ON d.id=c.document_id WHERE c.id=%s", (h.chunk_id,)).fetchone()
                if row[0] != gold_doc:
                    others.append((row[1], row[2]))
                if len(others) >= MAX_OTHER:
                    break
            if not others:
                continue
            listing = "\n".join(f"{i+1}. [{fn}] {txt}" for i, (fn, txt) in enumerate(others))
            prompt = (f"Question: {question}\n\nThe passages below are from DIFFERENT documents than the "
                      f"one the question was written for. For EACH, decide whether it, on its own, FULLY "
                      f"answers the question.\n\n{listing}\n\n"
                      'Return JSON {"judgments":[{"n":<num>,"verdict":"answers"|"related"|"unrelated"}]}.')
            try:
                resp = client.chat.completions.create(model=settings.generation_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, max_completion_tokens=1500)
                js = json.loads(resp.choices[0].message.content).get("judgments", [])
            except Exception as e:  # noqa: BLE001
                print(f"  judge error q{qid}: {e}")
                continue
            ans = [j["n"] for j in js if j.get("verdict") == "answers" and 1 <= j.get("n", 0) <= len(others)]
            rel = any(j.get("verdict") == "related" for j in js)
            if ans:
                clear.append((qid, question, gold_fn, others[ans[0]-1][0], others[ans[0]-1][1]))
            elif rel:
                residual.append((qid, gold_fn))

        n = len(qs)
        print(f"DESCRIPTIVE UNIQUENESS CHECK ({n} questions, vector top-{TOPN}, other-doc judged)")
        print(f"  clear conflicts (another document fully answers) — DROP: {len(clear)}/{n} = {len(clear)/n:.0%}")
        print(f"  residual topical overlap (related, not answering) — KEEP: {len(residual)}/{n} = {len(residual)/n:.0%}")
        print("\n  CLEAR CONFLICTS (dropped):")
        for qid, q, gold_fn, conf_fn, snip in clear:
            print(f"\n   q{qid}  gold={gold_fn}  conflict={conf_fn}")
            print(f"     Q: {q}")
            print(f"     other-doc passage: {snip[:150]!r}")
        if clear:
            conn.execute("DELETE FROM eval_questions WHERE id = ANY(%s)", ([c[0] for c in clear],))
        surv = conn.execute("SELECT count(*) FROM eval_questions WHERE generated_by='openai' "
                            "AND status='candidate' AND question_type='descriptive'").fetchone()[0]
        print(f"\n  descriptive survivors after dropping clear conflicts: {surv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
