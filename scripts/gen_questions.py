#!/usr/bin/env python3
"""Phase 4 (generation) — author candidate eval questions in two types.

  cited       (default): a question pinned by an exact identifier or quoted term
              from the document (rule/section number, defined term, named entity).
              Tests retrieval when the user already knows where the answer lives.
  descriptive (arg 'descriptive'): the concept in the user's OWN words — no citation,
              no quoted phrase, no rare alphanumeric token, no multi-digit number.
              Uniqueness from semantics. Tests retrieval when the user does NOT know
              the rule number — the half where lexical should stop winning.

For each question the model returns an exact supporting quote; we locate it in the
document's cleaned text to set the GOLD CHARACTER SPAN. Scoring reads the span, not a
chunk id. Content-word overlap vs the gold span is stored for the paraphrase check.

Does NOT score. Deletes and regenerates only the type being run.

Usage:  python scripts/gen_questions.py [cited|descriptive]
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from rag.config import settings
from rag.db import connect

MODE = sys.argv[1] if len(sys.argv) > 1 else "cited"
assert MODE in ("cited", "descriptive"), "mode must be 'cited' or 'descriptive'"

RUN_LABEL = "chunk-500-overlap-50"
MIN_TOKENS = 400
MIN_PAGE = 4
MIN_GOLD_LEN, MAX_GOLD_LEN, MIN_GOLD_WORDS = 80, 400, 10
CAP = 3 if MODE == "cited" else 2
MAX_TRIES = 10 if MODE == "cited" else 16

META = re.compile(r"ISBN|©|rights reserved|\(online\)|\(print\)|www\.|https?:|bis\.org|iosco\.org", re.I)
TOC = re.compile(r"\.\s*\.\s*\.\s*\.|\.{4,}")
ANCHOR_BAD = re.compile(r"ISBN|Release\s*Nos?\.|File\s*No\.|©|effective date", re.I)
# Tokens a DESCRIPTIVE question may not contain (citations / rare alphanumerics / quotes).
CODE = re.compile(
    r"[A-Za-z]+\d|\d+[A-Za-z]|\d+\.\d|§|\bCFR\b|\bU\.?S\.?C\.?\b|"
    r"\b(?:Rule|Principle|Section|Part|Release|Paragraph|Article|Schedule|Regulation)\s+\S*\d|"
    r"\bNo\.\s*\d|\d{2,}|[\"“”]"
)

SYSTEM_CITED = (
    "You write evaluation questions for a retrieval system over US financial-market "
    "regulation documents. You are given one passage from ONE specific document.\n"
    "Write exactly ONE question obeying ALL of:\n"
    "1. Answerable ONLY from this passage, pinned by a detail UNIQUE to this document — a "
    "defined term as THIS document defines it, a rule/section/release number, a named entity, "
    "or a specific numeric threshold or date. Put that detail in the question.\n"
    "2. PARAPHRASE: different words and structure from the passage; reuse only the anchor.\n"
    "3. Answerable from a short, contiguous span of the passage.\n"
    "4. Anchor on SUBSTANTIVE regulatory content, never ISBNs, copyright/rights notices, "
    "publication/'revised'/effective dates, release/file numbers, or edition labels.\n"
    "5. Do NOT state or reveal the answer inside the question.\n"
    "Return ONLY JSON: {\"question\": str, \"uniqueness_anchor\": str, "
    "\"supporting_quote\": str} — supporting_quote copied VERBATIM from the passage."
)

SYSTEM_DESC = (
    "You write DESCRIPTIVE evaluation questions for a retrieval system over US financial-"
    "market regulation documents. You are given one passage from ONE specific document.\n"
    "Write exactly ONE question obeying ALL of:\n"
    "1. NO citations and NO quoting. Do NOT include any rule/section/principle number, "
    "CFR/USC cite, release/file number, ISBN, date, percentage, dollar/euro amount, any "
    "multi-digit number, or any token mixing letters and digits (e.g. 2hRTO, 17Ad-22). Do "
    "NOT copy any phrase from the passage. Describe everything in your OWN words.\n"
    "2. Ask the way someone would who does NOT know where the answer lives — describe the "
    "concept, situation, or requirement in plain language.\n"
    "3. Still answerable from ONLY this document: uniqueness must come from SEMANTICS — a "
    "named organisation, a threshold DESCRIBED (not quoted), or a distinctive combination of "
    "facts specific to how THIS document treats the topic. Avoid concepts so generic another "
    "publisher's document would answer equally.\n"
    "4. Natural phrasing a person would actually type. Do NOT state or hint at the answer.\n"
    "5. Answerable from a short, contiguous span of the passage.\n"
    "Return ONLY JSON: {\"question\": str, \"uniqueness_anchor\": str, "
    "\"supporting_quote\": str} — uniqueness_anchor names the semantic basis; "
    "supporting_quote copied VERBATIM from the passage."
)
SYSTEM = SYSTEM_CITED if MODE == "cited" else SYSTEM_DESC

_WORD = re.compile(r"[a-z]{3,}")


def content_words(s):
    return {w for w in _WORD.findall(s.lower()) if w not in ENGLISH_STOP_WORDS}


def overlap(question, gold):
    q = content_words(question)
    return len(q & content_words(gold)) / len(q) if q else 0.0


def locate(quote, chunk_text):
    toks = quote.split()
    if not toks:
        return None
    m = re.compile(r"\s+".join(re.escape(t) for t in toks)).search(chunk_text)
    return (m.start(), m.end()) if m else None


def descriptive_ok(question, gold):
    if CODE.search(question):
        return False
    qt = re.findall(r"\w+", question.lower())
    g = gold.lower()
    return not any(" ".join(qt[i:i + 4]) in g for i in range(len(qt) - 3))


def call_llm(client, model, title, passage):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Document title: {title}\n\nPassage:\n{passage}"}],
            response_format={"type": "json_object"}, max_completion_tokens=2000)
        return json.loads(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        print(f"    LLM/parse error: {type(e).__name__}: {str(e)[:100]}")
        return None


def main() -> int:
    from openai import OpenAI
    client = OpenAI()
    model = settings.generation_model
    root = pathlib.Path(__file__).resolve().parent.parent
    cache: dict[str, str] = {}
    with connect() as conn:
        run_id = conn.execute("SELECT id FROM ingestion_runs WHERE label=%s", (RUN_LABEL,)).fetchone()[0]
        docs = conn.execute("SELECT id, filename FROM documents ORDER BY filename").fetchall()
        conn.execute("DELETE FROM eval_questions WHERE generated_by='openai' AND status='candidate' AND question_type=%s", (MODE,))
        made = failed = 0
        for doc_id, filename in docs:
            cleaned = cache.setdefault(filename, (root/"corpus"/"cleaned"/(filename[:-4]+".txt")).read_text())
            chunks = conn.execute(
                "SELECT id,char_start,char_end,page_number,text FROM chunks WHERE ingestion_run_id=%s "
                "AND document_id=%s AND token_count>=%s AND page_number>=%s ORDER BY chunk_index",
                (run_id, doc_id, MIN_TOKENS, MIN_PAGE)).fetchall()
            pool = [c for c in chunks if not META.search(c[4])] or chunks
            if not pool:
                continue
            order = pool[:: max(1, len(pool)//MAX_TRIES)][:MAX_TRIES]
            accepted = 0
            for cid, cstart, cend, page, text in order:
                if accepted >= CAP:
                    break
                out = call_llm(client, model, filename, text)
                if not out or not all(k in out for k in ("question", "uniqueness_anchor", "supporting_quote")):
                    failed += 1
                    continue
                if ANCHOR_BAD.search(out["uniqueness_anchor"] or ""):
                    continue
                loc = locate(out["supporting_quote"], text)
                if not loc:
                    failed += 1
                    continue
                gstart, gend = cstart + loc[0], cstart + loc[1]
                g = cleaned[gstart:gend].strip()
                if (len(g) < MIN_GOLD_LEN or len(g) > MAX_GOLD_LEN or len(g.split()) < MIN_GOLD_WORDS
                        or META.search(g) or TOC.search(g)):
                    continue
                if MODE == "descriptive" and not descriptive_ok(out["question"], g):
                    continue
                conn.execute(
                    """INSERT INTO eval_questions
                       (question, source_document_id, gold_page, gold_char_start, gold_char_end,
                        answer_chunk_id, generated_from_run, generated_by, generator_model,
                        uniqueness_anchor, question_type, gold_overlap, status)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,'openai',%s,%s,%s,%s,'candidate')""",
                    (out["question"], doc_id, page, gstart, gend, cid, run_id, model,
                     out["uniqueness_anchor"], MODE, overlap(out["question"], g)))
                made += 1
                accepted += 1
            print(f"  {filename}: {accepted}", end="\r", flush=True)
        print(" " * 60, end="\r")
        print(f"[{MODE}] generated {made} candidates ({failed} failed) with {model}\n")
        _report(conn)
    return 0


def _report(conn) -> None:
    rows = conn.execute("SELECT gold_overlap FROM eval_questions WHERE generated_by='openai' "
                        "AND status='candidate' AND question_type=%s AND gold_overlap IS NOT NULL "
                        "ORDER BY gold_overlap", (MODE,)).fetchall()
    ov = [r[0] for r in rows]
    n = len(ov)
    pct = lambda p: ov[min(n - 1, int(p * n))]
    print(f"[{MODE}] CONTENT-WORD OVERLAP vs gold span:")
    print(f"  n={n}  min={ov[0]:.2f}  p25={pct(.25):.2f}  median={pct(.5):.2f}  p75={pct(.75):.2f}  p90={pct(.9):.2f}  max={ov[-1]:.2f}")
    print(f"  fraction >0.5: {sum(1 for x in ov if x > 0.5)/n:.0%}   median verdict: "
          + ("OK (<0.5)" if pct(.5) < 0.5 else "TOO HIGH"))
    root = pathlib.Path(__file__).resolve().parent.parent
    show = conn.execute(
        """SELECT d.filename,q.uniqueness_anchor,q.gold_overlap,q.question,q.gold_char_start,q.gold_char_end,q.source_document_id
           FROM eval_questions q JOIN documents d ON d.id=q.source_document_id
           WHERE q.generated_by='openai' AND q.status='candidate' AND q.question_type=%s ORDER BY q.id LIMIT 20""", (MODE,)).fetchall()
    cache: dict[int, str] = {}
    print(f"\n20 SAMPLE {MODE.upper()} QUESTIONS:")
    for fn, anc, ov_, q, gs, ge, did in show:
        txt = cache.setdefault(did, (root/"corpus"/"cleaned"/(fn[:-4]+".txt")).read_text())
        gold = re.sub(r"\s+", " ", txt[gs:ge]).strip()
        print(f"\n  [{fn}]  anchor={anc!r}  overlap={ov_:.2f}")
        print(f"    Q: {q}")
        print(f"    gold span: {gold[:150]!r}")


if __name__ == "__main__":
    raise SystemExit(main())
