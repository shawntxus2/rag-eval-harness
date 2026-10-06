#!/usr/bin/env python3
"""Phase 4 (generation) — author candidate eval questions, paraphrased not quoted.

For each document, up to CAP source chunks (500-token run) are handed to the LLM
(OpenAI, settings.generation_model) with instructions to write ONE question that:
  - is anchored by a detail unique to this document (defined term, rule/section
    number, named entity, threshold/date), and
  - is phrased in DIFFERENT words from the source (paraphrase), reusing only the anchor.
The model also returns the exact supporting quote; we locate it (whitespace-tolerant)
in the document's cleaned text to set the GOLD CHARACTER SPAN. Scoring later reads the
span, not a chunk id, so one question is valid across all chunk sizes.

Then we VERIFY rather than trust: content-word overlap between the question and its
gold span is computed and stored. If the median is high the set is measuring string
matching and must be regenerated with a stronger paraphrase instruction.

Does NOT score anything. Prints the overlap distribution and sample questions.

Usage:  python scripts/gen_questions.py
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

CAP = 3                       # questions per document (stratification)
RUN_LABEL = "chunk-500-overlap-50"
MIN_TOKENS = 400              # author from near-full chunks (more content to anchor on)
MIN_PAGE = 4                  # skip title/TOC/front matter
MAX_TRIES = 10                # candidate chunks to try per doc to reach CAP
MIN_GOLD_LEN = 80             # gold span must carry real content, not a stub
MIN_GOLD_WORDS = 10           # ... and be a sentence, not a heading
MAX_GOLD_LEN = 400            # ... but still a short span (keeps containment meaningful)

# Metadata / front-matter the questions must NOT be about (bibliographic trivia).
META = re.compile(r"ISBN|©|rights reserved|\(online\)|\(print\)|www\.|https?:|bis\.org|iosco\.org", re.I)
# Table-of-contents / heading lines (dotted leaders) — not real answer spans.
TOC = re.compile(r"\.\s*\.\s*\.\s*\.|\.{4,}")
# Anchors that lead to shallow citation/date lookups rather than substantive content.
ANCHOR_BAD = re.compile(r"ISBN|Release\s*Nos?\.|File\s*No\.|©|effective date", re.I)

SYSTEM = (
    "You write evaluation questions for a retrieval system over US financial-market "
    "regulation documents. You are given one passage from ONE specific document.\n"
    "Write exactly ONE question obeying ALL of:\n"
    "1. It must be answerable ONLY from this passage, pinned by a detail UNIQUE to this "
    "document — a defined term as THIS document defines it, a rule/section/release number, "
    "a named entity, or a specific numeric threshold or date. Put that detail in the question.\n"
    "2. PARAPHRASE. Write the question in DIFFERENT words from the passage: change the "
    "sentence structure and use synonyms. Do NOT reuse the passage's distinctive noun "
    "phrases or verbs — EXCEPT the unique anchor itself, which may appear verbatim.\n"
    "3. It must be answerable from a short, contiguous span of the passage.\n"
    "4. Anchor on SUBSTANTIVE regulatory content: a defined term, a numbered "
    "rule/section/principle/key-consideration that states a requirement, a named entity in a "
    "substantive role, or a quantitative threshold (a percentage, dollar amount, or time "
    "limit). Do NOT use ISBNs, copyright or rights notices, publication or 'revised' dates, "
    "release/file numbers, edition or print/online labels, effective dates, or any "
    "front-matter or bibliographic metadata. The answer must be a regulatory fact, not a "
    "citation or publication detail.\n"
    "Return ONLY JSON: {\"question\": str, \"uniqueness_anchor\": str, "
    "\"supporting_quote\": str} where supporting_quote is copied VERBATIM, "
    "character-for-character, from the passage (the span that answers the question)."
)

_WORD = re.compile(r"[a-z]{3,}")


def content_words(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in ENGLISH_STOP_WORDS}


def overlap(question: str, gold: str) -> float:
    q = content_words(question)
    if not q:
        return 0.0
    return len(q & content_words(gold)) / len(q)


def locate(quote: str, chunk_text: str) -> tuple[int, int] | None:
    """Whitespace-tolerant location of quote within chunk_text -> (start, end) offsets."""
    toks = quote.split()
    if not toks:
        return None
    pat = re.compile(r"\s+".join(re.escape(t) for t in toks))
    m = pat.search(chunk_text)
    return (m.start(), m.end()) if m else None


def call_llm(client, model: str, title: str, passage: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Document title: {title}\n\nPassage:\n{passage}"},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2000,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        print(f"    LLM/parse error: {type(e).__name__}: {str(e)[:120]}")
        return None


def main() -> int:
    from openai import OpenAI

    client = OpenAI()
    model = settings.generation_model
    root = pathlib.Path(__file__).resolve().parent.parent
    cleaned_cache: dict[str, str] = {}

    with connect() as conn:
        run_id = conn.execute("SELECT id FROM ingestion_runs WHERE label=%s", (RUN_LABEL,)).fetchone()[0]
        docs = conn.execute("SELECT id, filename FROM documents ORDER BY filename").fetchall()
        conn.execute("DELETE FROM eval_questions WHERE generated_by='openai' AND status='candidate'")

        made = failed = 0
        for doc_id, filename in docs:
            cleaned = cleaned_cache.setdefault(
                filename, (root / "corpus" / "cleaned" / (filename[:-4] + ".txt")).read_text()
            )
            chunks = conn.execute(
                "SELECT id, char_start, char_end, page_number, text FROM chunks "
                "WHERE ingestion_run_id=%s AND document_id=%s AND token_count>=%s AND page_number>=%s "
                "ORDER BY chunk_index",
                (run_id, doc_id, MIN_TOKENS, MIN_PAGE),
            ).fetchall()
            pool = [c for c in chunks if not META.search(c[4])] or chunks
            if not pool:
                continue
            # evenly spaced candidate order, up to MAX_TRIES, to reach CAP accepted
            step = max(1, len(pool) // MAX_TRIES)
            order = pool[::step][:MAX_TRIES]
            accepted = 0
            for cid, cstart, cend, page, text in order:
                if accepted >= CAP:
                    break
                out = call_llm(client, model, filename, text)
                if not out or not all(k in out for k in ("question", "uniqueness_anchor", "supporting_quote")):
                    failed += 1
                    continue
                if ANCHOR_BAD.search(out["uniqueness_anchor"] or ""):
                    continue  # metadata anchor — skip, try another chunk
                loc = locate(out["supporting_quote"], text)
                if not loc:
                    failed += 1
                    continue
                gstart, gend = cstart + loc[0], cstart + loc[1]
                gold_text = cleaned[gstart:gend]
                g = gold_text.strip()
                if (len(g) < MIN_GOLD_LEN or len(g) > MAX_GOLD_LEN or len(g.split()) < MIN_GOLD_WORDS
                        or META.search(g) or TOC.search(g)):
                    continue  # trivial / metadata / TOC-heading gold — skip
                ov = overlap(out["question"], gold_text)
                conn.execute(
                    """INSERT INTO eval_questions
                       (question, source_document_id, gold_page, gold_char_start, gold_char_end,
                        answer_chunk_id, generated_from_run, generated_by, generator_model,
                        uniqueness_anchor, gold_overlap, status)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,'openai',%s,%s,%s,'candidate')""",
                    (out["question"], doc_id, page, gstart, gend, cid, run_id, model,
                     out["uniqueness_anchor"], ov),
                )
                made += 1
                accepted += 1
            print(f"  {filename}: {accepted} authored", end="\r", flush=True)
        print(" " * 60, end="\r")
        print(f"generated {made} candidates ({failed} failed to author/locate) with {model}\n")
        _report(conn)
    return 0


def _report(conn) -> None:
    rows = conn.execute(
        "SELECT gold_overlap FROM eval_questions WHERE generated_by='openai' AND status='candidate' "
        "AND gold_overlap IS NOT NULL ORDER BY gold_overlap"
    ).fetchall()
    ov = [r[0] for r in rows]
    n = len(ov)
    def pct(p): return ov[min(n - 1, int(p * n))]
    print("CONTENT-WORD OVERLAP (question vs gold span) distribution:")
    print(f"  n={n}  min={ov[0]:.2f}  p25={pct(.25):.2f}  median={pct(.5):.2f}  "
          f"p75={pct(.75):.2f}  p90={pct(.9):.2f}  max={ov[-1]:.2f}")
    bands = [(0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, 1.01)]
    for lo, hi in bands:
        c = sum(1 for x in ov if lo <= x < hi)
        print(f"    [{lo:.2f},{hi:.2f}): {c:3d}  {'#' * (c * 40 // max(1, n))}")
    hi_frac = sum(1 for x in ov if x > 0.5) / n
    med = pct(.5)
    print(f"  fraction with overlap > 0.5: {hi_frac:.0%}")
    print(f"  VERDICT: median={med:.2f} -> " +
          ("OK (< 0.5, paraphrased)" if med < 0.5 else "TOO HIGH (>= 0.5) — regenerate with stronger paraphrase instruction"))

    print("\n20 SAMPLE QUESTIONS (spread across publishers):")
    samples = conn.execute(
        """SELECT DISTINCT ON (left(d.filename,4)) d.filename, q.uniqueness_anchor, q.gold_overlap,
                  q.question, q.gold_char_start, q.gold_char_end, q.source_document_id
           FROM eval_questions q JOIN documents d ON d.id=q.source_document_id
           WHERE q.generated_by='openai' AND q.status='candidate'
           ORDER BY left(d.filename,4), q.id"""
    ).fetchall()
    # top up to 20 across all
    extra = conn.execute(
        """SELECT d.filename, q.uniqueness_anchor, q.gold_overlap, q.question,
                  q.gold_char_start, q.gold_char_end, q.source_document_id
           FROM eval_questions q JOIN documents d ON d.id=q.source_document_id
           WHERE q.generated_by='openai' AND q.status='candidate' ORDER BY q.id"""
    ).fetchall()
    seen, show = set(), []
    for r in samples + extra:
        if r[3] in seen:
            continue
        seen.add(r[3]); show.append(r)
        if len(show) == 20:
            break
    root = pathlib.Path(__file__).resolve().parent.parent
    cache: dict[int, str] = {}
    for fn, anchor, ov_, q, gs, ge, did in show:
        txt = cache.setdefault(did, (root / "corpus" / "cleaned" / (fn[:-4] + ".txt")).read_text())
        gold = re.sub(r"\s+", " ", txt[gs:ge]).strip()
        print(f"\n  [{fn}]  anchor={anchor!r}  overlap={ov_:.2f}")
        print(f"    Q: {q}")
        print(f"    gold span: {gold[:160]!r}")


if __name__ == "__main__":
    raise SystemExit(main())
