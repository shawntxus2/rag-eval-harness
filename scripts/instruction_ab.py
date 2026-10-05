#!/usr/bin/env python3
"""A/B the bge query instruction (vector-only), before the eval.

Identical (query, gold) pairs scored with the instruction ON vs OFF; the delta on
the SAME pairs is the signal. Gold chunks are located by a distinctive substring in
the 500-token run. Reports the rank of the gold target for each query under both
settings, and a summary, so the instruction default can be set on evidence.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from statistics import median

from rag.config import settings
from rag.db import connect
from rag.retrieval import Retriever

RUN = "chunk-500-overlap-50"
DEPTH = 100

# (label, query, distinctive substring identifying the gold chunk)
PAIRS = [
    # --- drift-anchored (query paraphrases the cited passage's actual content) ---
    ("haircut", "How should a CCP's board assess and limit procyclical increases in the margins and collateral discounts it collects?", "collateral haircuts. The approach established by the board"),
    ("waterfall", "What prefunded resources form the layers a clearing house draws on to cover losses from a defaulting member?", 'commonly referred to as a “waterfall”'),
    ("margin", "What are the assets a clearing house requires members to post against their credit exposures called?", "they may call these requirements margin"),
    ("exposure", "What should a clearing house stress test to confirm it holds enough resources against a member default?", "stress testing for both credit and liquidity"),
    ("netting", "How does central clearing reduce the total size of members' mutual obligations?", "aggregate size of these exposures through multilateral netting"),
    ("clearing", "Why should supervisory authorities coordinate given the interdependencies among post-trade processing systems?", "it is important for authorities to cooperate"),
    ("position", "What must institutional investment managers report about their short selling, and on what form?", "Short Position and Short Activity Reporting by Institutional"),
    ("settlement", "Does the affirmation rule force a trade to finalize on the day it is executed?", "settlement of the transaction on trade date"),
    ("collateral", "What may a clearing house in cash markets require members to pledge to cover credit exposures?", "require each participant to provide collateral"),
    # --- distinctive-fact paraphrases ---
    ("two_hours", "Within what time should a market infrastructure aim to resume critical operations after a major disruption?", "within two hours"),
    ("99pct", "To what confidence level does NSCC size each member's required fund deposit?", "99 percent confidence"),
    ("cover_two", "How many members' simultaneous default must a systemically important clearing organization withstand?", "largest combined"),
    ("clock_sync", "What must broker-dealers synchronize so audit-trail timestamps line up?", "Synchronization of Member Business Clocks"),
    ("fdta", "Which 2022 statute directs the Commission to adopt machine-readable data standards for its collections?", "Financial Data Transparency Act"),
    ("fail_charge", "What additional charge does NSCC apply when a member's trades do not settle on schedule?", "fail charge"),
    ("same_day_affirm", "What same-day step lets institutional trades settle on the shortened cycle?", "same-day affirmation"),
    ("reg_sho", "What close-out timing under the short-sale rule changes when the settlement cycle shortens?", "bona fide market"),
    ("ppr", "Which Federal Reserve policy governs the risks institutions pose in payment and settlement systems?", "Policy on Payment System Risk"),
    ("guaranty", "What pooled default resource backs a clearing organization if a member fails?", "guaranty fund"),
]


def main() -> int:
    r = Retriever()  # bge-base
    instr = settings.query_instruction
    with connect() as conn:
        r.configure_session(conn)
        rid = conn.execute("SELECT id FROM ingestion_runs WHERE label=%s", (RUN,)).fetchone()[0]

        # resolve gold chunk ids
        golds = {}
        for label, _q, sub in PAIRS:
            row = conn.execute(
                "SELECT c.id, d.filename, c.page_number FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE c.ingestion_run_id=%s AND c.text ILIKE %s ORDER BY c.id LIMIT 1",
                (rid, "%" + sub.replace("\n", " ") + "%"),
            ).fetchone()
            golds[label] = row

        def ranks(use_instruction: bool):
            r.query_instruction = instr if use_instruction else ""
            out = {}
            for label, q, _sub in PAIRS:
                g = golds[label]
                if not g:
                    out[label] = None
                    continue
                hits = r.vector(conn, q, rid, depth=DEPTH)
                out[label] = next((i + 1 for i, h in enumerate(hits) if h.chunk_id == g[0]), None)
            return out

        on = ranks(True)
        off = ranks(False)

        print(f"bge query-instruction A/B (vector-only, run={RUN}, depth={DEPTH})")
        print(f"instruction = {instr!r}\n")
        print(f"{'query':16}{'gold doc/pg':38}{'ON':>5}{'OFF':>5}  better")
        def fmt(x): return ">100" if x is None else str(x)
        won_on = won_off = tie = 0
        for label, _q, _sub in PAIRS:
            g = golds[label]
            gd = "NOT FOUND" if not g else f"{g[1][:30]} p{g[2]}"
            a, b = on[label], off[label]
            av, bv = (a or 999), (b or 999)
            tag = "ON" if av < bv else ("OFF" if bv < av else "=")
            won_on += av < bv; won_off += bv < av; tie += av == bv
            print(f"{label:16}{gd:38}{fmt(a):>5}{fmt(b):>5}  {tag}")

        def summ(d):
            vals = [v for v in d.values() if v is not None]
            top1 = sum(1 for v in d.values() if v == 1)
            top5 = sum(1 for v in d.values() if v and v <= 5)
            inwin = len(vals)
            med = median([v if v else 999 for v in d.values()])
            return top1, top5, inwin, med
        for name, d in (("ON ", on), ("OFF", off)):
            t1, t5, iw, med = summ(d)
            print(f"\n  {name}: top1={t1}/{len(PAIRS)}  top5={t5}/{len(PAIRS)}  in-top100={iw}/{len(PAIRS)}  median_rank={med}")
        print(f"\n  per-query: ON better on {won_on}, OFF better on {won_off}, tie {tie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
