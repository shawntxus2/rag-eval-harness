"""Furniture removal — strip running headers/footers, Federal Register print
lines, and the FR vertical-sidebar watermark, without touching body or footnotes.

Rules (approved):
  1. Repetition: a line appearing in the top-3 or bottom-3 non-empty lines of a
     page, whose digit-masked form recurs on >= 50% of pages, is furniture.
     Confined to edge positions so body text can never be caught.
  2. Federal Register typesetting footer: any line beginning 'VerDate'.
  3. FR sidebar watermark '<user> on DSK...PROD with RULES#' — removed as a
     substring wherever it occurs. It appears both glued to the VerDate footer
     AND as standalone lines on ~71 pages that the VerDate anchor alone misses.

Footnotes are KEPT deliberately: they carry definitions eval questions target,
so stripping them would cap recall below what any retrieval strategy could fix.

The text this returns is the CLEANED text. chunks.char_start/char_end are offsets
into it (see README), and it is persisted per document so those offsets resolve.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_EDGE = 3            # how many lines at each end count as "edge"
_RECURRENCE = 0.5    # fraction of pages a line must recur on to be furniture
_VERDATE = re.compile(r"^VerDate\b")
_WATERMARK = re.compile(r"\S+\s+on\s+DSK\S*PROD\s+with\s+[A-Z0-9]+", re.I)
_DIGITS = re.compile(r"\d+")


def _norm(line: str) -> str:
    """Furniture key: collapse a running header/footer to one form regardless of
    where its page number sits.

    Leading/trailing page-number tokens are dropped entirely (BIS prints the
    number as a prefix on even pages and a suffix on odd pages, so masking it to
    '#' in place would split one header into two sub-threshold variants);
    remaining interior digits are masked so dates/volume numbers still collapse.
    """
    s = _DIGITS.sub("#", line.strip())        # mask all numbers (dates, vol/no, page #)
    s = re.sub(r"^(#\s+)+", "", s)            # drop leading masked tokens (prefix page #)
    s = re.sub(r"(\s+#)+$", "", s)            # drop trailing masked tokens (suffix page #, year)
    return s.strip()


@dataclass
class CleanResult:
    text: str                                   # cleaned text (offsets index into this)
    page_spans: list[tuple[int, int, int]]      # (page_number, start, end) into text
    stripped: dict = field(default_factory=dict)  # norm -> {count, example, chars}
    raw_chars: int = 0                           # content chars before cleaning
    removed_chars: int = 0                       # chars removed as furniture

    @property
    def pct_removed(self) -> float:
        return 100.0 * self.removed_chars / self.raw_chars if self.raw_chars else 0.0


def _furniture_set(pages: list[str]) -> set[str]:
    n = len(pages)
    freq: Counter[str] = Counter()
    for pg in pages:
        ne = [l for l in pg.split("\n") if l.strip()]
        for l in ne[:_EDGE] + ne[-_EDGE:]:
            freq[_norm(l)] += 1
    return {k for k, c in freq.items() if c >= _RECURRENCE * n}


def clean_document(pages: list[str]) -> CleanResult:
    furniture = _furniture_set(pages)
    out_pages: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    stripped: dict = {}
    raw_chars = removed_chars = 0
    cursor = 0

    def record(norm_key: str, line: str) -> None:
        nonlocal removed_chars
        removed_chars += len(line)
        s = stripped.setdefault(norm_key, {"count": 0, "example": line.strip(), "chars": 0})
        s["count"] += 1
        s["chars"] += len(line)

    for pageno, pg in enumerate(pages, start=1):
        ne = [l for l in pg.split("\n") if l.strip()]
        edge_idx = set(range(min(_EDGE, len(ne)))) | set(range(max(0, len(ne) - _EDGE), len(ne)))
        kept: list[str] = []
        for i, line in enumerate(ne):
            raw_chars += len(line)
            # Rule 2: Federal Register print footer.
            if _VERDATE.match(line.strip()):
                record("VerDate…", line)
                continue
            # Rule 1: recurring header/footer at an edge position.
            if i in edge_idx and _norm(line) in furniture:
                record(_norm(line), line)
                continue
            # Rule 3: watermark substring (standalone -> line vanishes; glued -> body kept).
            if _WATERMARK.search(line):
                cleaned_line = _WATERMARK.sub("", line).strip()
                removed_amt = len(line) - len(cleaned_line)
                if removed_amt:
                    s = stripped.setdefault("…on DSK…PROD with RULES#",
                                            {"count": 0, "example": _WATERMARK.search(line).group(0), "chars": 0})
                    s["count"] += 1
                    s["chars"] += removed_amt
                    removed_chars += removed_amt
                if cleaned_line:
                    kept.append(cleaned_line)
                continue
            kept.append(line)
        page_text = "\n".join(kept)
        start = cursor
        out_pages.append(page_text)
        cursor += len(page_text)
        page_spans.append((pageno, start, cursor))
        cursor += 1  # for the "\n" joined between pages

    text = "\n".join(out_pages)
    return CleanResult(text=text, page_spans=page_spans, stripped=stripped,
                       raw_chars=raw_chars, removed_chars=removed_chars)
