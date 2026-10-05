"""Token-based chunking over cleaned text, with page attribution.

Chunk size is measured in tokens of the embedding model's own tokenizer
(bge-base), so "500 tokens" means what the embedder will actually see.
char_start/char_end are offsets into the CLEANED text (see rag/furniture and the
README), recovered from the fast tokenizer's offset mapping. page_number/page_end
come from the per-page char spans produced during cleaning.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    page_number: int
    page_end: int
    text: str


def _page_of(pos: int, page_spans: list[tuple[int, int, int]]) -> int:
    for pageno, start, end in page_spans:
        if start <= pos < end:
            return pageno
    return page_spans[-1][0] if page_spans else 1


def chunk_text(
    text: str,
    page_spans: list[tuple[int, int, int]],
    tokenizer,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Sliding-window chunks of `chunk_size` tokens, stepping by size-overlap."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    idx = ci = 0
    while idx < n:
        window = offsets[idx: idx + chunk_size]
        if not window:
            break
        cs = window[0][0]
        ce = window[-1][1]
        chunks.append(
            Chunk(
                chunk_index=ci,
                char_start=cs,
                char_end=ce,
                token_count=len(window),
                page_number=_page_of(cs, page_spans),
                page_end=_page_of(max(cs, ce - 1), page_spans),
                text=text[cs:ce],
            )
        )
        ci += 1
        if idx + chunk_size >= n:  # last window reached the end
            break
        idx += step
    return chunks
