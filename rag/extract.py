"""Raw PDF text extraction.

Deliberately ONE extractor with ONE setting for the entire corpus: pypdf's
default ``extract_text()``. Holding extraction constant is a measurement
decision — when a retrieval number later looks wrong, the extractor is not a
suspect because it never varied. (pypdf reads single- and three-column pages in
correct reading order; it does hoist Federal Register footnotes above the body,
which is accepted: the text is real and correctly attributed, the only cost is
chunk coherence.)
"""
from __future__ import annotations

from pypdf import PdfReader


def extract_pages(path: str) -> list[str]:
    """Return raw text per page, exactly as pypdf returns it — no cleaning."""
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]
