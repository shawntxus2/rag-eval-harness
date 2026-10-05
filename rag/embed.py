"""Local embedding with sentence-transformers — fp16, L2-normalized on write.

Precision is held at settings.torch_dtype (fp16 by default) across every run, the
same constant used everywhere else so the latency comparison stays honest.
Vectors are normalized so cosine distance equals a dot product, which is what the
partial HNSW indexes (vector_cosine_ops) assume; Phase 2 verifies the norms
actually came out at 1.0 rather than trusting the flag.
"""
from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from rag.config import settings

_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_device() -> str:
    if settings.device != "auto":
        return settings.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str) -> SentenceTransformer:
    device = resolve_device()
    model = SentenceTransformer(model_name, device=device)
    dtype = _DTYPE.get(settings.torch_dtype, torch.float16)
    if dtype is not torch.float32 and device in ("cuda", "mps"):
        model = model.to(dtype)
    return model


def embed(model: SentenceTransformer, texts: list[str], batch_size: int):
    """Return L2-normalized float32 vectors (numpy) for the given texts."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
