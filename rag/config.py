"""Central configuration. Every knob is an environment variable with a default,
so parameters can change without editing code (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# --- Supported embedding models → (chunks column, vector dimension) ---
# Adding a model means adding a column + partial HNSW index in db/schema.sql.
EMBEDDING_MODELS: dict[str, dict] = {
    "BAAI/bge-base-en-v1.5": {"column": "embedding_bge", "dim": 768},
    "sentence-transformers/all-MiniLM-L6-v2": {"column": "embedding_minilm", "dim": 384},
}


@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = _int("DB_PORT", 5433)
    db_name: str = os.getenv("DB_NAME", "ragdb")
    db_user: str = os.getenv("DB_USER", "rag")
    db_password: str = os.getenv("DB_PASSWORD", "rag")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    embed_batch_size: int = _int("EMBED_BATCH_SIZE", 64)
    device: str = os.getenv("DEVICE", "auto")
    # bge-*-en-v1.5 is trained asymmetric: an instruction prefix on the QUERY only
    # (never on passages) for s2p retrieval. Toggle + string; default set by the A/B
    # in scripts/instruction_ab.py. Ignored for non-bge models.
    use_query_instruction: bool = os.getenv("USE_QUERY_INSTRUCTION", "true").lower() == "true"
    query_instruction: str = os.getenv(
        "QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
    )
    # Inference precision, held constant across ALL model runs (embedding AND the
    # cross-encoder reranker) so the latency comparison is apples-to-apples. fp16
    # on this GPU is ~3.9x faster than fp32 with top-10 retrieval ranking 99.7%
    # identical (see scripts/dtype_bench.py). Recorded per run in eval_runs.dtype.
    torch_dtype: str = os.getenv("TORCH_DTYPE", "fp16")

    chunk_size: int = _int("CHUNK_SIZE", 500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 50)

    rrf_k: int = _int("RRF_K", 60)
    retrieval_depth: int = _int("RETRIEVAL_DEPTH", 100)  # candidates each arm returns before fusion
    top_k: int = _int("TOP_K", 50)                       # hybrid candidates handed to the reranker
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

    # Gold is a CHARACTER SPAN (source_document_id + gold_char_start/end in cleaned
    # text). A retrieval is correct iff a retrieved chunk FULLY CONTAINS that span
    # (chunk.char_start <= gold_start and chunk.char_end >= gold_end) — not partial
    # overlap. Span-based gold is what makes cross-chunk-size comparison valid.
    correctness_rule: str = os.getenv("CORRECTNESS_RULE", "span_contains")

    # --- Generation (Phase 5) ---
    # Provider is a seam: 'openai' is wired today, 'anthropic' is reserved.
    # The model id is a config value, never hardcoded at the call site.
    generation_provider: str = os.getenv("GENERATION_PROVIDER", "openai")
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-5.6-terra")

    def embedding_column(self, model: str | None = None) -> str:
        m = model or self.embedding_model
        if m not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model {m!r}. "
                f"Known: {list(EMBEDDING_MODELS)}. Add a column in db/schema.sql to support more."
            )
        return EMBEDDING_MODELS[m]["column"]

    def embedding_dim(self, model: str | None = None) -> int:
        m = model or self.embedding_model
        return EMBEDDING_MODELS[m]["dim"]


settings = Settings()
