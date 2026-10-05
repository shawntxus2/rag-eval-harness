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

    chunk_size: int = _int("CHUNK_SIZE", 500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 50)

    rrf_k: int = _int("RRF_K", 60)
    top_k: int = _int("TOP_K", 50)
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

    correctness_rule: str = os.getenv("CORRECTNESS_RULE", "span_overlap")

    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

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
