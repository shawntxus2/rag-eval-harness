"""Phase 3 — four retrieval modes over one (ingestion_run × embedding_model).

All four share one signature — (conn, query, run_id, depth) -> list[Hit] ranked by
score desc:
  * vector        — HNSW cosine over the model's embedding column
  * lexical       — ts_rank_cd over the tsv column (same text-search config as tsv)
  * hybrid        — Reciprocal Rank Fusion (k=RRF_K) of vector + lexical
  * hybrid_rerank — bge cross-encoder reranks the top TOP_K of hybrid

The embedding model and the cross-encoder are loaded ONCE, at construction — a
cross-encoder reloaded per query makes every latency number meaningless. Query and
chunk symmetry (same model, dtype, normalization; same ts config) is asserted, not
assumed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from rag.config import settings
from rag.embed import load_model, resolve_device

_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class Hit:
    chunk_id: int
    score: float


class Retriever:
    def __init__(self, embedding_model: str | None = None, rerank_model: str | None = None,
                 use_instruction: bool | None = None):
        self.embedding_model = embedding_model or settings.embedding_model
        self.column = settings.embedding_column(self.embedding_model)
        self.ts_config = "english"  # must match chunks.tsv: to_tsvector('english', text)
        self.device = resolve_device()
        self.embed_model = load_model(self.embedding_model)          # fp16, loaded once
        self.reranker = CrossEncoder(rerank_model or settings.rerank_model, device=self.device)
        if settings.torch_dtype == "fp16" and self.device in ("cuda", "mps"):
            self.reranker.model.half()                               # reranker at the held dtype
        # bge asymmetric query instruction (query only). Off for non-bge models.
        use = settings.use_query_instruction if use_instruction is None else use_instruction
        self.query_instruction = (
            settings.query_instruction if (use and "bge" in self.embedding_model.lower()) else ""
        )

    # ---- session + symmetry assertions ----
    def configure_session(self, conn) -> None:
        # With per-run partial indexes (scripts/build_run_indexes.py) each query hits
        # an index holding only its own run, so no iterative scan is needed — latency
        # no longer depends on run selectivity. A wide ef_search keeps recall high.
        conn.execute(f"SET hnsw.ef_search = {max(100, settings.retrieval_depth * 2)}")

    def assert_symmetry(self, conn) -> None:
        expr = conn.execute(
            """SELECT pg_get_expr(adbin, adrelid) FROM pg_attrdef ad
               JOIN pg_attribute a ON a.attrelid=ad.adrelid AND a.attnum=ad.adnum
               WHERE a.attrelid='chunks'::regclass AND a.attname='tsv'"""
        ).fetchone()[0]
        assert f"'{self.ts_config}'" in expr, (
            f"lexical ts config {self.ts_config!r} does not match tsv definition: {expr}"
        )

    # ---- query embedding (same model/dtype/normalization as chunks) ----
    def embed_query(self, query: str) -> np.ndarray:
        instr = self.query_instruction
        if instr and not instr.endswith(" "):
            instr += " "
        q = instr + query
        v = self.embed_model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        v = v.astype(np.float32)
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) <= 1e-3, f"query vector not normalized: ‖q‖={norm}"
        return v

    # ---- the four modes, identical signature ----
    def vector(self, conn, query: str, run_id: int, depth: int | None = None) -> list[Hit]:
        depth = depth or settings.retrieval_depth
        qv = self.embed_query(query)
        # run_id is inlined (it is our own int) so the WHERE predicate is a literal the
        # planner can match to the per-run partial HNSW index's predicate.
        rows = conn.execute(
            f"SELECT id, 1 - ({self.column} <=> %s) AS score FROM chunks "
            f"WHERE ingestion_run_id = {int(run_id)} AND {self.column} IS NOT NULL "
            f"ORDER BY {self.column} <=> %s LIMIT %s",
            (qv, qv, depth),
        ).fetchall()
        return [Hit(r[0], float(r[1])) for r in rows]

    def lexical(self, conn, query: str, run_id: int, depth: int | None = None) -> list[Hit]:
        depth = depth or settings.retrieval_depth
        # OR-of-terms (not AND): derive the 'english'-normalized lexemes with
        # plainto_tsquery (same config as tsv), then OR them so a chunk matching
        # any term ranks, as a real keyword search does. ts_rank_cd scores by term
        # frequency and proximity. AND semantics would give ~zero recall on NL questions.
        rows = conn.execute(
            "WITH q AS (SELECT replace(plainto_tsquery(%s, %s)::text, ' & ', ' | ')::tsquery AS tsq) "
            "SELECT c.id, ts_rank_cd(c.tsv, q.tsq) AS score "
            "FROM chunks c, q "
            "WHERE c.ingestion_run_id = %s AND q.tsq <> '' AND c.tsv @@ q.tsq "
            "ORDER BY score DESC, c.id LIMIT %s",
            (self.ts_config, query, run_id, depth),
        ).fetchall()
        return [Hit(r[0], float(r[1])) for r in rows]

    def hybrid(self, conn, query: str, run_id: int, depth: int | None = None,
               weights: list[float] | None = None) -> list[Hit]:
        depth = depth or settings.retrieval_depth
        vec = self.vector(conn, query, run_id, depth)
        lex = self.lexical(conn, query, run_id, depth)
        w = weights or [settings.rrf_weight_vector, settings.rrf_weight_lexical]
        return _rrf([vec, lex], settings.rrf_k, w)[:depth]

    def _rerank(self, conn, query: str, cand: list[Hit]) -> list[Hit]:
        """Cross-encoder rescoring of the top TOP_K candidates (shared by both rerank modes)."""
        ids = [h.chunk_id for h in cand[: settings.top_k]]
        if not ids:
            return []
        text_by_id = dict(
            conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()
        )
        scores = self.reranker.predict([(query, text_by_id[i]) for i in ids])
        return [Hit(i, float(s)) for i, s in sorted(zip(ids, scores), key=lambda t: -float(t[1]))]

    def hybrid_rerank(self, conn, query: str, run_id: int, depth: int | None = None) -> list[Hit]:
        return self._rerank(conn, query, self.hybrid(conn, query, run_id, depth or settings.retrieval_depth))

    def vector_rerank(self, conn, query: str, run_id: int, depth: int | None = None) -> list[Hit]:
        # Cross-encoder over VECTOR's top candidates — avoids handing the reranker the
        # lexical arm's damaged list (which hurts on descriptive queries).
        return self._rerank(conn, query, self.vector(conn, query, run_id, depth or settings.retrieval_depth))


def _rrf(lists: list[list[Hit]], k: int, weights: list[float] | None = None) -> list[Hit]:
    """Weighted Reciprocal Rank Fusion: score(d) = Σ w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(lists)
    agg: dict[int, float] = defaultdict(float)
    for w, lst in zip(weights, lists):
        for rank, hit in enumerate(lst, start=1):
            agg[hit.chunk_id] += w / (k + rank)
    return [Hit(cid, s) for cid, s in sorted(agg.items(), key=lambda t: -t[1])]
