-- RAG evaluation harness schema
-- Idempotent: safe to run on first boot (initdb) and to re-apply via scripts/init_db.py.
-- Postgres 17 + pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- documents: one row per source PDF. Failed parses are recorded, not hidden.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename      TEXT        NOT NULL,
    rel_path      TEXT        NOT NULL,
    sha256        TEXT        NOT NULL UNIQUE,     -- content hash → dedup across re-runs
    num_pages     INTEGER,
    size_bytes    BIGINT,
    parse_status  TEXT        NOT NULL DEFAULT 'pending'
                   CHECK (parse_status IN ('pending','ok','failed')),
    parse_error   TEXT,                            -- populated iff parse_status='failed'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- ingestion_runs: one row per chunking configuration (chunk_size, overlap).
-- The 250/500/1000-token runs each get a row so chunks are traceable to a run.
-- Embedding model is NOT part of a run — a chunk is embedded by every model
-- into its own column (see chunks), so models are compared on identical chunks.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    label         TEXT        NOT NULL UNIQUE,     -- e.g. 'chunk-500-overlap-50'
    chunk_size    INTEGER     NOT NULL,            -- target tokens per chunk
    overlap       INTEGER     NOT NULL,            -- token overlap between chunks
    tokenizer     TEXT        NOT NULL,            -- tokenizer used to measure size
    status        TEXT        NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','embedding','ready','failed')),
    num_chunks    INTEGER     NOT NULL DEFAULT 0,
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- chunks: one row per chunk. Carries text, full provenance, a tsvector for
-- lexical search, and one nullable vector column per supported embedding model.
-- Two models → two dimensions (bge=768, MiniLM=384); a single pgvector column
-- is fixed-dimension, so each model gets its own column + partial HNSW index.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id       BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ingestion_run_id  BIGINT NOT NULL REFERENCES ingestion_runs(id) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL,            -- 0-based order within (run, document)
    page_number       INTEGER,                     -- 1-based page the chunk starts on
    page_end          INTEGER,                     -- 1-based page the chunk ends on
    char_start        INTEGER NOT NULL,            -- offset into the document's cleaned text
    char_end          INTEGER NOT NULL,
    token_count       INTEGER NOT NULL,
    text              TEXT    NOT NULL,
    tsv               tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding_bge     vector(768),                 -- BAAI/bge-base-en-v1.5
    embedding_minilm  vector(384),                 -- sentence-transformers/all-MiniLM-L6-v2
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (ingestion_run_id, document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin        ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_run_idx        ON chunks (ingestion_run_id);
CREATE INDEX IF NOT EXISTS chunks_doc_idx        ON chunks (document_id);

-- Partial HNSW indexes: only rows that actually have a vector are indexed.
-- Cosine ops; vectors are L2-normalized on write so cosine == dot product.
CREATE INDEX IF NOT EXISTS chunks_bge_hnsw
    ON chunks USING hnsw (embedding_bge vector_cosine_ops)
    WHERE embedding_bge IS NOT NULL;

CREATE INDEX IF NOT EXISTS chunks_minilm_hnsw
    ON chunks USING hnsw (embedding_minilm vector_cosine_ops)
    WHERE embedding_minilm IS NOT NULL;

-- ---------------------------------------------------------------------------
-- eval_questions: the evaluation set. Gold location is stored as a document +
-- page + character span so "correct retrieval" can be judged against chunks in
-- ANY chunk-size run (the same answer maps to different chunk rows per size).
-- answer_chunk_id is a convenience pointer to the chunk it was generated from.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_questions (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    question            TEXT    NOT NULL,
    source_document_id  BIGINT  REFERENCES documents(id) ON DELETE SET NULL,
    gold_page           INTEGER,
    gold_char_start     INTEGER,                   -- span in the document's cleaned text
    gold_char_end       INTEGER,
    answer_chunk_id     BIGINT  REFERENCES chunks(id) ON DELETE SET NULL,  -- provenance: the chunk/run the question was authored against (NOT the scoring key)
    generated_from_run  BIGINT  REFERENCES ingestion_runs(id) ON DELETE SET NULL,
    generated_by        TEXT    NOT NULL DEFAULT 'openai'
                         CHECK (generated_by IN ('openai','claude','human')),
    generator_model     TEXT,                      -- e.g. gpt-5.6-terra
    machine_generated   BOOLEAN NOT NULL DEFAULT TRUE,
    human_verified      BOOLEAN NOT NULL DEFAULT FALSE,
    status              TEXT    NOT NULL DEFAULT 'candidate'
                         CHECK (status IN ('candidate','approved','rejected')),
    verified_at         TIMESTAMPTZ,
    uniqueness_anchor   TEXT,                      -- the detail that makes the question answerable from exactly one document (section/rule no., date, threshold, named entity, defined term)
    gold_overlap        DOUBLE PRECISION,          -- content-word overlap of question vs gold span (paraphrase check; high => quoting)
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Scoring reads the gold CHARACTER SPAN (source_document_id + gold_char_start/end in
-- cleaned text), not answer_chunk_id, so one question scores across all chunk sizes.
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS uniqueness_anchor TEXT;
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS gold_overlap DOUBLE PRECISION;
ALTER TABLE eval_questions ADD COLUMN IF NOT EXISTS generator_model TEXT;
ALTER TABLE eval_questions DROP CONSTRAINT IF EXISTS eval_questions_generated_by_check;
ALTER TABLE eval_questions ADD CONSTRAINT eval_questions_generated_by_check CHECK (generated_by IN ('openai','claude','human'));

CREATE INDEX IF NOT EXISTS eval_questions_verified_idx ON eval_questions (human_verified, status);

-- ---------------------------------------------------------------------------
-- eval_runs: one row per (retrieval_mode × chunk_size × embedding_model)
-- evaluation. This is the deliverable table — the comparison numbers.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_label          TEXT,
    retrieval_mode     TEXT NOT NULL
                        CHECK (retrieval_mode IN ('vector','lexical','hybrid','hybrid_rerank')),
    chunk_size         INTEGER NOT NULL,
    ingestion_run_id   BIGINT REFERENCES ingestion_runs(id) ON DELETE SET NULL,
    embedding_model    TEXT NOT NULL,
    rerank_model       TEXT,                        -- null unless mode='hybrid_rerank'
    rrf_k              INTEGER,                      -- null unless a fusion mode
    top_k              INTEGER NOT NULL,            -- candidate depth used for the run
    correctness_rule   TEXT NOT NULL,               -- 'span_contains' (chosen: chunk fully contains the gold span) | 'span_overlap' | 'exact_chunk' | 'same_page'
    dtype              TEXT,                         -- inference precision that produced the latency ('fp16'|'bf16'|'fp32')
    num_questions      INTEGER NOT NULL,
    recall_at_5        DOUBLE PRECISION,
    recall_at_10       DOUBLE PRECISION,
    mrr                DOUBLE PRECISION,
    latency_p50_ms     DOUBLE PRECISION,
    latency_p95_ms     DOUBLE PRECISION,
    params             JSONB,                       -- full config snapshot for reproducibility
    git_sha            TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Backfill the dtype column onto any eval_runs table created before it existed
-- (CREATE TABLE IF NOT EXISTS above is a no-op on an existing table).
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS dtype TEXT;

-- Per-question detail behind each eval_run, for drill-down and debugging.
CREATE TABLE IF NOT EXISTS eval_results (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    eval_run_id      BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id      BIGINT NOT NULL REFERENCES eval_questions(id) ON DELETE CASCADE,
    gold_rank        INTEGER,                       -- 1-based rank of first correct chunk; NULL if absent
    hit_at_5         BOOLEAN NOT NULL,
    hit_at_10        BOOLEAN NOT NULL,
    reciprocal_rank  DOUBLE PRECISION NOT NULL,     -- 0 if the gold chunk was not retrieved
    retrieval_ms     DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (eval_run_id);
