# rag-eval-harness

A retrieval-augmented generation system built from primitives, with a measurement
harness that compares four retrieval strategies on the same question set over
**your own** PDF corpus.

The point is not a demo. The point is numbers: recall and latency for
**vector-only**, **lexical-only**, **hybrid (RRF)**, and **hybrid + reranking**,
across three chunk sizes, on your documents.

## Architecture (at a glance)

```
corpus/*.pdf
   │  (Phase 1) extract text + page map, strip furniture, chunk @ 250/500/1000 tok
   ▼
Postgres + pgvector  ── documents, ingestion_runs, chunks (text + tsvector + vectors),
   │                     eval_questions, eval_runs, eval_results
   │  (Phase 2) embed locally with sentence-transformers (bge-base / MiniLM), normalized
   ▼
Retrieval (Phase 3): vector (HNSW cosine) · lexical (ts_rank_cd) · hybrid (RRF, k=60) ·
   │                  hybrid+rerank (bge-reranker-base cross-encoder, top 50 → top 5)
   ▼
Evaluation (Phase 4): recall@5/10, MRR, latency p50/p95  → eval_runs  → comparison table
   │
   ▼
Generation (Phase 5): Claude (claude-sonnet-5, configurable) with required citations
UI (Phase 6): Streamlit — one page demo + an evaluation-table tab
```

- **Datastore:** PostgreSQL 17 + pgvector only. Vectors, full-text index, chunk
  metadata, and eval results all live here. No separate vector DB. Runs in Docker.
- **Embeddings:** local, via `sentence-transformers`. No embedding API calls. Uses
  CUDA if present, else MPS (Apple Silicon), else CPU.
- **Generation:** Claude API, mid-tier `claude-sonnet-5` by default (configurable via
  `CLAUDE_MODEL`).
- **Core pipeline is hand-built** — chunking, retrieval, RRF fusion, and reranking are
  written here, not pulled from LangChain/LlamaIndex/Haystack.

## Quick start

> **Prerequisite: the Docker daemon must be running before anything else.**
> On macOS that means Docker Desktop must be launched (`open -a Docker`) and fully
> started — the `docker` CLI being on `PATH` is not enough; if the daemon is down,
> `docker compose up` fails with a "cannot connect to the Docker API" socket error.

```bash
# 1. Bring up Postgres + pgvector (schema auto-applies on first boot)
docker compose up -d

# 2. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see the torch note below on arm64 + CUDA

# 3. (Re)apply schema and print it
python scripts/init_db.py
```

Then drop PDFs into `corpus/` — later phases add the single command that ingests,
embeds, evaluates, and serves.

## Configuration

Every parameter is an environment variable (see `.env.example`). Copy it to `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `DB_*` | see file | Postgres connection (host port 5433) |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | embedding model; also supports `sentence-transformers/all-MiniLM-L6-v2` |
| `DEVICE` | `auto` | `auto`/`cuda`/`mps`/`cpu` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | chunk grain (tokens) |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `TOP_K` | `50` | candidate depth (hybrid → rerank takes top 50) |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder reranker |
| `CORRECTNESS_RULE` | `span_overlap` | what counts as a correct retrieval |
| `CLAUDE_MODEL` | `claude-sonnet-5` | generation model |

### Swapping the embedding model

Set `EMBEDDING_MODEL` to another supported model and re-run embed. Each model writes
to its own fixed-dimension column on `chunks` (`embedding_bge` 768, `embedding_minilm`
384), each with its own partial HNSW index, so both can coexist and be compared.
Adding a *new* model = add a column + partial HNSW index in `db/schema.sql` and an
entry in `EMBEDDING_MODELS` in `rag/config.py`.

## The `torch` pin, and why it MUST change on arm64 + CUDA

`requirements.txt` pins **`torch==2.2.2`**. That number is a *development-machine
floor*, not the intended production version:

- **2.2.2 is the last `torch` release with an Intel-macOS (`x86_64`) wheel.** The repo
  was bootstrapped on an Intel Mac, where `pip` offers nothing newer than 2.2.2 —
  `torch==2.5.1` has **no installable wheel there** and fails at install time. 2.2.2 is
  the highest version that installs on that host at all.
- On generic **Linux + CUDA (x86_64)**, PyPI's 2.2.2 wheels are CUDA-enabled, so the
  pin happens to work there unchanged.
- **On arm64 + CUDA — including the NVIDIA DGX Spark (GB10 Grace-Blackwell) — the
  2.2.2 pin WILL NOT WORK and MUST be bumped.** There is no aarch64 CUDA wheel for
  2.2.2, and even if there were, Blackwell-class GPUs need a recent CUDA toolkit and a
  correspondingly recent `torch`. Install the arm64 + CUDA build NVIDIA recommends for
  the Spark instead. Typical flow:

  ```bash
  # On the DGX Spark: install the CUDA/aarch64 torch FIRST, from the PyTorch index
  # matching the Spark's CUDA toolkit (use NVIDIA's recommended build/index for GB10).
  pip install torch --index-url https://download.pytorch.org/whl/cu128   # example index

  # Then edit requirements.txt: delete (or bump) the `torch==2.2.2` line so the rest of
  # the install does not drag torch back down to 2.2.2, and install everything else:
  pip install -r requirements.txt
  ```

  Verify GPU is live before embedding:
  `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

With `DEVICE=auto`, the embed/rerank code picks CUDA when `torch.cuda.is_available()`,
else MPS on Apple Silicon, else CPU.

## Schema

Six tables (defined in `db/schema.sql`, applied idempotently). Your four core tables
plus `ingestion_runs` (so the three chunk-size runs are first-class, joinable rows) and
`eval_results` (per-question drill-down behind each eval run).

| Table | One row per… | Notable columns |
|---|---|---|
| `documents` | source PDF | `sha256` (unique → dedup), `num_pages`, `parse_status`, `parse_error` — **failed parses are recorded, not hidden** |
| `ingestion_runs` | chunking configuration (250/500/1000 tok) | `chunk_size`, `overlap`, `tokenizer`, `status`, `num_chunks` |
| `chunks` | chunk | `document_id`, `ingestion_run_id`, `chunk_index`, `page_number`/`page_end`, `char_start`/`char_end`, `token_count`, `text`, `tsv` (GIN), `embedding_bge vector(768)` + `embedding_minilm vector(384)` (each partial-HNSW, cosine) |
| `eval_questions` | evaluation question | `question`, gold location (`source_document_id` + `gold_page` + `gold_char_start`/`gold_char_end`), `answer_chunk_id`, `generated_by`, `machine_generated`, `human_verified`, `status` |
| `eval_runs` | retrieval_mode × chunk_size × embedding_model | `recall_at_5`, `recall_at_10`, `mrr`, `latency_p50_ms`, `latency_p95_ms`, `rerank_model`, `rrf_k`, `correctness_rule`, `params` (jsonb snapshot) — **the deliverable table** |
| `eval_results` | (eval_run × question) | `gold_rank`, `hit_at_5`, `hit_at_10`, `reciprocal_rank`, `retrieval_ms` |

Indexes: `chunks_tsv_gin` (GIN on the tsvector), `chunks_bge_hnsw` and
`chunks_minilm_hnsw` (partial HNSW, cosine ops — vectors are L2-normalized on write so
cosine equals dot product), plus btrees on the run/doc foreign keys.

### Why gold answers are anchored to document + page + char offsets, not `chunk_id`

A single `chunk_id` cannot be "the correct chunk" across all three chunk sizes: the
text that answers a question lands in **different chunk rows** at 250 vs 500 vs 1000
tokens (and would change again if chunking were re-run). If the gold answer were a hard
`chunk_id`, every evaluation would be locked to the one run that generated it.

Instead, the gold answer is stored as a **location in the document** —
`source_document_id` + `gold_page` + `gold_char_start`/`gold_char_end` — which is
stable regardless of how the text is chunked. "Did retrieval find the right chunk?" is
then a predicate evaluated at eval time against whatever run is under test
(`CORRECTNESS_RULE`: `exact_chunk` / `span_overlap` / `same_page`). `answer_chunk_id` is
retained only as a convenience pointer back to the chunk the question was generated
from. This keeps "what counts as a correct retrieval" a deliberate, swappable decision
rather than something baked into the question set.

## Measurement caveat: CPU vs GPU

The repo was built and Phase 0 was verified on an **Intel Mac with no NVIDIA GPU**. Any
embedding/retrieval **latency measured on that host is CPU-only** and is not
representative of the GPU numbers. Phase 2+ latency figures should be taken on the DGX
Spark (CUDA) for the comparison to mean anything. The database layer (pgvector HNSW /
GIN query latency) is comparable across hosts; the model inference (embedding, cross-
encoder reranking) is what shifts dramatically from CPU to GPU.

## Phase 0 operational notes (so a fresh machine reproduces cleanly)

- **Docker daemon first** (see Quick start) — the most common first-run failure is the
  CLI being present while Docker Desktop isn't started.
- **Postgres data is a Docker named volume, not a file in this tree.** It does not move
  with `git` or `rsync`. On a new machine, `docker compose up -d` creates a fresh volume
  and re-runs `db/schema.sql` via initdb — expected, since there is no data to preserve
  yet (Phase 0 only creates empty tables).
- **Schema is idempotent** — both the first-boot initdb apply and
  `python scripts/init_db.py` re-apply run without error.
- **`torch` pin** — see the dedicated section above; bump it on the Spark.

## Status

- [x] **Phase 0** — infrastructure, schema, pinned deps
- [ ] Phase 1 — ingest & chunking
- [ ] Phase 2 — embedding & storage
- [ ] Phase 3 — four retrieval modes
- [ ] Phase 4 — evaluation (the deliverable)
- [ ] Phase 5 — generation
- [ ] Phase 6 — Streamlit UI
- [ ] Domain-drift finding (written up here)

## Results

_Populated in Phase 4 — which configuration won, by what margin, and what I'd change
if the corpus doubled._
