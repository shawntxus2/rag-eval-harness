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
Generation (Phase 5): OpenAI (gpt-5.6-terra, configurable) with required citations
UI (Phase 6): Streamlit — one page demo + an evaluation-table tab
```

- **Datastore:** PostgreSQL 17 + pgvector only. Vectors, full-text index, chunk
  metadata, and eval results all live here. No separate vector DB. Runs in Docker.
- **Embeddings:** local, via `sentence-transformers`. No embedding API calls. Uses
  CUDA if present, else MPS (Apple Silicon), else CPU.
- **Generation:** OpenAI, mid-tier `gpt-5.6-terra` by default (configurable via
  `GENERATION_MODEL`). The provider is a thin seam (`rag/generation.py`, selected by
  `GENERATION_PROVIDER`): OpenAI is wired, Anthropic is a reserved branch. **Retrieval
  evaluation (Phases 1–4) does not call it** — recall/latency never depend on an API key.
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
| `GENERATION_PROVIDER` | `openai` | generation provider seam (`openai` wired; `anthropic` reserved) |
| `GENERATION_MODEL` | `gpt-5.6-terra` | generation model (must be reachable by your `OPENAI_API_KEY`) |
| `OPENAI_API_KEY` | — | OpenAI key; read from env, never committed. Missing/rejected → fails loudly |

### Swapping the embedding model

Set `EMBEDDING_MODEL` to another supported model and re-run embed. Each model writes
to its own fixed-dimension column on `chunks` (`embedding_bge` 768, `embedding_minilm`
384), each with its own partial HNSW index, so both can coexist and be compared.
Adding a *new* model = add a column + partial HNSW index in `db/schema.sql` and an
entry in `EMBEDDING_MODELS` in `rag/config.py`.

## The ML stack pins: two sets, and why they differ

The ML stack (`torch`/`torchvision`/`sentence-transformers`/`transformers`) has to
co-resolve as one set, and the right set depends entirely on the host. There are two:

**Intel macOS set (bootstrap host, CPU only — torch ceiling 2.2.2):**

```
torch==2.2.2          # last release with an Intel-macOS (x86_64) wheel
sentence-transformers==3.3.1
transformers==4.46.3
numpy==1.26.4
```

`2.2.2` is a *development-machine floor*, not a production choice: it is the highest
`torch` that installs on an Intel Mac at all (anything newer has no x86_64 wheel and
fails at install time). This set is CPU-only and is **not** what ships in
`requirements.txt` anymore — it is recorded here as the origin.

**arm64 + CUDA set (this machine — NVIDIA DGX Spark, GB10 Grace-Blackwell, sm_121,
CUDA 13 — what `requirements.txt` now pins):**

```
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.14.1+cu130
torchvision==0.29.1+cu130
sentence-transformers==6.1.0
transformers==5.18.0
numpy==2.5.3
```

Why they differ: the 2.2.2 pin **cannot work here** — there is no aarch64 CUDA wheel
for it, and a Blackwell GPU needs a recent CUDA toolkit and a correspondingly recent
`torch`. The DGX Spark is supported by PyTorch's **stable aarch64 `cu130` wheels**
(the public build that matches the Spark's CUDA 13 driver; the NGC
`nvcr.io/nvidia/pytorch` container ships the pre-release `2.10.0a0…nv` equivalent).
Bumping `torch` dragged the whole set forward: `transformers` and
`sentence-transformers` moved to current majors and `numpy` to 2.x. The `+cu130` local
tags force pip to fetch the CUDA aarch64 wheels from the PyTorch index — never a CPU
wheel from PyPI.

Note on compute capability: GB10 is **sm_121**, but the `cu130` wheel's `arch_list`
tops out at **sm_120** (no sm_121 cubin, no `compute_120` PTX). It still runs — sm_120
binaries are forward-compatible within the Blackwell 12.x major — which is why the pin
must be *verified with a real on-device kernel*, not just a successful install. Do that
with `python scripts/gpu_check.py` (fails loudly, never falls back to CPU).

Install flow on the Spark:

```bash
# torch/torchvision come from the cu130 index; the +cu130 pins make that unambiguous.
pip install -r requirements.txt
# Prove the GPU is real (device matmul verified vs CPU + a real embedding pass):
python scripts/gpu_check.py
```

With `DEVICE=auto`, the embed/rerank code picks CUDA when `torch.cuda.is_available()`,
else MPS on Apple Silicon, else CPU.

### Embedding batch size on this hardware (`EMBED_BATCH_SIZE`, default 64)

The default `64` was sized for a laptop CPU. On the GB10 the constraint inverts:
**memory is a non-constraint** (unified 122 GiB pool; bge-base uses ~2.3 GiB per 1024
~60-token chunks, so batches of many thousands fit), but **throughput does not improve
with larger batches** — it is flat from batch 64→1024 (~710 embeddings/s for ~60-token
chunks) and *degrades* past 1024 (the GPU is already compute-saturated at 64). Recommended
`EMBED_BATCH_SIZE` here is **256** (amortizes per-call overhead vs 64 at ~1 GiB cost,
no throughput penalty); anything in 128–512 is equivalent, and going to the thousands
only wastes memory. (Left at the default in code; this is a recommendation.)

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
