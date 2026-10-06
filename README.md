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
| `RETRIEVAL_DEPTH` | `100` | candidates each arm returns before fusion |
| `TOP_K` | `50` | hybrid candidates handed to the reranker |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder reranker |
| `CORRECTNESS_RULE` | `span_contains` | hit = a top-k chunk from the gold doc fully contains the gold span |
| `TORCH_DTYPE` | `fp16` | inference precision, held constant across all runs (embedding + reranker) |
| `USE_QUERY_INSTRUCTION` | `true` | apply the bge s2p query instruction (query only); chosen by A/B |
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

### Embedding dtype — the knob that actually matters here (`scripts/dtype_bench.py`)

The forward pass runs in **fp32** by default, and on this GPU dtype buys far more
than batch size. Measured on 288 real ~500-token corpus chunks (bge-base, batch 64):

| dtype | tokens/s | speedup | min cosine vs fp32 | top-10 ranking overlap vs fp32 |
|---|---|---|---|---|
| fp32 | 45,550 | 1.0× | 1.00000 | 100% |
| bf16 | 176,823 | 3.9× | 0.99803 | 97.7% |
| **fp16** | **177,016** | **3.9×** | **0.99976** | **99.7%** |

Decision: **fp16 is the default (`TORCH_DTYPE`), held constant across every run** —
embedding *and* the cross-encoder reranker — so the latency comparison (a deliverable)
is apples-to-apples. fp16 is ~3.9× faster than fp32 here for essentially no retrieval
cost (top-10 neighbours 99.7% identical to fp32; per-vector cosine 0.9998). bf16 is the
same speed but agrees less well (97.7% top-10; fewer mantissa bits). The reranker, not
ingest, is why this matters — it dominates query latency. Batch size, by contrast, buys
~0×. Every `eval_runs` row records the `dtype` that produced its latency.
`scripts/dtype_bench.py` reproduces the table. (Caveat: a synthetic near-duplicate
corpus understates ranking agreement badly — measure on real chunks.)

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

## Phase 1 — extraction, furniture, chunking

**Extraction is held constant.** Every document is extracted with pypdf's default
`extract_text()` (`rag/extract.py`) — one extractor, one setting, no per-publisher
code — so that when a retrieval number looks wrong the extractor is never a suspect.
pypdf reads single- and three-column pages in correct reading order; on Federal
Register pages it hoists footnotes above the body (accepted: the text is real and
correctly attributed, the only cost is chunk coherence). pdfplumber's default was
rejected — it interleaves the three FR columns into fluent-looking nonsense.

**Furniture removal** (`rag/furniture.py`) strips, without touching body or footnotes:
1. Running headers/footers — a line in the top-3 or bottom-3 non-empty lines of a page
   whose digit-masked key recurs on ≥50% of pages. Leading/trailing page-number tokens
   are trimmed before keying so a page number that alternates prefix/suffix across
   pages (BIS) collapses to one key instead of two sub-threshold variants.
2. Federal Register typesetting footers (lines beginning `VerDate`).
3. The FR vertical-sidebar watermark `<user> on DSK…PROD with RULES#`, removed as a
   substring wherever it occurs — it also appears standalone on ~71 pages that the
   `VerDate` anchor alone would miss.

Footnotes are **kept** (they carry definitions eval questions target). The first
occurrence of each furniture line is preserved in every document and only its
subsequent repeats are stripped, so no line is ever wholly deleted from the corpus
and the detector never has to judge whether a repeated line is substantive.
`scripts/ingest.py` prints the per-document removal report; removal is ≤8% on every
document, and anything >15% aborts the load for review rather than silently over-stripping.

**Offset basis (important when tracing a chunk back to a page).**
`chunks.char_start`/`char_end` are offsets into the **cleaned text** — after furniture
removal, because that is what chunks are cut from — **not** raw-PDF offsets. The cleaned
text is persisted per document at `corpus/cleaned/<name>.txt` (gitignored, regenerable
by re-running ingest) so those offsets always resolve to something. `page_number`/
`page_end` are mapped from per-page char spans recorded during cleaning.

**Chunking** (`rag/chunk.py`) uses sliding windows measured in **bge-base tokenizer
tokens** (so "size" means what the embedder sees), three runs at 250/500/1000 tokens
with 10% overlap (25/50/100), loaded into `documents` / `ingestion_runs` / `chunks`
(27 docs → 10,025 / 5,018 / 2,515 chunks). Embedding columns are left NULL for Phase 2.

## Phase 2 — embedding & storage

Every chunk is embedded by **both** models (`scripts/embed.py` → `rag/embed.py`):
`BAAI/bge-base-en-v1.5` (768-dim) and `all-MiniLM-L6-v2` (384-dim), each into its own
column, at the held-constant `fp16` precision, L2-normalized on write so cosine
distance is a dot product (what the partial HNSW `vector_cosine_ops` indexes assume).
Passages carry no instruction prefix (bge's query instruction is applied to queries
in Phase 3). The script is resumable (only embeds NULL columns).

**Normalization is verified, not assumed** — after writing, vectors are re-read from
Postgres and their L2 norm checked: `max|‖v‖−1|` was 4.8e-4 (bge) / 4.7e-4 (MiniLM),
within a 1e-3 tolerance (residual is fp16→float32 rounding). All 17,567 chunks embedded
in both columns, 0 NULLs; a silent normalization failure would have degraded every
cosine result with no error, so this check gates Phase 3.

## Phase 3 — four retrieval modes

`rag/retrieval.py` exposes four modes with one signature — `(conn, query, run_id, depth)
→ list[Hit]` ranked by score:
- **vector** — HNSW cosine over the model's embedding column
- **lexical** — `ts_rank_cd` over the `tsv` column; the query is reduced to
  `english`-normalized lexemes (same config as `tsv`) and **OR-ed, not AND-ed**, so a
  chunk matching any term ranks (AND semantics gives ~zero recall on NL questions).
  **Caveat (stated for credibility): this is Postgres full-text search with
  `ts_rank_cd`, which has no IDF term weighting — it is not BM25.** That is why the
  lexical arm can surface passages dense in common words; a true BM25 implementation
  would narrow the gap against hybrid. The arm is left as-is and named honestly.
- **hybrid** — Reciprocal Rank Fusion (`RRF_K=60`) of the vector and lexical lists
- **hybrid_rerank** — a `bge-reranker-base` cross-encoder rescoring of hybrid's top `TOP_K`

Getting it right:
1. **Depth.** Each arm returns its top `RETRIEVAL_DEPTH=100` (not 10), so RRF fuses real
   lists and the reranker has 50 real candidates to work from.
2. **Query/chunk symmetry, asserted not assumed.** The query is embedded with the same
   model, dtype (fp16) and normalization as the chunks; `‖q‖≈1` is asserted per query, and
   the lexical config is asserted to match the `tsv` generation expression. The bge s2p
   query instruction (`"Represent this sentence for searching relevant passages: "`, query
   only — per the model card) is **ON** (`USE_QUERY_INSTRUCTION`), chosen by A/B in
   `scripts/instruction_ab.py`: over 19 matched (query, gold) pairs, vector-only with the
   instruction beat without on 6 pairs, lost on 2, tied on 11 (top-1 2 vs 1, top-5 4 vs 1),
   winning on exactly the paraphrase queries it should. This follows documented bge usage
   rather than deviating from it.
   **Vector latency confound removed:** rather than filter one shared all-runs HNSW index
   (whose latency then tracked run *selectivity* — the 14%-of-corpus 1000-run was the
   *slowest* at 3.98 ms, not the fastest), there is now one **partial HNSW index per run
   per model** (`scripts/build_run_indexes.py`, six indexes), the run_id is inlined so the
   planner matches each, and `iterative_scan` is dropped. Latency is now monotonic in each
   run's own index size (1000-run 2.07 ms → 250-run 2.63 ms) — a real property, not an
   artifact.
3. **Reranker loaded once** at `Retriever` construction (fp16), never per query.
4. **Latency discipline** (`scripts/phase3_check.py`): a warmup pass is discarded, and the
   query-embed step is timed separately from the retrieval step. p50/p95 over the full
   question set is reported in Phase 4, not over ad hoc queries here.
5. **Self-retrieval check:** 20 random chunks, each used as its own query — vector-only
   returned the exact chunk at **rank 1, 20/20**. (If this failed, query embedding or
   normalization would be broken and every downstream figure noise.)

## Phase 4 — recorded decisions (before generation)

- **Correctness is strict single-gold-chunk** (`CORRECTNESS_RULE=exact_chunk`): a retrieval
  is correct iff the gold chunk itself is in the top-k — not a chunk that merely answers
  the question.
- That is only defensible if **questions are document-unique by construction**: every
  question must contain a detail appearing in exactly one document (section/rule number,
  specific date or threshold, named entity, or a term as that document defines it). A
  question answerable from several publishers describing the same mechanism is rejected and
  regenerated. The distinguishing detail is stored in `eval_questions.uniqueness_anchor`
  for review.
- **Stratify:** cap questions per document so the six SEC rules that hold **50.4%** of
  chunks (recomputed: 8,850 / 17,567; Reg SCI alone 13.9%) do not hold half the questions.

### The one component that needs per-corpus configuration

Everything in this build runs on any corpus unchanged **except descriptive question
generation**, which must name the issuing institution (to avoid deixis like "this
guidance"). That configuration lives in the **manifest, not in code**: each document in
`corpus/manifest.json` carries `publisher_display` (the name to put in a question, e.g.
"the CFTC", "CPMI-IOSCO", "NSCC") and `publisher_tokens` (what must appear for
validation). `scripts/gen_questions.py` reads these per document — there is no hardcoded
publisher list and no generic fallback. A document missing `publisher_display` is
**skipped loudly and reported by name**, never given a deictic question; hand-edit the
manifest to add it. (Cited generation and all retrieval/scoring need no per-corpus setup.)

## Status

- [x] **Phase 0** — infrastructure, schema, pinned deps
- [x] **Phase 1** — ingest & chunking (extraction constant, furniture stripped, 3 chunk sizes)
- [x] **Phase 2** — embedding & storage (bge + MiniLM, fp16, normalized & verified)
- [x] **Phase 3** — four retrieval modes (same signature; symmetry asserted; self-retrieval 20/20)
- [x] **Phase 4** — evaluation: split cited/descriptive, span-containment, McNemar significance
- [ ] Phase 5 — generation
- [ ] Phase 6 — Streamlit UI
- [ ] Domain-drift finding (written up here)

## Results

Scored on **bge-base, fp16**, over **51 cited + 51 descriptive** questions, split by
type and **never pooled** (`scripts/score.py` → `eval_runs`/`eval_results`). Retrieval is
corpus-wide; a hit is a top-k chunk **from the gold document that fully contains the gold
span**. recall@5 as rate (hits/51). Five modes: vector, lexical, hybrid (RRF), hybrid+
rerank, and **vector+rerank** (cross-encoder over vector's top 50, skipping lexical).

**recall@5 — cited**

| mode | 250 | 500 | 1000 |
|---|---|---|---|
| vector | .47 | .31 | .31 |
| lexical | .24 | .27 | .25 |
| hybrid | .41 | .41 | .39 |
| hybrid+rerank | .65 | **.69** | .59 |
| vector+rerank | **.73** | .65 | .61 |

**recall@5 — descriptive**

| mode | 250 | 500 | 1000 |
|---|---|---|---|
| vector | .24 | **.41** | .35 |
| lexical | .10 | .16 | .20 |
| hybrid | .16 | .27 | .25 |
| hybrid+rerank | .29 | .33 | .27 |
| vector+rerank | .33 | .35 | .27 |

`tok@5` ≈ 1,250 / 2,500 / 5,000 at 250/500/1000. Retrieval-step p50 (query-embed ~6 ms
excluded): vector ≈ 3 ms; lexical & hybrid 45–170 ms; hybrid+rerank 183–265 ms;
**vector+rerank 94–135 ms**.

**Significant differences (McNemar exact, paired hit@5, p<0.05):**
- **vector+rerank is the cited winner and dominates hybrid+rerank:** significant over
  plain vector at all sizes (p=.001/.000/.001); vs hybrid+rerank it is *equal in recall*
  (p=.125/.50/1.0) at **~half the latency** — it reranks vector's clean list instead of
  the lexical-damaged fused list. On cited 250 it reaches **.73 (37/51)**, the table's best.
- **hybrid+rerank > hybrid:** significant on cited at all sizes; on descriptive only at 250.
- **vector > lexical:** significant on descriptive 250/500 and cited 250. **hybrid never
  significantly beats vector** (closest cited 500/1000, not significant).
- **Chunk size follows query type (vector):** cited **250 > 500** (24 vs 16, p=.039) and
  descriptive **500 > 250** (21 vs 12, p=.035) — two significant inversions in opposite
  directions. Rerank modes are robust to chunk size (no size pair significant).

**Weighted-RRF sweep (`scripts/rrf_sweep.py`), descriptive:** equal weighting is why
fusion imports lexical's errors. Up-weighting vector recovers most of the loss but
**never reaches plain vector within an 8:1 weight** (500: .27 at 1:1 → .39 at 8:1, vs
vector .41; 250/1000 similar). Fusion only equals vector in the limit where lexical
contributes nothing — so on descriptive the answer is not to down-weight lexical but to
**drop it** (use vector or vector+rerank).

**Why lexical is weak even on cited (the load-bearing caveat):** the feared citation-leak
to lexical **did not materialise** — lexical is the *weakest* arm even on cited questions
containing the exact rule number. `ts_rank_cd` has **no IDF**, so the rare identifier is
one term among many and never dominates ranking. A real BM25 arm would likely flip the
cited column; the comparison is only honest with that named.

**Large-chunk bias, named:** bigger chunks return ~4× the text for no recall gain
(cited vector+rerank .73 @250 ≈ .65 @500 ≈ .61 @1000 at 1×/2×/4× the tokens). recall@250
is capped at the containment **ceiling 0.96** (two gold spans straddle a 250-token
boundary); 500/1000 ceilings are 1.00.

**Descriptive uniqueness floor:** `scripts/check_uniqueness.py` dropped **6%** with a
clear second home; ~80% have a topically-related (non-answering) passage elsewhere, valid
under single-gold scoring. In a corpus built for overlap, that residual is a known floor —
some natural-language questions legitimately have more than one home, counted as misses.

### Conclusions (each supported by the numbers above)

- **Chunk size should follow query type: 250 for cited, 500 for descriptive.** The two
  vector inversions are significant and opposite (cited 250>500 p=.039; descriptive
  500>250 p=.035); 1000 never wins and costs 4× the tokens.
- **The lexical arm costs 15–50× vector's latency and contributes negative recall on
  descriptive.** Lexical p50 45–170 ms vs vector ~3 ms; fusing it *lowers* descriptive
  recall (hybrid .27 < vector .41 @500) and no RRF weight recovers it.
- **`ts_rank_cd` without IDF cannot exploit a rare identifier**, which is why citation
  anchors did not favour lexical and why a true BM25 arm would likely flip the cited
  column. Best configs: **vector+rerank @250 on cited (.73)** — equal to hybrid+rerank at
  half the latency; **plain vector @500 on descriptive (.41)** — reranking and fusion
  both fail to beat it.
