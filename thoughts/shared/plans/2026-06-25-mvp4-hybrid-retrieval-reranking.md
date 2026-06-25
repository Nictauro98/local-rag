# MVP 4 — Hybrid Retrieval + Reranking — Implementation Plan

> Milestone 4 of 4. Depends on MVP 1 (best demoed after MVP 3, so retrieval-quality gains show up in Langfuse). See [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md). Roadmap addition decided 2026-06-25.

## Overview

Upgrade retrieval from plain dense top-k to **hybrid search** (sparse BM25/keyword + dense vector, fused) followed by a **cross-encoder reranker** that reorders the fused candidates before generation. This is the single biggest answer-quality lever in a RAG system and is what most RAG portfolios are judged on. All of it runs locally. The improvement is measurable against the MVP 2 eval scores and visible in MVP 3's Langfuse dashboards.

## Current State Analysis (assumes MVP 1+; ideally MVP 3)

- `src/nodes/retrieve.py` does: embed query → `VectorStore.search(top_k)` (dense only) → chunks into state.
- `QdrantStore` stores chunk payloads (`text`, `source_filename`, `chunk_index`) with cosine dense vectors.
- Ingestion (`core/ingestion.py`) produces dense vectors only; no sparse representation is stored.
- Eval (MVP 2) and tracing (MVP 3) give a baseline to measure retrieval changes against.

## Desired End State

- `retrieve` returns candidates from **both** a dense search and a sparse/keyword search, fused (Reciprocal Rank Fusion), then **reranked** by a cross-encoder; the top-N reranked chunks go to `generate`.
- Hybrid + rerank are individually toggleable via env (`RAG_HYBRID_ENABLED`, `RAG_RERANK_ENABLED`) so the pipeline degrades cleanly to MVP 1 behavior and A/B comparison is possible.
- Measurable quality lift: on a small labeled eval set, mean faithfulness/context_relevance (MVP 2 scores) improves vs the dense-only baseline; the difference is visible in Langfuse (MVP 3).
- `uv run pytest` passes: fusion + reranker unit tests, and a retrieval-quality regression test on a fixture set.

### Key Decisions

- **Sparse search via Qdrant native sparse vectors** (Qdrant supports hybrid/sparse + a Query API with built-in fusion) rather than bolting on a separate BM25 service. Keeps one store, one swap point, and matches the "Qdrant is production-ready" thesis. Sparse vectors built with a BM25/`fastembed` sparse encoder at ingestion time.
- **Fusion = Reciprocal Rank Fusion (RRF)** — robust, parameter-light, supported by Qdrant's Query API.
- **Reranker = cross-encoder** (e.g. `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`) via `sentence-transformers`, loaded once (singleton, GPU if available). Model name env-configurable; never hardcoded.
- **Two-stage widths**: retrieve a wide candidate set (e.g. fused top-30), rerank down to `top_k` (e.g. 5). Both widths env-configurable.
- **Reranker reuses the same model-loading discipline** as the MVP 2 NLI evaluator (shared `sentence-transformers`, lazy singleton).

## What We're NOT Doing (this MVP)

- No external reranking API (Cohere/Voyage) — local cross-encoder only, preserving the privacy thesis.
- No re-architecting chunking (size/overlap unchanged); chunking experiments are future work.
- No multi-vector / ColBERT-style late interaction (possible future work; note in README).
- No change to eval or tracing logic — they're the measurement harness, used as-is.

## Implementation Approach

Add sparse vectors at ingestion (and a migration/backfill for existing docs) → extend the vector store interface + Qdrant adapter for hybrid query with RRF → add the reranker component → rewire the `retrieve` node into retrieve-wide → rerank-narrow, behind toggles → measure and document.

---

## Phase 1: Sparse representation at ingestion

### Changes Required

#### 1. Sparse encoder
Add a sparse encoder (BM25-style via `fastembed`'s sparse models) wrapped behind a small `SparseEncoder` helper in `src/core/` (pure, injectable). Add deps (`fastembed`).

#### 2. `src/core/ingestion.py` + `src/interfaces/vector_store.py`
`ingest` also computes a sparse vector per chunk. `ChunkPoint` gains an optional `sparse` field. `VectorStore.upsert` stores both dense and sparse vectors. `ensure_collection` declares both a dense and a named sparse vector config.

#### 3. Backfill
A one-shot `scripts/backfill_sparse.py` (or an arq task) that re-ingests/updates existing documents to add sparse vectors. Idempotent (reuses `delete_by_source`).

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest tests/unit/test_sparse_ingestion.py` — `ingest` produces both vectors; mocked store receives sparse payloads.
#### Manual Verification
- [ ] Backfill updates an existing collection without data loss; point count unchanged.

---

## Phase 2: Hybrid query + RRF in the store

### Changes Required

#### 1. `src/interfaces/vector_store.py`
Add `async hybrid_search(dense_vec, sparse_vec, candidate_k) -> list[RetrievedChunk]`.

#### 2. `src/adapters/qdrant_store.py`
Implement `hybrid_search` using Qdrant's Query API with prefetch (dense + sparse) and `FusionQuery(RRF)`, returning `candidate_k` fused results with scores.

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest tests/unit/test_hybrid_search.py` — mocked Qdrant client invoked with both prefetches + RRF fusion; results mapped to `RetrievedChunk`.
#### Manual Verification
- [ ] Against live Qdrant: hybrid query returns sensible candidates for a keyword-heavy query that dense-only misses.

---

## Phase 3: Cross-encoder reranker

### Changes Required

#### 1. `src/core/reranker.py`
`Reranker` (injectable): `rerank(query, chunks, top_k) -> list[RetrievedChunk]`. Cross-encoder via `sentence-transformers`, lazy singleton, GPU-if-available. Model name from `RAG_RERANK_MODEL`. Scores each (query, chunk.text) pair, sorts desc, returns `top_k` with rerank score attached.

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest tests/unit/test_reranker.py` — given candidates with a known-relevant one buried, reranker surfaces it to the top (small fixed model or mocked scorer).

---

## Phase 4: Rewire the retrieve node (behind toggles)

### Changes Required

#### 1. `src/nodes/retrieve.py`
New flow, controlled by env toggles:
- If `RAG_HYBRID_ENABLED`: compute dense + sparse query vectors → `store.hybrid_search(candidate_k)`; else dense `search(candidate_k)`.
- If `RAG_RERANK_ENABLED`: `reranker.rerank(query, candidates, top_k)`; else take `candidates[:top_k]`.
- Deps (`embedder`, `sparse_encoder`, `store`, `reranker`, widths) injected via partial when wiring the graph. Defaults reproduce MVP 1 behavior when both toggles are off.

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest tests/unit/test_retrieve_hybrid.py` — toggle matrix (off/off = dense top-k; on/on = hybrid→rerank); correct widths used at each stage.

---

## Phase 5: Measurement + docs

### Changes Required

#### 1. `scripts/eval_retrieval.py`
Runs a fixture question set through the pipeline twice (dense-only vs hybrid+rerank), reports mean MVP 2 eval scores for each. Optionally tags runs in Langfuse (MVP 3) for side-by-side dashboards.

#### 2. README
"Retrieval" section: hybrid + RRF + reranking explained, the toggles, and the before/after numbers + Langfuse screenshots. Note local-only (no external rerank API).

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest -m integration tests/integration/test_retrieval_quality.py` — hybrid+rerank mean score ≥ dense-only baseline on the fixture set (assert non-regression).
#### Manual Verification
- [ ] On a keyword/acronym-heavy question that dense-only got wrong, hybrid+rerank returns the correct grounded answer; improvement visible in Langfuse.

**Implementation Note**: Pause for manual confirmation of the before/after quality comparison before declaring MVP 4 done.

---

## Testing Strategy

### Unit
- Sparse ingestion produces both vectors.
- Hybrid query issues dense+sparse prefetch with RRF.
- Reranker surfaces the buried-relevant candidate.
- Retrieve-node toggle matrix.

### Integration
- Retrieval-quality non-regression (hybrid+rerank ≥ dense baseline) on a fixture set.

### Manual
- Keyword-heavy query corrected by hybrid+rerank; Langfuse before/after.

## Performance Considerations

- Reranking adds a cross-encoder forward pass over `candidate_k` chunks — bound `candidate_k` (default ~20–30) to cap latency; both widths env-tunable.
- Reranker + NLI evaluator + Ollama share the GPU — document VRAM budget and allow disabling rerank on constrained machines (graceful fallback to dense top-k).
- Sparse vectors add storage but are cheap; Qdrant fuses server-side.

## Migration Notes

- Sparse + hybrid live behind the `VectorStore` interface; an AWS swap (e.g. OpenSearch k-NN + BM25) re-implements `hybrid_search` in one adapter. Reranker is provider-agnostic (could swap to Bedrock/SageMaker endpoint via the same `Reranker` seam).

## References

- Roadmap: [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md)
- Prev: [`2026-06-25-mvp3-observability-langfuse.md`](2026-06-25-mvp3-observability-langfuse.md)
