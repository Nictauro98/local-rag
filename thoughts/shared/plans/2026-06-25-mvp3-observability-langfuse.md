# MVP 3 — Observability with Langfuse — Implementation Plan

> Milestone 3 of 4. Depends on MVP 2. See [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md). This MVP is a roadmap addition (not in the original spec), decided 2026-06-25.

## Overview

Add **self-hosted Langfuse** for full LLM observability. Every query becomes a trace whose nested spans (`retrieve → generate → evaluate → rewrite_query`) record inputs, outputs, latency, and token counts. The MVP 2 `EvalResult` scores are pushed onto each trace as Langfuse **scores**, turning the in-UI eval panel into historical dashboards (faithfulness/relevance trends, flag rate, retry rate over time). Langfuse runs entirely in Docker — no data leaves the perimeter, consistent with the project's data-sovereignty thesis.

**Langfuse does not replace the MVP 2 eval layer.** MVP 2 *computes* scores locally; Langfuse *stores and visualizes* them. The local NLI + deterministic tiers remain the differentiator.

## Current State Analysis (assumes MVP 2 complete)

- LangGraph graph (`retrieve→generate→evaluate→rewrite_query`) is compiled in `src/graphs/query_graph.py` and invoked from `POST /query`.
- `CompositeEvaluator` returns `EvalResult` with `tier_scores: dict[str,float]` — already shaped for per-score export.
- Query history persists to SQLite via `HistoryStore`.
- No tracing exists; latency/token data is not captured anywhere.

## Desired End State

- `docker compose up` additionally starts Langfuse (web + worker + its Postgres + ClickHouse + Redis as Langfuse requires) reachable at `http://localhost:3000`.
- Every `/query` call produces one Langfuse trace with child spans for each graph node, including model name, latency, and token usage on the generate/judge spans.
- Each trace carries scores: `faithfulness`, `context_relevance`, `answer_grounded`, `flagged`, plus per-tier scores from `tier_scores`.
- Langfuse dashboards show score trends, flag rate, and p50/p95 latency per node.
- Tracing is **toggleable** via env (`RAG_TRACING_ENABLED`); when off, the app behaves exactly as MVP 2 (no hard dependency on Langfuse being up).
- `uv run pytest` passes, including a test that asserts the tracer is a no-op when disabled.

### Key Decisions

- **Integration path = LangGraph callback handler.** Langfuse ships a callback/handler that auto-captures graph node spans; we pass it via `config={"callbacks": [handler]}` on graph invocation. Minimal code in the nodes.
- **Scores pushed explicitly** after the evaluate node, mapping `EvalResult` → `langfuse.score(...)` on the current trace. Booleans (`flagged`, `answer_grounded`) sent as 0/1 scores for chartability.
- **Tracing behind a thin interface** (`src/observability/tracer.py`) so "no Langfuse" is a null tracer — keeps `core/`/`nodes/` free of a hard SDK dependency and preserves the adapter discipline.
- **Self-hosted only.** Use Langfuse's official docker-compose service definitions; document that no telemetry/keys leave the host. Keys are local-generated.

## What We're NOT Doing (this MVP)

- Not replacing the MVP 2 evaluators with Langfuse's built-in LLM-judge (keeps local-only story).
- Not adding hybrid/rerank retrieval (MVP 4).
- Not using Langfuse Cloud — self-hosted only.
- Not building prompt-management workflows (possible future work; mention in README).

## Implementation Approach

Stand up Langfuse in compose first and verify the UI, then add the tracer interface + null implementation, then the Langfuse-backed tracer, then wire the LangGraph callback + score export, then dashboards + README.

---

## Phase 1: Langfuse infrastructure

### Changes Required

#### 1. `docker-compose.yml`
Add Langfuse's required services from its official compose (langfuse-web, langfuse-worker, its Postgres, ClickHouse, Redis, MinIO-for-langfuse or reuse). Pin versions. Health checks; the app's `api`/`worker` `depends_on` Langfuse only when `RAG_TRACING_ENABLED=true` (document that tracing-off skips needing it).

#### 2. Config
Add `RAG_TRACING_ENABLED` (default false), `RAG_LANGFUSE_HOST`, `RAG_LANGFUSE_PUBLIC_KEY`, `RAG_LANGFUSE_SECRET_KEY` to `Settings` + `.env.example`. Add `langfuse` to deps.

### Success Criteria
#### Automated Verification
- [ ] `docker compose config` validates with Langfuse services.
#### Manual Verification
- [ ] Langfuse UI loads at `:3000`; create an org/project; generate local API keys.

---

## Phase 2: Tracer abstraction

### Changes Required

#### 1. `src/observability/tracer.py`
`Tracer` protocol/ABC: `callbacks() -> list`, `score(trace_id, name, value, comment=None)`, `flush()`. `NullTracer` (no-ops, empty callbacks) and `LangfuseTracer` (wraps the Langfuse client + LangGraph callback handler). A factory `get_tracer(settings)` returns Null when disabled.

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest tests/unit/test_tracer.py` — `get_tracer` returns `NullTracer` when disabled; `NullTracer.callbacks()` is empty and `score()` is a no-op.

---

## Phase 3: Wire tracing into the query path

### Changes Required

#### 1. `src/graphs/query_graph.py` / `src/api/routes/query.py`
On `/query`, build the tracer, pass `tracer.callbacks()` into the graph `config`. Each node automatically becomes a span. Tag the trace with the question and `evaluate` flag.

#### 2. Score export
After the evaluate node (or post-run from the returned `EvalResult`), call `tracer.score(...)` for `faithfulness`, `context_relevance`, `answer_grounded` (0/1), `flagged` (0/1), and each `tier_scores` entry. Call `tracer.flush()` at request end.

### Success Criteria
#### Automated Verification
- [ ] `uv run pytest tests/unit/test_query_tracing.py` — with a fake tracer, asserts `score()` called once per expected metric and `callbacks()` passed into the graph invocation.

#### Manual Verification
- [ ] With tracing on, a `/query` produces a trace in Langfuse showing retrieve/generate/evaluate spans, token counts on generate, and the eval scores attached.

---

## Phase 4: Dashboards + docs

### Changes Required

#### 1. Langfuse dashboards
Configure (and screenshot for README): score trends over time, flag rate, retry rate, per-node latency p50/p95.

#### 2. README
New "Observability" section: how to enable tracing, how to read the dashboards, and the privacy note (self-hosted, no external egress). Clarify Langfuse complements — not replaces — the MVP 2 eval layer.

### Success Criteria
#### Manual Verification
- [ ] Dashboards render with real data after a handful of queries; screenshots added to README.
- [ ] Disabling `RAG_TRACING_ENABLED` and bringing the stack up without Langfuse still serves queries normally.

**Implementation Note**: Pause for manual confirmation of traces + dashboards before declaring MVP 3 done.

---

## Testing Strategy

### Unit
- Tracer factory (null vs langfuse), null no-ops, score mapping from `EvalResult`.

### Integration / Manual
- End-to-end trace appears with correct spans and scores; tracing-off path unaffected.

## Performance Considerations

- Langfuse SDK batches and flushes asynchronously; ensure `flush()` doesn't block the response on the hot path (flush in background / on shutdown). Tracing adds negligible latency when batched.
- Langfuse's ClickHouse/Postgres add memory footprint — document recommended resources and that tracing can be left off on constrained demo machines.

## Migration Notes

- `Tracer` interface means swapping Langfuse for another backend (or AWS-hosted) touches one adapter.

## References

- Roadmap: [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md)
- Prev: [`2026-06-25-mvp2-evaluation-layer.md`](2026-06-25-mvp2-evaluation-layer.md)
- Next: [`2026-06-25-mvp4-hybrid-retrieval-reranking.md`](2026-06-25-mvp4-hybrid-retrieval-reranking.md)
