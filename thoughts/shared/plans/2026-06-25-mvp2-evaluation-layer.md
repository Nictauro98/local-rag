# MVP 2 — Evaluation Layer — Implementation Plan

> Milestone 2 of 4. Depends on MVP 1. See [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md). Source spec: [`../../local_rag_implementation_plan.md`](../../local_rag_implementation_plan.md) (MVP 2 scope: lines 185–243).

## Overview

Add a multi-tier, fully-local evaluation layer to the query pipeline that detects hallucinations, scores faithfulness/relevance, and conditionally retries with a rewritten query when quality is below threshold — bounded to avoid infinite loops. Eval results surface in the Streamlit UI and persist to a query-history store. No external API is used at any tier.

## Current State Analysis (assumes MVP 1 complete)

After MVP 1:
- `src/graphs/query_graph.py` compiles `retrieve → generate → END`; `RAGState` already carries `eval_result` (always None) and an `evaluate: bool` flag threaded through but unused.
- `POST /query` accepts `evaluate` but returns `eval_result: null`.
- `src/evaluation/` exists as an empty package placeholder (from the spec tree).
- Adapters (`Embedder`, `VectorStore`) and the Ollama generate node are injectable — the deterministic and LLM-judge tiers reuse them.
- No persistence layer exists yet (job status lives in Redis only).

## Desired End State

- `POST /query` with `evaluate: true` returns a populated `eval_result` (faithfulness, context_relevance, answer_grounded, flagged, judge_reasoning).
- When `flagged` is True, the graph rewrites the query and retries `retrieve→generate→evaluate`, bounded by `max_retries` (default 2), then returns the best/last attempt.
- Each enabled tier (deterministic / NLI / LLM-judge) is independently toggleable via env var.
- `GET /query/history` returns past queries with their eval results (SQLite).
- Streamlit shows a color-coded eval panel (green/yellow/red) and a history tab.
- `uv run pytest` passes: per-evaluator unit tests, a "misleading context → flagged" integration test, and a "retry triggers and resolves within max_retries" test.

### Key Decisions

- **The judge model is distinct from the generation model** (env `RAG_JUDGE_MODEL`, e.g. `phi3`) to avoid self-evaluation bias ([spec line 210](../../local_rag_implementation_plan.md)).
- **Malformed judge output ⇒ flagged** (fail-safe), parsed via Pydantic.
- **History store = SQLite** via a small async wrapper (`aiosqlite`). Chosen over Postgres for MVP 2 to keep scope tight; the access is behind a `HistoryStore` interface so it can swap later. (Postgres becomes natural in MVP 3 alongside Langfuse — noted, not done here.)
- **Composite runs cheap→expensive**: deterministic (ms) → NLI (GPU, fast) → LLM-judge (slow). Short-circuit configurable.

## What We're NOT Doing (this MVP)

- No Langfuse / external tracing (MVP 3) — the eval panel is in-UI + SQLite only.
- No hybrid/rerank retrieval (MVP 4).
- No change to the generation model or chunking.
- No replacing the eval logic with a hosted evaluator — all tiers run locally.

## Implementation Approach

Build the result model + interface first, then the three evaluators independently (each unit-tested in isolation), then the composite, then wire the LangGraph extension (evaluate + rewrite_query + conditional edge + retry bound), then persistence, API, and UI.

---

## Phase 1: Result model + evaluator interface

### Changes Required

#### 1. `src/evaluation/interfaces/evaluator.py`
ABC `Evaluator`: `async evaluate(query, context: list[str], answer) -> EvalResult`.

#### 2. `EvalResult` (Pydantic)
Fields per [spec line 193](../../local_rag_implementation_plan.md): `faithfulness: float`, `context_relevance: float`, `answer_grounded: bool`, `flagged: bool`, `judge_reasoning: str | None`. Add `tier_scores: dict[str, float]` for per-tier breakdown (feeds MVP 3 Langfuse scores cleanly) and `retries: int`.

### Success Criteria
#### Automated Verification
- [x] `uv run python -c "from src.evaluation.interfaces.evaluator import Evaluator, EvalResult"` succeeds.
- [x] `EvalResult` rejects out-of-range scores (validator test).

---

## Phase 2: Tier 1 — Deterministic evaluator

### Changes Required

#### 1. `src/evaluation/deterministic.py`
`DeterministicEvaluator(Evaluator)`, deps: an `Embedder` (injected).
- **context_relevance**: cosine similarity between embedded query and the mean of retrieved chunk vectors.
- **answer_grounded / faithfulness proxy**: token-overlap ratio between answer and concatenated context.
- No LLM call. `flagged` if either falls below its env threshold.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_deterministic_eval.py` — high-overlap answer scores high; off-topic answer flagged. Embedder mocked with deterministic vectors.

---

## Phase 3: Tier 2 — NLI evaluator

### Changes Required

#### 1. `src/evaluation/nli_evaluator.py`
`NLIEvaluator(Evaluator)` using `cross-encoder/nli-deberta-v3-small` via `sentence-transformers` (add deps `sentence-transformers`, `torch`). Loads the model once (module-level lazy singleton). For each (premise=context, hypothesis=answer-sentence) pair, compute entailment probability; aggregate to a faithfulness score in 0–1. Runs on GPU if available, else CPU.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_nli_eval.py` — entailed answer scores high, contradicted answer scores low (uses a tiny fixed text pair; may be marked slow if it downloads the model — cache in CI).

#### Manual Verification
- [ ] Model loads on GPU when available; CPU fallback works.

---

## Phase 4: Tier 3 — LLM-as-judge evaluator

### Changes Required

#### 1. `src/evaluation/llm_judge.py`
`LLMJudgeEvaluator(Evaluator)` calling Ollama with `RAG_JUDGE_MODEL` (distinct from generation model). Structured prompt asks for faithfulness + groundedness on 1–5 + reasoning, requesting JSON. Parse into a `JudgeOutput` Pydantic model; **malformed ⇒ flagged + reasoning="unparseable judge response"**. Normalize 1–5 to 0–1.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_llm_judge.py` — mocked Ollama returning valid JSON parses correctly; malformed JSON ⇒ flagged.

---

## Phase 5: Composite evaluator

### Changes Required

#### 1. `src/evaluation/composite_evaluator.py`
`CompositeEvaluator(Evaluator)` taking a list of enabled tiers (built from env flags `RAG_EVAL_DETERMINISTIC/NLI/JUDGE`). Runs them in cheap→expensive order, populates `tier_scores`, and sets `flagged = any(tier flags)`. Each tier independently enable/disable-able.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_composite_eval.py` — disabling a tier omits it from `tier_scores`; any tier flagging ⇒ overall flagged.

---

## Phase 6: LangGraph extension (evaluate + bounded retry)

### Changes Required

#### 1. `src/nodes/evaluate.py`
`evaluate(state, *, evaluator)`: runs the composite on `query`/`chunks`/`answer`, stores `EvalResult` in state.

#### 2. `src/nodes/rewrite_query.py`
`rewrite_query(state, *, llm_url, llm_model)`: uses the LLM to rephrase the original query given the failure reason; increments `state["retries"]`; sets `state["query"]` to the rewrite.

#### 3. `src/graphs/query_graph.py` (extend)
- Add nodes `evaluate`, `rewrite_query`.
- Edge `generate → evaluate` **only when `state["evaluate"]` is True**; otherwise `generate → END` (preserves MVP 1 behavior and the spec's "eval must not block MVP 1" rule, [spec line 281](../../local_rag_implementation_plan.md)).
- Conditional edge from `evaluate`: `flagged is False` → END; `flagged and retries < max_retries` → `rewrite_query`; else → END (return last attempt).
- `rewrite_query → retrieve` (loop). `RAGState` gains `retries: int` and `max_retries: int`.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_graph_eval.py` — with `evaluate=False`, graph ends at generate, `eval_result is None`. With `evaluate=True` + a stub evaluator that flags once then passes, the loop runs exactly one retry. With an always-flagging evaluator, it stops at `max_retries`.

---

## Phase 7: Persistence (query history)

### Changes Required

#### 1. `src/interfaces/history.py` + `src/adapters/sqlite_history.py`
`HistoryStore` ABC: `async save(record)`, `async list(limit) -> list[record]`. `SQLiteHistory` via `aiosqlite`; schema: id, timestamp, question, answer, sources(json), eval_result(json), retries. DB path env-configurable; file mounted as a volume in compose.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_history.py` — save then list round-trips an `EvalResult`.

---

## Phase 8: API update

### Changes Required

#### 1. `src/api/routes/query.py`
- `POST /query`: when `evaluate=true`, run the eval-enabled graph; populate `eval_result`; persist to `HistoryStore`.
- `GET /query/history?limit=`: return saved records.

#### 2. `src/api/schemas.py`
Add `EvalResult` to `QueryResponse`; add `HistoryItem` / `HistoryResponse`.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest tests/unit/test_api_eval.py` — `evaluate=true` returns populated `eval_result`; `/query/history` returns the saved item.
- [x] `curl -s localhost:8000/openapi.json` includes `/query/history`.

---

## Phase 9: Streamlit UI update

### Changes Required

#### 1. `ui/app.py`
- Eval toggle in the UI; when on, send `evaluate: true`.
- Eval panel under each answer: faithfulness, context_relevance, grounding, flagged, judge_reasoning. Color-coded green/yellow/red by thresholds.
- "History" tab: `GET /query/history` table of past Q/A + scores.

### Success Criteria
#### Manual Verification
- [ ] Toggling eval on shows the panel; a clearly grounded answer is green, a hallucinated/off-topic answer is red.
- [ ] History tab lists prior queries with scores.

---

## Phase 10: Integration tests

### Changes Required

#### 1. `tests/integration/test_eval_flagging.py`
Ingest a doc, ask a question whose answer is *not* supported (or inject misleading context) → assert `flagged is True`.

#### 2. `tests/integration/test_retry_loop.py`
Force a low-quality first retrieval (e.g., tiny top_k / unrelated doc) → assert the retry loop triggers and terminates within `max_retries`, with `retries <= max_retries` in the result.

### Success Criteria
#### Automated Verification
- [x] `uv run pytest -m integration tests/integration/test_eval_flagging.py tests/integration/test_retry_loop.py` green.

**Implementation Note**: Pause for manual confirmation of the eval panel + flagging demo before declaring MVP 2 done.

---

## Testing Strategy

### Unit
- Each evaluator in isolation with mocked inputs (deterministic vectors, mocked Ollama).
- Composite tier toggling + flag aggregation.
- Graph routing: no-eval path, single-retry path, max-retry stop.
- History round-trip.

### Integration
- Misleading-context ⇒ flagged.
- Retry loop triggers and is bounded.

### Manual
- UI eval panel color-coding; history tab.

## Performance Considerations

- NLI model loaded once (singleton); shares the GPU with Ollama — document VRAM headroom, allow disabling the NLI tier on constrained machines.
- LLM-judge is the slow tier; default the judge model to a small one (`phi3`) and allow disabling.
- Retry loop bounded by `max_retries` to cap worst-case latency at ~`(1+max_retries)` full passes.

## Migration Notes

- `HistoryStore` interface lets SQLite swap to Postgres later (MVP 3) without touching API logic.

## References

- Spec: [`../../local_rag_implementation_plan.md`](../../local_rag_implementation_plan.md) (lines 185–243)
- Prev: [`2026-06-25-mvp1-working-local-rag.md`](2026-06-25-mvp1-working-local-rag.md)
- Next: [`2026-06-25-mvp3-observability-langfuse.md`](2026-06-25-mvp3-observability-langfuse.md)
