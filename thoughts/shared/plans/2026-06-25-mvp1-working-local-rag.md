# MVP 1 — Working Local RAG System — Implementation Plan

> Milestone 1 of 4. See [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md) for the full program and locked decisions. Source spec: [`../../local_rag_implementation_plan.md`](../../local_rag_implementation_plan.md).

## Overview

A fully functional, end-to-end, **fully local** RAG system. A user uploads PDF/DOCX/TXT documents through a Streamlit UI (or directly to the MinIO console); MinIO fires an `ObjectCreated` event notification to a webhook endpoint on the FastAPI app, which enqueues an `arq`+Redis ingestion job (parse → chunk → embed → upsert to Qdrant). The user asks a natural-language question and gets an answer grounded in the documents via a LangGraph `retrieve → generate` graph running on Ollama. The evaluation hook exists but is inert (`evaluate=False` → `eval_result=None`). All storage/embedding/LLM access is behind adapter interfaces, with AWS stubs present from day one.

The MinIO → webhook → arq path **directly mirrors** the AWS production path: S3 `ObjectCreated` → Lambda (`lambda_handler.py`) → ingestion. Both entry points receive the same S3-compatible event JSON; only the delivery mechanism differs.

## Current State Analysis

**Greenfield.** The repository contains only the spec at [`thoughts/local_rag_implementation_plan.md`](../../local_rag_implementation_plan.md) and the `.claude/` config. No `src/`, no `pyproject.toml`, no Docker assets exist. Verified 2026-06-25: `ls` shows only `.claude/` and `thoughts/`. The project is **not** a git repo yet (`git init` required as Phase 0).

Constraints carried from the spec:
- Only `langchain-text-splitters` allowed from LangChain ([spec line 279](../../local_rag_implementation_plan.md)).
- GPU access for Ollama needs NVIDIA container toolkit; CPU fallback must be documented ([spec line 283](../../local_rag_implementation_plan.md)).
- All config via env vars; no hardcoded model names ([spec line 282](../../local_rag_implementation_plan.md)).

Delta from spec (locked in roadmap): ingestion uses **arq + Redis**, not FastAPI `BackgroundTasks`.

## Desired End State

- `docker compose up` brings up MinIO, Qdrant, Ollama, Redis, the FastAPI API, the arq worker, and Streamlit — all healthy.
- Streamlit at `http://localhost:8501`: upload a PDF, see ingestion status reach "completed", ask a question, get an answer + expandable source chunks.
- FastAPI at `http://localhost:8000/docs`: `POST /documents/upload`, `GET /documents`, `GET /documents/{filename}/status`, `POST /query` all work.
- `POST /query` with `{"question": "...", "evaluate": false}` returns `{"answer": str, "sources": [...], "eval_result": null}`.
- Re-uploading the same filename replaces its vectors (idempotent ingestion).
- `uv run pytest` passes: unit tests (mocked adapters, each graph node) + one integration test (live Docker upload→query cycle).
- `uv run ruff check . && uv run black --check .` clean. CI green.

### Key Decisions

- **arq over Celery**: asyncio-native, minimal boilerplate, fits an async FastAPI app. Redis doubles as arq's broker and result backend.
- **Ingestion trigger = MinIO event notification → webhook**, not a direct enqueue from the upload endpoint. The upload endpoint stores to MinIO and returns immediately; MinIO calls `POST /internal/events/minio` on `ObjectCreated`, which enqueues the arq job. This is the exact local mirror of S3 → Lambda on AWS.
- **Manual MinIO uploads** (via the MinIO console) trigger the same webhook and the same ingestion path — no special case.
- **Ingestion idempotency** via a deterministic Qdrant filter on `source_filename`: delete-by-filter then upsert.
- **Job status** keyed by filename in Redis (arq job result via a deterministic `_job_id = filename`). `GET /documents/{filename}/status` looks up by this key.
- **Embedding dimension** is read once from the embed model at startup and used to create the Qdrant collection; never hardcoded.

## What We're NOT Doing (this MVP)

- No evaluation logic (MVP 2) — only the inert `evaluate` flag and `eval_result=None`.
- No Langfuse / tracing (MVP 3).
- No hybrid search or reranking (MVP 4) — dense top-k only.
- No auth, no multi-user, no React UI.
- No actual AWS calls — `s3_storage.py` / `bedrock_embedder.py` are documented stubs only.
- No query history persistence (added in MVP 2).

## Implementation Approach

Bottom-up and independently verifiable per phase: scaffold + config → interfaces → adapters → core ingestion → arq worker → LangGraph query graph → FastAPI → Streamlit → infra wiring → tests/CI. Each phase has a runnable verification before the next.

---

## Phase 0: Project scaffold, tooling, config

### Changes Required

#### 1. Repo + packaging
- `git init`.
- `pyproject.toml` with `uv`. Dependencies: `fastapi`, `uvicorn[standard]`, `streamlit`, `httpx`, `minio`, `qdrant-client`, `arq`, `redis`, `pydantic`, `pydantic-settings`, `pypdf`, `python-docx`, `unstructured`, `langchain-text-splitters`. Dev: `pytest`, `pytest-asyncio`, `ruff`, `black`, `pre-commit`.
- `ruff` + `black` config in `pyproject.toml`. `.pre-commit-config.yaml` running both.

#### 2. Directory skeleton
Create the tree from [spec lines 51–113](../../local_rag_implementation_plan.md) under `src/`, plus `ui/`, `tests/unit/`, `tests/integration/`. Add `__init__.py` files.

#### 3. Configuration — `src/config.py`
**File**: `src/config.py`
`pydantic-settings` `Settings` class. Every value env-driven with defaults:
```python
class Settings(BaseSettings):
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "chunks"
    ollama_url: str = "http://ollama:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    redis_url: str = "redis://redis:6379"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env")
```
`.env.example` documents every variable.

### Success Criteria

#### Automated Verification
- [x] `uv sync` resolves and installs.
- [x] `uv run python -c "from src.config import Settings; Settings()"` succeeds.
- [x] `uv run ruff check .` and `uv run black --check .` clean.

#### Manual Verification
- [ ] Repo structure matches the spec tree.

---

## Phase 1: Interfaces (migration boundary)

### Changes Required

#### 1. `src/interfaces/storage.py`
ABC `StorageBackend`: `async upload(filename, data) -> str`, `async download(filename) -> bytes`, `async list() -> list[str]`, `async exists(filename) -> bool`.

#### 2. `src/interfaces/embedder.py`
ABC `Embedder`: `async embed(texts: list[str]) -> list[list[float]]`, `async embed_one(text: str) -> list[float]`, `dimension() -> int`.

#### 3. `src/interfaces/vector_store.py`
ABC `VectorStore`: `async ensure_collection(dim: int)`, `async upsert(points: list[ChunkPoint])`, `async delete_by_source(filename: str)`, `async search(vector, top_k) -> list[RetrievedChunk]`. Define `ChunkPoint` and `RetrievedChunk` Pydantic models (text, source_filename, chunk_index, score).

### Success Criteria

#### Automated Verification
- [x] `uv run python -c "import src.interfaces.storage, src.interfaces.embedder, src.interfaces.vector_store"` succeeds.
- [x] ABCs cannot be instantiated (test asserts `TypeError`).

---

## Phase 2: Adapters

### Changes Required

#### 1. `src/adapters/minio_storage.py`
`MinIOStorage(StorageBackend)` using the `minio` SDK. Creates the bucket on init if missing. SDK calls are sync → wrap in `asyncio.to_thread`.

#### 2. `src/adapters/ollama_embedder.py`
`OllamaEmbedder(Embedder)` via `httpx.AsyncClient` POST to `/api/embed`. `dimension()` lazily probes the model with a one-token embed and caches the length.

#### 3. `src/adapters/qdrant_store.py`
`QdrantStore(VectorStore)` via `qdrant-client` (async client). `ensure_collection` creates the collection with the given dim + cosine distance if absent. `delete_by_source` uses a payload filter on `source_filename`. Payload stores `text`, `source_filename`, `chunk_index`.

#### 4. AWS stubs
`src/adapters/s3_storage.py` (`S3Storage(StorageBackend)`) and `src/adapters/bedrock_embedder.py` (`BedrockEmbedder(Embedder)`): every method raises `NotImplementedError` with a docstring describing the exact migration step (boto3 client, env vars, API shape).

#### 5. Adapter factory — `src/adapters/__init__.py`
`get_storage()`, `get_embedder()`, `get_vector_store()` return the concrete adapter selected by an env var (default local). This is the single swap point for AWS.

### Success Criteria

#### Automated Verification
- [x] Unit tests with mocked SDK clients cover each adapter's happy path.
- [x] `uv run pytest tests/unit/test_adapters.py` passes.
- [x] AWS stub methods raise `NotImplementedError` (asserted in tests).

#### Manual Verification
- [ ] Against live `docker compose up minio qdrant ollama`: a scratch script uploads a file to MinIO, embeds a string via Ollama, and upserts+searches one vector in Qdrant.

---

## Phase 3: Core ingestion (pure logic, injected deps)

### Changes Required

#### 1. `src/core/parsing.py`
`parse(filename: str, data: bytes) -> str` dispatching by extension: PDF→`pypdf`, DOCX→`python-docx`, TXT→decode. `unstructured` as fallback for other types. No adapter imports.

#### 2. `src/core/chunking.py`
`chunk(text: str, size: int, overlap: int) -> list[str]` using `RecursiveCharacterTextSplitter` from `langchain-text-splitters`.

#### 3. `src/core/ingestion.py`
`async ingest(filename, data, *, parser, embedder, store, settings)` — orchestrates parse → chunk → embed → `delete_by_source` → `upsert`. All collaborators injected. Returns chunk count. This is the function the worker and tests both call.

### Success Criteria

#### Automated Verification
- [x] `uv run pytest tests/unit/test_ingestion.py` passes with fully mocked parser/embedder/store, asserting delete-before-upsert ordering (idempotency) and chunk count.
- [x] Parsing unit tests cover PDF, DOCX, TXT fixtures.

---

## Phase 4: arq worker + ingestion trigger

### Changes Required

#### 1. `src/handlers/ingest_worker.py`
arq `WorkerSettings` with a task `ingest_document(ctx, filename)`: downloads bytes from storage, calls `core.ingestion.ingest` with adapters from the factory. Redis is broker + result backend (`redis_url`). Job enqueued with `_job_id=filename` (deterministic key). Returns `{"filename", "chunks", "status": "completed"}`; arq stores it as the job result. Concurrency configurable.

#### 2. `src/handlers/lambda_handler.py`
The **AWS production entry point**. Receives an S3 `ObjectCreated` event JSON, extracts the object key, enqueues (or directly calls) `core.ingestion.ingest`. Structurally identical to `POST /internal/events/minio` (Phase 6) — both parse the same S3-compatible event payload; only the delivery mechanism differs. Pure adapter; carries a docstring that no logic belongs here. No `watchdog` equivalent exists — local file watching is intentionally absent; drop files directly into the MinIO console to test the same trigger path locally.

### Success Criteria

#### Automated Verification
- [x] `uv run pytest tests/unit/test_worker.py` passes (task calls `ingest` with downloaded bytes; mocked arq ctx).

#### Manual Verification
- [ ] With the worker container running, manually enqueuing a job ingests a file and the job result shows `status: completed` with a chunk count.

---

## Phase 5: LangGraph query graph

### Changes Required

#### 1. `src/graphs/query_graph.py`
`RAGState` TypedDict: `query`, `chunks`, `answer`, `eval_result` (None in MVP 1), plus `evaluate: bool`. Build a `StateGraph`: `retrieve → generate → END`. Compile once at import; expose `run_query(question, evaluate=False) -> RAGState`. The `evaluate` flag is threaded into state but unused in MVP 1 (MVP 2 reads it for conditional routing).

#### 2. `src/nodes/retrieve.py`
`retrieve(state, *, embedder, store, top_k)`: embed query, `store.search`, put `RetrievedChunk`s in state. Deps injected via `functools.partial` when wiring the graph.

#### 3. `src/nodes/generate.py`
`generate(state, *, llm_url, llm_model)`: build a grounded prompt from retrieved chunks (with explicit "answer only from context" instruction), call Ollama `/api/generate` via httpx, store answer.

### Success Criteria

#### Automated Verification
- [x] `uv run pytest tests/unit/test_nodes.py` passes — each node tested in isolation with mocked embedder/store/LLM.
- [x] Graph-level test: mocked nodes produce a populated `RAGState` with `eval_result is None`.

---

## Phase 6: FastAPI backend

### Changes Required

#### 1. `src/api/schemas.py`
`UploadResponse{filename, message}`, `IngestionStatus{filename, status, chunks?}`, `DocumentList{documents}`, `QueryRequest{question, evaluate=False}`, `QueryResponse{answer, sources, eval_result=None}`.

#### 2. `src/api/routes/documents.py`
- `POST /documents/upload`: store file bytes via storage adapter; return `{"filename": ..., "message": "uploaded; ingestion will begin via storage event"}`. Does **not** enqueue directly — ingestion is triggered by the MinIO notification.
- `GET /documents`: list filenames present in storage.
- `GET /documents/{filename}/status`: read the arq job result from Redis by the deterministic key `filename`. Returns `{filename, status: pending|completed|failed, chunks?}`.

#### 3. `src/api/routes/events.py` *(new)*
- `POST /internal/events/minio`: receives MinIO's S3-compatible `ObjectCreated` webhook payload (JSON). Extracts the object key, enqueues `ingest_document` on arq with `_job_id=filename`. Returns `202 Accepted`. This is the **local equivalent of `lambda_handler.py`** — same event JSON, same enqueue logic, different invocation context.

#### 3. `src/api/routes/query.py`
- `POST /query`: call `run_query(question, evaluate)`. In MVP 1, `eval_result` is always null; if `evaluate=true`, return `501`-style note that eval ships in MVP 2 (or just null). Sources = distinct `source_filename`s from chunks.

#### 4. `src/api/main.py`
App factory; lifespan creates the arq redis pool and ensures the Qdrant collection (dim from `embedder.dimension()`). Mount routers. All endpoints async. OpenAPI at `/docs`.

### Success Criteria

#### Automated Verification
- [x] `uv run pytest tests/unit/test_api.py` passes (TestClient, mocked queue + webhook handler).
- [ ] `curl -s localhost:8000/openapi.json` lists all expected endpoints including `/internal/events/minio`.

#### Manual Verification
- [ ] `/docs` renders; uploading via Swagger stores the file in MinIO and returns `202` from the webhook; status for that filename reaches `completed`.

---

## Phase 7: Streamlit UI

### Changes Required

#### 1. `ui/app.py`
- Sidebar: file uploader → `POST /documents/upload`; poll `GET /documents/{filename}/status` until `completed` or `failed`, show a status line. The ingestion is triggered asynchronously by MinIO — polling makes the async flow visible in the UI.
- Sidebar: ingested documents list (`GET /documents`).
- Main: chat input → `POST /query`; render answer + an expander per source chunk (filename, chunk text). API base URL from env.

### Success Criteria

#### Manual Verification
- [ ] At `http://localhost:8501`: upload a PDF, watch status reach completed, ask a question grounded in it, get a correct answer with source chunks shown.
- [ ] Asking about something not in any document yields an "I don't have that in the provided context"-style answer (prompt grounding works).

---

## Phase 8: Infrastructure (Docker Compose)

### Changes Required

#### 1. `docker-compose.yml`
Services: `minio` (+console, default bucket via init), `qdrant`, `ollama` (GPU via `deploy.resources.reservations.devices` NVIDIA), `redis`, `api`, `worker` (arq), `ui`. Health checks on minio/qdrant/ollama/redis; `api`/`worker`/`ui` `depends_on` healthy. An `ollama-pull` init step (or documented manual step) pulls `nomic-embed-text` + `llama3.2`.

#### 2. MinIO bucket event notification
After the bucket is created (init container or entrypoint script), register an `ObjectCreated` event notification on the documents bucket using the MinIO `mc` client:
```bash
mc alias set local http://minio:9000 minioadmin minioadmin
mc event add local/documents arn:minio:sqs::primary:webhook --event put
```
The webhook ARN target must be pre-configured in MinIO's environment (`MINIO_NOTIFY_WEBHOOK_ENABLE_PRIMARY=on`, `MINIO_NOTIFY_WEBHOOK_ENDPOINT_PRIMARY=http://api:8000/internal/events/minio`). These go in `.env.example` and the compose `environment` block for the `minio` service. On startup the `api` service must be up before MinIO can deliver events — `depends_on` ordering handles this.

#### 3. `Dockerfile`
Multi-stage, `uv`-based, Lambda-container-compatible base (per spec migration note). One image used by `api`, `worker`, and `ui` with different commands.

#### 4. README
Setup (`docker compose up`), architecture description, **CPU-fallback** instructions for Ollama (remove GPU reservation), and the AWS migration guide table. Include a note explaining the MinIO → webhook → arq trigger and its S3 → Lambda equivalent so the architecture diagram reads clearly.

### Success Criteria

#### Automated Verification
- [ ] `docker compose config` validates.
- [ ] `docker compose up -d` → all health checks pass within timeout.
- [ ] `mc event list local/documents` shows the `ObjectCreated` webhook registered.

#### Manual Verification
- [ ] Upload a file via the Streamlit UI → MinIO stores it → webhook fires → arq worker ingests → status reaches `completed` without any manual enqueue step.
- [ ] Upload a file **directly via the MinIO console** → same webhook fires → same ingestion path completes.
- [ ] GPU-less machine: documented CPU fallback brings the stack up (slower).

---

## Phase 9: Tests + CI

### Changes Required

#### 1. Integration test — `tests/integration/test_upload_query.py`
Against live Docker services: PUT a small known TXT directly to MinIO (bypassing the API upload endpoint to isolate the trigger path), wait for the webhook to fire and arq to complete (`GET /documents/{filename}/status` polling), query it, assert the answer contains the expected fact and sources include the filename. `pytest.mark.integration`, skipped unless services are up.

#### 2. CI — `.github/workflows/ci.yml`
On push/PR: `uv sync`, `ruff check`, `black --check`, `pytest tests/unit`. Integration tests run in a job that spins up the compose services (or are marked manual if CI lacks a GPU — document CPU model for CI).

### Success Criteria

#### Automated Verification
- [x] `uv run pytest tests/unit` green.
- [x] `uv run pytest -m integration` green against live services.
- [ ] CI workflow green on push.

#### Manual Verification
- [ ] CI badge green in README.

**Implementation Note**: After Phase 9 and all automated checks pass, pause for manual confirmation of the end-to-end demo before declaring MVP 1 done.

---

## Testing Strategy

### Unit
- Adapters with mocked SDK clients (incl. AWS stubs raising `NotImplementedError`).
- `ingest` with mocked collaborators; assert delete-before-upsert idempotency.
- Parsing per format; chunking boundaries/overlap.
- Each graph node in isolation; full-graph with mocked nodes (`eval_result is None`).
- `/internal/events/minio` webhook: assert it parses the S3 event JSON and enqueues with `_job_id=filename`; assert malformed payloads return `400`.
- API routes with TestClient + mocked queue/graph.

### Integration
- PUT file to MinIO → webhook fires → poll status → query; assert grounded answer + sources.

### Manual
- Streamlit golden path (upload via UI) + MinIO console upload (direct drop); both paths reach `completed`; ungrounded-question edge case; GPU and CPU-fallback bring-up.

## Performance Considerations

- Embedding batches per document (single Ollama call per chunk batch) to limit round-trips.
- arq concurrency bounded so Ollama isn't overwhelmed (GPU is the bottleneck).
- Qdrant collection created once with the correct dim; cosine distance.

## Migration Notes

- AWS stubs + adapter factory + `lambda_handler.py` exist so MVP 1's seam is migration-ready. Actual AWS deploy is future work (see roadmap).
- `lambda_handler.py` and `POST /internal/events/minio` are intentionally parallel: both receive an S3-compatible `ObjectCreated` JSON, extract the object key, and trigger ingestion. Migrating to AWS means deploying `lambda_handler.py` as a Lambda function pointed at the S3 bucket — no logic change required.

## References

- Spec: [`../../local_rag_implementation_plan.md`](../../local_rag_implementation_plan.md) (MVP 1 scope: lines 117–181)
- Roadmap: [`2026-06-25-roadmap.md`](2026-06-25-roadmap.md)
- Next: [`2026-06-25-mvp2-evaluation-layer.md`](2026-06-25-mvp2-evaluation-layer.md)
