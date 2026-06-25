# Local RAG System — Implementation Plan for Claude Code

## Project Overview

Build a **production-grade, privacy-first Retrieval-Augmented Generation (RAG) system** designed to run entirely on local or company-controlled infrastructure. The core motivation is data sovereignty: documents never leave the company's network perimeter, making this architecture compliant with data residency regulations such as GDPR, LGPD, and Argentina's Ley 25.326.

The system allows users to upload documents through a simple web UI, ask natural language questions about their content, and receive grounded, verifiable answers from a local LLM. A dedicated evaluation layer (MVP 2) guards against hallucinations and measures answer quality without relying on any external API.

The architecture is designed from day one for a future migration to AWS: storage, embedding, and LLM adapters are abstracted behind interfaces so that swapping MinIO for S3, or Ollama for Bedrock, requires changing a single adapter file with no changes to business logic.

---

## System Architecture Summary

The system is composed of four logical layers:

1. **Storage layer** — MinIO (S3-compatible), used to receive and store raw documents
2. **Ingestion pipeline** — triggered when a new document arrives; handles parsing, chunking, embedding, and vector storage
3. **Query pipeline** — a LangGraph graph that handles retrieval, generation, and (in MVP 2) evaluation and conditional retry
4. **Interface layer** — a FastAPI backend exposing REST endpoints, and a Streamlit frontend for user interaction

All LLM inference and embedding runs locally via Ollama on the available GPU. The vector store is Qdrant, running as a Docker container.

---

## Tech Stack

| Concern | Technology | Rationale |
|---|---|---|
| Object storage | MinIO | S3-compatible API; drop-in swap to AWS S3 |
| Vector store | Qdrant | Self-hosted, Docker-native, production-ready |
| LLM inference | Ollama | GPU-accelerated, local, no external calls |
| Embedding model | `nomic-embed-text` via Ollama | High quality, runs locally |
| LLM model | `llama3.2` or `mistral` via Ollama | Configurable via environment variable |
| Query orchestration | LangGraph | Stateful graph with conditional routing; AWS-migratable |
| API backend | FastAPI + async Python | Standard, performant, production-grade |
| User interface | Streamlit | Rapid, functional UI; sufficient for portfolio demo |
| Document parsing | `pypdf`, `python-docx`, `unstructured` | Multi-format support (PDF, DOCX, TXT) |
| Text chunking | `langchain-text-splitters` (only this subpackage) | Avoids full LangChain coupling |
| NLI evaluation model | `cross-encoder/nli-deberta-v3-small` via `sentence-transformers` | Local faithfulness scoring (MVP 2) |
| Containerization | Docker + Docker Compose | Full stack runs with single command |
| Testing | pytest + pytest-asyncio | Unit and integration tests per layer |
| Linting / formatting | ruff + black | Project standard |
| Configuration | `pydantic-settings` + `.env` | Environment-driven, AWS-ready |
| CI | GitHub Actions | Lint, test, build on push |

---

## Project Structure

```
local-rag/
├── src/
│   ├── interfaces/          # Abstract base classes (contracts)
│   │   ├── storage.py
│   │   ├── embedder.py
│   │   └── vector_store.py
│   │
│   ├── adapters/            # Concrete implementations of interfaces
│   │   ├── minio_storage.py
│   │   ├── s3_storage.py        # Stub — ready for AWS migration
│   │   ├── ollama_embedder.py
│   │   ├── bedrock_embedder.py  # Stub — ready for AWS migration
│   │   └── qdrant_store.py
│   │
│   ├── core/                # Pure business logic — no framework imports
│   │   ├── ingestion.py     # Orchestrates parse → chunk → embed → upsert
│   │   ├── chunking.py      # Text splitting strategies
│   │   └── parsing.py       # Multi-format document parsing
│   │
│   ├── nodes/               # LangGraph node functions (one per file)
│   │   ├── retrieve.py
│   │   ├── generate.py
│   │   ├── evaluate.py      # MVP 2
│   │   └── rewrite_query.py # MVP 2
│   │
│   ├── graphs/
│   │   └── query_graph.py   # LangGraph StateGraph definition and compilation
│   │
│   ├── evaluation/          # MVP 2 — evaluation layer
│   │   ├── interfaces/
│   │   │   └── evaluator.py
│   │   ├── deterministic.py
│   │   ├── nli_evaluator.py
│   │   ├── llm_judge.py
│   │   └── composite_evaluator.py
│   │
│   ├── handlers/            # Entry points — thin adapters only, no logic
│   │   ├── watchdog_handler.py  # Local file watcher trigger
│   │   └── lambda_handler.py   # AWS Lambda-compatible entry point
│   │
│   ├── api/
│   │   ├── main.py          # FastAPI app instantiation
│   │   ├── routes/
│   │   │   ├── documents.py # Upload endpoint
│   │   │   └── query.py     # Query endpoint
│   │   └── schemas.py       # Pydantic request/response models
│   │
│   └── config.py            # pydantic-settings configuration
│
├── ui/
│   └── app.py               # Streamlit application
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

---

## MVP 1 — Working Local RAG System

**Goal:** A fully functional end-to-end RAG system running locally. User can upload documents, ask questions, and receive answers grounded in document content. The evaluation hook is in place but inactive.

### Scope

**Infrastructure**
- Docker Compose file that starts MinIO, Qdrant, and Ollama with a single `docker compose up` command
- Ollama container configured to use the host GPU via NVIDIA container toolkit
- MinIO configured with a default bucket for document storage
- Health checks on all services before application starts

**Interfaces (contracts)**
- Define abstract base classes for `StorageBackend`, `Embedder`, and `VectorStore`
- These are the migration boundary — all business logic depends only on these, never on concrete implementations

**Adapters**
- `MinIOStorage`: upload, download, list files using the MinIO Python SDK
- `OllamaEmbedder`: call Ollama's embedding endpoint via httpx, return vectors as lists of floats
- `QdrantStore`: upsert vectors with payload (chunk text, source file, chunk index); similarity search returning top-k chunks
- `S3Storage` and `BedrockEmbedder`: stubbed with `NotImplementedError` and docstrings explaining migration steps

**Ingestion pipeline**
- Triggered by the watchdog handler monitoring a local directory, or by the MinIO event notification webhook
- Parses uploaded files: PDF via pypdf, DOCX via python-docx, plain text natively
- Splits text into overlapping chunks using `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- Embeds chunks using `OllamaEmbedder`
- Upserts vectors to Qdrant with metadata (source filename, chunk index, raw text)
- Idempotent: re-uploading the same file deletes existing vectors for that source before reinserting

**LangGraph query graph**
- `RAGState` TypedDict with fields: `query`, `chunks`, `answer`, `eval_result` (None in MVP 1)
- Nodes: `retrieve` → `generate`
- `retrieve` node: embeds the query, searches Qdrant for top-k chunks, stores in state
- `generate` node: constructs a prompt with retrieved context, calls Ollama LLM, stores answer in state
- Graph compiled and exposed as a callable in `graphs/query_graph.py`
- Evaluation hook: `evaluate: bool = False` parameter in the query route; when False, `eval_result` is None and the graph skips evaluation nodes

**FastAPI backend**
- `POST /documents/upload`: accepts a file, stores it in MinIO, triggers ingestion pipeline asynchronously
- `GET /documents`: lists all ingested documents
- `POST /query`: accepts `{ "question": str, "evaluate": bool }`, runs the LangGraph query graph, returns `{ "answer": str, "sources": list[str], "eval_result": null }`
- All endpoints are async
- OpenAPI docs auto-generated at `/docs`

**Streamlit UI**
- Sidebar: file uploader widget; on upload, calls `POST /documents/upload` and shows ingestion status
- Sidebar: list of ingested documents
- Main area: chat-style interface with a text input for questions
- Each answer displays the answer text and the source document chunks used (expandable)
- No authentication — local use only in MVP 1

**Configuration**
- All service URLs, model names, chunk size, chunk overlap, and top-k are environment variables with defaults
- `.env.example` documents every variable

**Testing**
- Unit tests for ingestion pipeline with mocked adapters
- Unit tests for each LangGraph node in isolation
- Integration test that runs a full upload → query cycle against live Docker services

### MVP 1 Deliverables
- Running system accessible at `http://localhost:8501` (Streamlit) and `http://localhost:8000` (API)
- README with setup instructions, architecture diagram description, and AWS migration guide
- All tests passing in CI

---

## MVP 2 — Evaluation Layer

**Goal:** Add a multi-tier evaluation layer to the query pipeline that detects hallucinations, measures faithfulness, and optionally retries with a rewritten query when quality is below threshold. Evaluation results are surfaced in the UI.

### Scope

**Evaluation interfaces and result model**
- `Evaluator` abstract base class with method `evaluate(query, context, answer) → EvalResult`
- `EvalResult` Pydantic model with fields: `faithfulness` (float 0–1), `context_relevance` (float 0–1), `answer_grounded` (bool), `flagged` (bool), `judge_reasoning` (str or None)
- `flagged` is True when any score falls below its configured threshold

**Tier 1 — Deterministic evaluator**
- Context relevance: cosine similarity between the embedded query vector and the mean of retrieved chunk vectors
- Answer grounding: token overlap check between answer and retrieved chunks
- No LLM call; executes in milliseconds

**Tier 2 — NLI evaluator**
- Faithfulness scoring using `cross-encoder/nli-deberta-v3-small` from `sentence-transformers`
- Checks whether the generated answer is entailed by the retrieved context
- Runs locally on GPU; no external calls

**Tier 3 — LLM-as-judge evaluator**
- Secondary Ollama call using a smaller/faster model (configurable, e.g. `phi3`)
- Structured prompt asks the judge to rate faithfulness and groundedness on a 1–5 scale and explain its reasoning
- Response parsed into `EvalResult` using Pydantic; malformed responses default to flagged
- The judge model is distinct from the generation model to avoid self-evaluation bias

**Composite evaluator**
- Runs all enabled tiers in sequence (configurable via environment variables)
- Aggregates scores; `flagged` is True if any tier flags the result
- Each tier can be independently enabled or disabled

**LangGraph graph extension**
- New nodes added to the existing graph: `evaluate` and `rewrite_query`
- After `generate`, graph routes to `evaluate`
- `evaluate` node runs the composite evaluator and stores `EvalResult` in state
- Conditional edge from `evaluate`: if `flagged` is False, route to END; if True, route to `rewrite_query`
- `rewrite_query` node uses the LLM to rephrase the original query based on the failure reason, then loops back to `retrieve`
- Maximum retry count (default: 2) is enforced in state to prevent infinite loops

**API update**
- `POST /query` now returns `eval_result` populated when `evaluate: true` is passed
- New endpoint `GET /query/history` returns past queries with their eval results (stored in-memory or in a simple SQLite file)

**Streamlit UI update**
- When evaluation is enabled (toggle in UI), each answer displays an evaluation panel showing: faithfulness score, context relevance score, grounding status, flagged status, and judge reasoning if available
- Color-coded indicators: green (pass), yellow (borderline), red (flagged)
- Query history tab showing past questions, answers, and eval scores

**Testing**
- Unit tests for each evaluator in isolation with mock inputs
- Integration test: submit a question with a deliberately misleading context and assert the response is flagged
- Integration test: verify the retry loop triggers and resolves within the max retry count

### MVP 2 Deliverables
- Evaluation panel visible in the UI for every query when evaluation is toggled on
- All three evaluator tiers operational and independently toggleable
- Retry loop functional and bounded
- Updated README documenting the evaluation architecture and how to interpret scores

---

## AWS Migration Path (Reference)

This section documents the migration steps when moving to AWS. No code changes are required outside the adapter layer.

| Local component | AWS replacement | Change required |
|---|---|---|
| MinIO | S3 | Swap `MinIOStorage` for `S3Storage` adapter |
| Watchdog handler | Lambda + S3 event trigger | Deploy `lambda_handler.py` as Lambda entrypoint |
| Ollama embedder | Amazon Bedrock Embeddings | Swap `OllamaEmbedder` for `BedrockEmbedder` adapter |
| Ollama LLM | Amazon Bedrock / SageMaker | Update `generate` node to call Bedrock API |
| Qdrant | Amazon OpenSearch (k-NN) or Pinecone | Swap `QdrantStore` for new adapter |
| FastAPI (local) | AWS Lambda + API Gateway or ECS Fargate | Package existing Dockerfile; push to ECR |
| Streamlit (local) | Amplify or CloudFront + S3 | Rebuild UI or host as-is on EC2 |

The `Dockerfile` is Lambda container-compatible. The `lambda_handler.py` file already exists and receives the S3 event JSON — no rewriting required.

---

## Development Conventions

- All business logic in `src/core/` and `src/nodes/` must be independently testable with mocked dependencies
- No adapter is imported directly in `core/` or `nodes/` — always injected via function parameters
- Environment variables are the only configuration mechanism — no hardcoded values
- All async functions use `asyncio`; no blocking calls in async context
- Every public function has a docstring
- `ruff` and `black` must pass before any commit (enforced by pre-commit hook and CI)
- Docker Compose is the only supported local setup method — no manual service installation instructions

---

## Notes for the Agent

- Do not use the full `langchain` package. Only `langchain-text-splitters` is permitted as a LangChain dependency. All other LangChain abstractions (retrievers, chains, agents) must be avoided to preserve architectural control.
- The `lambda_handler.py` must remain a thin adapter — any logic added to it should be moved to `core/` instead.
- The evaluation layer in MVP 2 must not block the query response in MVP 1. The `evaluate: bool` flag controls this at the API level.
- Ollama model names must be configurable via environment variable — never hardcoded.
- GPU access in Docker requires the NVIDIA container toolkit. Add a note in README if the GPU is unavailable and document CPU fallback configuration for Ollama.
