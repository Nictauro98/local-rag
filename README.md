# Local RAG

![CI](https://github.com/Nictauro98/local-rag/actions/workflows/ci.yml/badge.svg)

A fully local Retrieval-Augmented Generation system. Upload documents, ask questions, get grounded answers — no cloud calls, no API keys.

```
MinIO ──ObjectCreated──► FastAPI webhook ──► arq worker ──► Qdrant
                                                              │
Streamlit ──POST /query──► LangGraph (retrieve → generate) ◄─┘
                                         │
                                      Ollama
```

## Stack

| Layer | Tool |
|---|---|
| Storage | MinIO (S3-compatible) |
| Vector DB | Qdrant |
| Embeddings + LLM | Ollama (`nomic-embed-text` + `llama3.2`) |
| Job queue | arq + Redis |
| API | FastAPI |
| UI | Streamlit |

## Prerequisites

- Docker + Docker Compose v2
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional — see CPU fallback below)

## Quick start

**1. Start the stack:**

```bash
docker compose up -d
```

On first run, an `ollama-pull` init container automatically pulls `nomic-embed-text` and `llama3.2` (~5 GB total). The API waits for this to complete before starting. Subsequent runs skip the download if models are already cached.

**3. Open the UI:** http://localhost:8501

**4. API docs:** http://localhost:8000/docs

Wait ~30 s on first boot for all health checks to pass. Check status with:

```bash
docker compose ps
```

## CPU fallback

Remove the `deploy` block from the `ollama` service in `docker-compose.yml`:

```yaml
# Remove or comment out:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

Inference will be slower (expect 30–120 s per query depending on hardware).

For CI without a GPU, use `llama3.2:1b` instead of `llama3.2` to reduce memory requirements:

```bash
RAG_LLM_MODEL=llama3.2:1b docker compose up -d
```

## Configuration

All settings are env-driven with `RAG_` prefix. Copy `.env.example` to `.env` to override defaults:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `RAG_LLM_MODEL` | `llama3.2` | Ollama model for generation |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `RAG_CHUNK_SIZE` | `1000` | Characters per chunk |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |

## Development (running without Docker)

```bash
uv sync

# Start infrastructure only
docker compose up -d redis minio qdrant ollama

# Local .env must use localhost addresses (already set up correctly)
# Start API
uv run uvicorn src.api.main:app --reload --host 0.0.0.0

# Start worker (separate terminal)
uv run arq src.handlers.ingest_worker.WorkerSettings

# Start UI (separate terminal)
uv run streamlit run ui/app.py
```

Register the MinIO webhook manually after starting the containers:

```bash
docker exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec minio mc event add local/documents arn:minio:sqs::PRIMARY:webhook --event put
```

Run tests:

```bash
uv run pytest tests/unit
uv run ruff check . && uv run black --check .
```

## Architecture

### Ingestion path

```
User uploads file
  → POST /documents/upload  (stores bytes in MinIO, returns immediately)
  → MinIO fires ObjectCreated event
  → POST /internal/events/minio  (webhook endpoint)
  → arq enqueues ingest_document job in Redis
  → worker: download → parse → chunk → embed → upsert to Qdrant
  → GET /documents/{filename}/status  (polls Redis job result)
```

### Query path

```
User submits question
  → POST /query
  → LangGraph: retrieve node (embed query → Qdrant search)
  → LangGraph: generate node (build grounded prompt → Ollama /api/generate)
  → return {answer, sources}
```

### Why this mirrors AWS

The local ingestion trigger (MinIO → webhook → arq) is a structural mirror of the AWS path (S3 → Lambda). Both receive the same S3-compatible `ObjectCreated` JSON and call the same `core.ingestion.ingest` function. Migrating to AWS means deploying `src/handlers/lambda_handler.py` as a Lambda pointed at an S3 bucket — no logic change required.

## AWS migration guide

| Component | Local | AWS |
|---|---|---|
| Storage | MinIO | S3 (`S3Storage` stub in `src/adapters/s3_storage.py`) |
| Embeddings | Ollama | Amazon Bedrock (`BedrockEmbedder` stub in `src/adapters/bedrock_embedder.py`) |
| Ingestion trigger | MinIO webhook → FastAPI → arq | S3 ObjectCreated → Lambda (`src/handlers/lambda_handler.py`) |
| Job queue | arq + Redis | Lambda is the queue (or SQS if fan-out needed) |
| Vector DB | Qdrant (self-hosted) | Qdrant Cloud or OpenSearch |
| LLM | Ollama | Amazon Bedrock |

Swap `RAG_STORAGE_BACKEND=aws` and `RAG_EMBEDDER_BACKEND=aws` in `.env` to switch adapters. Implement the stubs using boto3.
