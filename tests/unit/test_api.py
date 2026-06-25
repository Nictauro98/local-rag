"""Unit tests for the FastAPI routes (mocked storage, queue, and query graph)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from arq.jobs import JobStatus
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test fixture: app with lifespan mocked out
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock()
    pool.aclose = AsyncMock()
    return pool


@pytest.fixture
def client(mock_pool):
    with (
        patch("src.api.main.get_embedder") as mock_get_embedder,
        patch("src.api.main.get_vector_store") as mock_get_store,
        patch("src.api.main.create_pool", new_callable=AsyncMock, return_value=mock_pool),
    ):
        mock_embedder = AsyncMock()
        mock_embedder.embed_one.return_value = [0.1] * 384
        mock_get_embedder.return_value = mock_embedder

        mock_get_store.return_value = AsyncMock()

        from src.api.main import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------


def test_upload_stores_file_and_returns_filename(client):
    mock_storage = AsyncMock()
    with patch("src.api.routes.documents.get_storage", return_value=mock_storage):
        response = client.post(
            "/documents/upload",
            files={"file": ("report.pdf", b"fake pdf content", "application/pdf")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert "ingestion" in body["message"]
    mock_storage.upload.assert_awaited_once_with("report.pdf", b"fake pdf content")


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


def test_list_documents_returns_filenames(client):
    mock_storage = AsyncMock()
    mock_storage.list.return_value = ["a.pdf", "b.txt"]
    with patch("src.api.routes.documents.get_storage", return_value=mock_storage):
        response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": ["a.pdf", "b.txt"]}


# ---------------------------------------------------------------------------
# GET /documents/{filename}/status
# ---------------------------------------------------------------------------


def test_status_completed(client):
    mock_job = AsyncMock()
    mock_job.status.return_value = JobStatus.complete
    mock_job.result.return_value = {"filename": "doc.pdf", "chunks": 7, "status": "completed"}
    with patch("src.api.routes.documents.Job", return_value=mock_job):
        response = client.get("/documents/doc.pdf/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["chunks"] == 7


def test_status_pending_when_not_found(client):
    mock_job = AsyncMock()
    mock_job.status.return_value = JobStatus.not_found
    with patch("src.api.routes.documents.Job", return_value=mock_job):
        response = client.get("/documents/doc.pdf/status")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_status_pending_while_queued(client):
    mock_job = AsyncMock()
    mock_job.status.return_value = JobStatus.queued
    with patch("src.api.routes.documents.Job", return_value=mock_job):
        response = client.get("/documents/doc.pdf/status")
    assert response.json()["status"] == "pending"


def test_status_failed_when_result_raises(client):
    mock_job = AsyncMock()
    mock_job.status.return_value = JobStatus.complete
    mock_job.result.side_effect = Exception("job exploded")
    with patch("src.api.routes.documents.Job", return_value=mock_job):
        response = client.get("/documents/doc.pdf/status")
    assert response.json()["status"] == "failed"


# ---------------------------------------------------------------------------
# POST /internal/events/minio
# ---------------------------------------------------------------------------


def _s3_payload(key: str) -> dict:
    return {"Records": [{"s3": {"object": {"key": key}}}]}


def test_minio_webhook_enqueues_job(client, mock_pool):
    response = client.post("/internal/events/minio", json=_s3_payload("notes.pdf"))
    assert response.status_code == 202
    assert response.json()["enqueued"] == "notes.pdf"
    mock_pool.enqueue_job.assert_awaited_once_with(
        "ingest_document", "notes.pdf", _job_id="notes.pdf"
    )


def test_minio_webhook_returns_400_for_missing_records(client):
    response = client.post("/internal/events/minio", json={"bad": "payload"})
    assert response.status_code == 400


def test_minio_webhook_returns_400_for_empty_records(client):
    response = client.post("/internal/events/minio", json={"Records": []})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


def test_query_returns_answer_and_sources(client):
    mock_chunk = MagicMock()
    mock_chunk.source_filename = "report.pdf"
    mock_state = {
        "answer": "The answer is 42.",
        "chunks": [mock_chunk, mock_chunk],
        "eval_result": None,
    }
    with patch("src.api.routes.query.run_query", new_callable=AsyncMock, return_value=mock_state):
        response = client.post("/query", json={"question": "What is the answer?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer is 42."
    assert body["sources"] == ["report.pdf"]  # deduplicated
    assert body["eval_result"] is None


def test_query_empty_chunks_returns_empty_sources(client):
    mock_state = {"answer": "I don't know.", "chunks": [], "eval_result": None}
    with patch("src.api.routes.query.run_query", new_callable=AsyncMock, return_value=mock_state):
        response = client.post("/query", json={"question": "anything?"})
    assert response.json()["sources"] == []
