"""Integration test: full ingestion → query cycle against live services.

Run with:
    docker compose up -d
    uv run pytest -m integration -v

Skipped automatically when services are not reachable.
"""

import io
import os
import time

import httpx
import pytest
from minio import Minio

MINIO_ENDPOINT = os.getenv("RAG_MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS = os.getenv("RAG_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("RAG_MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("RAG_MINIO_BUCKET", "documents")
API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")

_FILENAME = "rag_integration_test.txt"
_FACT = (
    "The Zephyr Bridge was completed on March 3, 2024. "
    "It spans exactly 4200 metres and cost 780 million euros."
)
_POLL_TIMEOUT = 120


def _services_up() -> bool:
    try:
        httpx.get(f"{API_BASE}/docs", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


@pytest.mark.integration
def test_upload_query_cycle():
    if not _services_up():
        pytest.skip("Live services not available — run 'docker compose up -d' first")

    # 1. PUT the document directly to MinIO to exercise the webhook trigger path
    mc = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
    data = _FACT.encode()
    mc.put_object(
        MINIO_BUCKET,
        _FILENAME,
        io.BytesIO(data),
        length=len(data),
        content_type="text/plain",
    )

    # 2. Poll until the arq worker finishes ingestion (webhook → enqueue → ingest)
    deadline = time.time() + _POLL_TIMEOUT
    status = "pending"
    while time.time() < deadline:
        resp = httpx.get(f"{API_BASE}/documents/{_FILENAME}/status", timeout=10)
        resp.raise_for_status()
        result = resp.json()
        status = result["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(3)

    assert (
        status == "completed"
    ), f"Ingestion did not complete within {_POLL_TIMEOUT}s (status={status!r})"

    # 3. Query the document
    resp = httpx.post(
        f"{API_BASE}/query",
        json={
            "question": "How long is the Zephyr Bridge and when was it completed?",
            "evaluate": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    payload = resp.json()

    # 4. Assert grounded answer and correct source attribution
    answer = payload["answer"].lower()
    assert any(
        kw in answer for kw in ("zephyr", "4200", "march", "780")
    ), f"Answer does not reference the known fact.\nQuestion about: Zephyr Bridge\nAnswer: {payload['answer']!r}"
    assert (
        _FILENAME in payload["sources"]
    ), f"Expected {_FILENAME!r} in sources, got {payload['sources']!r}"
    assert payload["eval_result"] is None
