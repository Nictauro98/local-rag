"""Unit tests for the arq ingest worker and Lambda handler."""

from unittest.mock import AsyncMock, patch

import pytest

from src.handlers.ingest_worker import ingest_document
from src.handlers.lambda_handler import handler

# ---------------------------------------------------------------------------
# ingest_document task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_document_returns_completed_result():
    ctx = {}

    mock_storage = AsyncMock()
    mock_storage.download.return_value = b"file bytes"

    mock_embedder = AsyncMock()
    mock_store = AsyncMock()

    with (
        patch("src.handlers.ingest_worker.get_storage", return_value=mock_storage),
        patch("src.handlers.ingest_worker.get_embedder", return_value=mock_embedder),
        patch("src.handlers.ingest_worker.get_vector_store", return_value=mock_store),
        patch(
            "src.handlers.ingest_worker.ingest", new_callable=AsyncMock, return_value=5
        ) as mock_ingest,
    ):
        result = await ingest_document(ctx, "report.pdf")

    assert result == {"filename": "report.pdf", "chunks": 5, "status": "completed"}
    mock_storage.download.assert_awaited_once_with("report.pdf")
    mock_ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_document_passes_settings_to_ingest():
    ctx = {}

    mock_storage = AsyncMock()
    mock_storage.download.return_value = b"bytes"

    with (
        patch("src.handlers.ingest_worker.get_storage", return_value=mock_storage),
        patch("src.handlers.ingest_worker.get_embedder", return_value=AsyncMock()),
        patch("src.handlers.ingest_worker.get_vector_store", return_value=AsyncMock()),
        patch(
            "src.handlers.ingest_worker.ingest", new_callable=AsyncMock, return_value=3
        ) as mock_ingest,
        patch("src.handlers.ingest_worker.get_settings") as mock_settings,
    ):
        mock_settings.return_value.chunk_size = 500
        mock_settings.return_value.chunk_overlap = 50
        mock_settings.return_value.redis_url = "redis://localhost:6379"

        await ingest_document(ctx, "doc.txt")

    _, kwargs = mock_ingest.call_args
    assert kwargs["chunk_size"] == 500
    assert kwargs["chunk_overlap"] == 50


@pytest.mark.asyncio
async def test_ingest_document_zero_chunks():
    ctx = {}

    mock_storage = AsyncMock()
    mock_storage.download.return_value = b""

    with (
        patch("src.handlers.ingest_worker.get_storage", return_value=mock_storage),
        patch("src.handlers.ingest_worker.get_embedder", return_value=AsyncMock()),
        patch("src.handlers.ingest_worker.get_vector_store", return_value=AsyncMock()),
        patch("src.handlers.ingest_worker.ingest", new_callable=AsyncMock, return_value=0),
    ):
        result = await ingest_document(ctx, "empty.txt")

    assert result["chunks"] == 0
    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def _s3_event(key: str) -> dict:
    return {"Records": [{"s3": {"object": {"key": key}}}]}


def test_lambda_handler_extracts_key_and_returns_result():
    with patch("src.handlers.lambda_handler._run") as mock_run:
        mock_run.return_value = {"filename": "notes.pdf", "chunks": 7, "status": "completed"}

        with patch(
            "src.handlers.lambda_handler.asyncio.run",
            side_effect=lambda coro: mock_run.return_value,
        ):
            result = handler(_s3_event("notes.pdf"), None)

    assert result["filename"] == "notes.pdf"
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_lambda_run_calls_ingest_with_correct_filename():
    from src.handlers.lambda_handler import _run

    mock_storage = AsyncMock()
    mock_storage.download.return_value = b"data"

    with (
        patch("src.handlers.lambda_handler.get_storage", return_value=mock_storage),
        patch("src.handlers.lambda_handler.get_embedder", return_value=AsyncMock()),
        patch("src.handlers.lambda_handler.get_vector_store", return_value=AsyncMock()),
        patch(
            "src.handlers.lambda_handler.ingest", new_callable=AsyncMock, return_value=4
        ) as mock_ingest,
    ):
        result = await _run("folder/doc.pdf")

    assert result == {"filename": "folder/doc.pdf", "chunks": 4, "status": "completed"}
    mock_storage.download.assert_awaited_once_with("folder/doc.pdf")
    first_arg = mock_ingest.call_args[0][0]
    assert first_arg == "folder/doc.pdf"
