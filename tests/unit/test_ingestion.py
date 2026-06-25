"""Unit tests for core ingestion — parse, chunk, and ingest with mocked deps."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.chunking import chunk
from src.core.ingestion import ingest

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_txt():
    from src.core.parsing import parse

    data = b"Hello world"
    assert parse("doc.txt", data) == "Hello world"


def test_parse_txt_unknown_extension_falls_through():
    from src.core.parsing import parse

    # Unknown extension falls to unstructured; we mock it to avoid the dep
    with patch("src.core.parsing._parse_unstructured", return_value="fallback") as m:
        result = parse("doc.xyz", b"data")
        assert result == "fallback"
        m.assert_called_once()


def test_parse_pdf():
    from src.core.parsing import parse

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "page text"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("src.core.parsing.PdfReader", return_value=mock_reader):
        result = parse("doc.pdf", b"fake-pdf-bytes")
        assert result == "page text"


def test_parse_docx():
    from src.core.parsing import parse

    mock_para = MagicMock()
    mock_para.text = "paragraph text"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]

    with patch("src.core.parsing.Document", return_value=mock_doc):
        result = parse("doc.docx", b"fake-docx-bytes")
        assert result == "paragraph text"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_splits_long_text():
    text = "word " * 500  # 2500 chars
    chunks = chunk(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)  # slight tolerance for overlap


def test_chunk_short_text_returns_one_chunk():
    text = "short text"
    chunks = chunk(text, size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_chunk_overlap():
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk(text, size=50, overlap=10)
    # Verify overlap: end of chunk N should appear at start of chunk N+1
    for i in range(len(chunks) - 1):
        assert chunks[i][-10:] in chunks[i + 1] or chunks[i + 1][:10] in chunks[i]


# ---------------------------------------------------------------------------
# Ingestion orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_calls_delete_before_upsert():
    """delete_by_source must be called before upsert — idempotency guarantee."""
    call_order = []

    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    store = AsyncMock()
    store.delete_by_source.side_effect = lambda _: call_order.append("delete")
    store.upsert.side_effect = lambda _: call_order.append("upsert")

    with (
        patch("src.core.ingestion.parse", return_value="some text"),
        patch("src.core.ingestion.chunk", return_value=["chunk one"]),
    ):
        count = await ingest(
            "doc.txt",
            b"data",
            embedder=embedder,
            store=store,
            chunk_size=1000,
            chunk_overlap=200,
        )

    assert count == 1
    assert call_order == ["delete", "upsert"], "delete must precede upsert"


@pytest.mark.asyncio
async def test_ingest_returns_chunk_count():
    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1] * 3, [0.2] * 3, [0.3] * 3]

    store = AsyncMock()

    with (
        patch("src.core.ingestion.parse", return_value="text"),
        patch("src.core.ingestion.chunk", return_value=["a", "b", "c"]),
    ):
        count = await ingest(
            "doc.txt",
            b"data",
            embedder=embedder,
            store=store,
            chunk_size=1000,
            chunk_overlap=200,
        )

    assert count == 3


@pytest.mark.asyncio
async def test_ingest_empty_text_returns_zero():
    embedder = AsyncMock()
    store = AsyncMock()

    with (
        patch("src.core.ingestion.parse", return_value=""),
        patch("src.core.ingestion.chunk", return_value=[]),
    ):
        count = await ingest(
            "empty.txt",
            b"",
            embedder=embedder,
            store=store,
            chunk_size=1000,
            chunk_overlap=200,
        )

    assert count == 0
    store.delete_by_source.assert_not_called()
    store.upsert.assert_not_called()
