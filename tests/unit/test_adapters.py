"""Unit tests for adapters — all SDK clients are mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adapters.bedrock_embedder import BedrockEmbedder
from src.adapters.s3_storage import S3Storage

# ---------------------------------------------------------------------------
# MinIOStorage
# ---------------------------------------------------------------------------


@pytest.fixture()
def minio_client_mock():
    with patch("src.adapters.minio_storage.Minio") as mock_cls:
        client = MagicMock()
        client.bucket_exists.return_value = True
        mock_cls.return_value = client
        yield client


@pytest.mark.asyncio
async def test_minio_upload(minio_client_mock):
    from src.adapters.minio_storage import MinIOStorage

    storage = MinIOStorage("localhost:9000", "key", "secret", "bucket")
    result = await storage.upload("file.txt", b"hello")
    assert result == "file.txt"
    minio_client_mock.put_object.assert_called_once()


@pytest.mark.asyncio
async def test_minio_download(minio_client_mock):
    from src.adapters.minio_storage import MinIOStorage

    response = MagicMock()
    response.read.return_value = b"content"
    minio_client_mock.get_object.return_value = response

    storage = MinIOStorage("localhost:9000", "key", "secret", "bucket")
    data = await storage.download("file.txt")
    assert data == b"content"


@pytest.mark.asyncio
async def test_minio_list(minio_client_mock):
    from src.adapters.minio_storage import MinIOStorage

    obj = MagicMock()
    obj.object_name = "file.txt"
    minio_client_mock.list_objects.return_value = [obj]

    storage = MinIOStorage("localhost:9000", "key", "secret", "bucket")
    names = await storage.list()
    assert names == ["file.txt"]


@pytest.mark.asyncio
async def test_minio_exists_true(minio_client_mock):
    from src.adapters.minio_storage import MinIOStorage

    storage = MinIOStorage("localhost:9000", "key", "secret", "bucket")
    assert await storage.exists("file.txt") is True


@pytest.mark.asyncio
async def test_minio_exists_false(minio_client_mock):
    from minio.error import S3Error

    from src.adapters.minio_storage import MinIOStorage

    minio_client_mock.stat_object.side_effect = S3Error(
        "NoSuchKey", "not found", "resource", "request-id", "host-id", MagicMock()
    )
    storage = MinIOStorage("localhost:9000", "key", "secret", "bucket")
    assert await storage.exists("missing.txt") is False


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_embed():
    from src.adapters.ollama_embedder import OllamaEmbedder

    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_ctx

        result = await embedder.embed(["hello"])
        assert result == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_ollama_resolve_dimension():
    from src.adapters.ollama_embedder import OllamaEmbedder

    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_ctx.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_ctx

        dim = await embedder.resolve_dimension()
        assert dim == 768
        assert embedder.dimension() == 768


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qdrant_ensure_collection_creates_when_absent():
    from src.adapters.qdrant_store import QdrantStore

    with patch("src.adapters.qdrant_store.AsyncQdrantClient") as mock_cls:
        client = AsyncMock()
        collections_response = MagicMock()
        collections_response.collections = []
        client.get_collections = AsyncMock(return_value=collections_response)
        mock_cls.return_value = client

        store = QdrantStore("http://localhost:6333", "chunks")
        await store.ensure_collection(768)
        client.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_qdrant_ensure_collection_skips_when_present():
    from src.adapters.qdrant_store import QdrantStore

    with patch("src.adapters.qdrant_store.AsyncQdrantClient") as mock_cls:
        client = AsyncMock()
        existing = MagicMock()
        existing.name = "chunks"
        collections_response = MagicMock()
        collections_response.collections = [existing]
        client.get_collections = AsyncMock(return_value=collections_response)
        mock_cls.return_value = client

        store = QdrantStore("http://localhost:6333", "chunks")
        await store.ensure_collection(768)
        client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_qdrant_delete_by_source():
    from src.adapters.qdrant_store import QdrantStore

    with patch("src.adapters.qdrant_store.AsyncQdrantClient") as mock_cls:
        client = AsyncMock()
        mock_cls.return_value = client

        store = QdrantStore("http://localhost:6333", "chunks")
        await store.delete_by_source("doc.pdf")
        client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_qdrant_search():
    from src.adapters.qdrant_store import QdrantStore

    with patch("src.adapters.qdrant_store.AsyncQdrantClient") as mock_cls:
        client = AsyncMock()
        hit = MagicMock()
        hit.payload = {"text": "chunk text", "source_filename": "doc.pdf", "chunk_index": 0}
        hit.score = 0.95
        mock_response = MagicMock()
        mock_response.points = [hit]
        client.query_points = AsyncMock(return_value=mock_response)
        mock_cls.return_value = client

        store = QdrantStore("http://localhost:6333", "chunks")
        results = await store.search([0.1] * 768, top_k=5)
        assert len(results) == 1
        assert results[0].text == "chunk text"
        assert results[0].score == 0.95


# ---------------------------------------------------------------------------
# AWS stubs raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_storage_raises():
    s = S3Storage()
    with pytest.raises(NotImplementedError):
        await s.upload("f", b"")
    with pytest.raises(NotImplementedError):
        await s.download("f")
    with pytest.raises(NotImplementedError):
        await s.list()
    with pytest.raises(NotImplementedError):
        await s.exists("f")


def test_bedrock_embedder_raises():
    b = BedrockEmbedder()
    with pytest.raises(NotImplementedError):
        b.dimension()
