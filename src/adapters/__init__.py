import os

from src.interfaces.embedder import Embedder
from src.interfaces.storage import StorageBackend
from src.interfaces.vector_store import VectorStore


def get_storage() -> StorageBackend:
    backend = os.getenv("RAG_STORAGE_BACKEND", "local")
    if backend == "aws":
        from src.adapters.s3_storage import S3Storage

        return S3Storage()
    from src.adapters.minio_storage import MinIOStorage
    from src.config import get_settings

    s = get_settings()
    return MinIOStorage(
        endpoint=s.minio_endpoint,
        access_key=s.minio_access_key,
        secret_key=s.minio_secret_key,
        bucket=s.minio_bucket,
        secure=s.minio_secure,
    )


def get_embedder() -> Embedder:
    backend = os.getenv("RAG_EMBEDDER_BACKEND", "local")
    if backend == "aws":
        from src.adapters.bedrock_embedder import BedrockEmbedder

        return BedrockEmbedder()
    from src.adapters.ollama_embedder import OllamaEmbedder
    from src.config import get_settings

    s = get_settings()
    return OllamaEmbedder(base_url=s.ollama_url, model=s.embed_model)


def get_vector_store() -> VectorStore:
    from src.adapters.qdrant_store import QdrantStore
    from src.config import get_settings

    s = get_settings()
    return QdrantStore(url=s.qdrant_url, collection=s.qdrant_collection)
