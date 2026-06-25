"""arq worker: receives ingest_document tasks from the Redis queue."""

from arq.connections import RedisSettings

from src.adapters import get_embedder, get_storage, get_vector_store
from src.config import get_settings
from src.core.ingestion import ingest


async def ingest_document(ctx: dict, filename: str) -> dict:
    """Download file from storage and run the full ingestion pipeline."""
    settings = get_settings()
    storage = get_storage()
    embedder = get_embedder()
    store = get_vector_store()

    data = await storage.download(filename)
    chunks = await ingest(
        filename,
        data,
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return {"filename": filename, "chunks": chunks, "status": "completed"}


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 10
    job_timeout = 300
