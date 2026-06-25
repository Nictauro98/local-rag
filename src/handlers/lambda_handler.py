"""AWS Lambda entry point for S3 ObjectCreated events.

No logic lives here. This function extracts the object key from the S3 event
and delegates to core.ingestion.ingest via the adapter factory. It is the
cloud-side mirror of POST /internal/events/minio: both receive the same
S3-compatible ObjectCreated JSON; only the delivery mechanism differs.

Migration: deploy this file as a Lambda function, set the S3 bucket notification
to invoke it on ObjectCreated events, and ensure the RAG_* environment variables
are set in the Lambda configuration.
"""

import asyncio

from src.adapters import get_embedder, get_storage, get_vector_store
from src.config import get_settings
from src.core.ingestion import ingest


def handler(event: dict, context) -> dict:
    """Invoked by AWS Lambda on S3 ObjectCreated."""
    record = event["Records"][0]
    filename = record["s3"]["object"]["key"]
    return asyncio.run(_run(filename))


async def _run(filename: str) -> dict:
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
