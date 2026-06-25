from urllib.parse import unquote_plus

from arq.jobs import result_key_prefix
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/events/minio", status_code=202)
async def minio_webhook(payload: dict, request: Request):
    """Receive MinIO ObjectCreated webhook and enqueue an ingestion job.

    This is the local mirror of lambda_handler.py: both parse the same
    S3-compatible event JSON and trigger ingest_document with the same filename.
    """
    try:
        # S3 event notifications URL-encode object keys (spaces become +).
        filename = unquote_plus(payload["Records"][0]["s3"]["object"]["key"])
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid S3 event payload") from None

    # arq refuses to re-enqueue if a result already exists for this job ID.
    # Delete it first so re-uploading the same file always triggers fresh ingestion.
    await request.app.state.redis_pool.delete(result_key_prefix + filename)

    await request.app.state.redis_pool.enqueue_job("ingest_document", filename, _job_id=filename)
    return {"enqueued": filename}
