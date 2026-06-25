from typing import Annotated

from arq.jobs import Job, JobStatus
from fastapi import APIRouter, File, Request, UploadFile

from src.adapters import get_storage
from src.api.schemas import DocumentList, IngestionStatus, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: Annotated[UploadFile, File()]):
    storage = get_storage()
    data = await file.read()
    await storage.upload(file.filename, data)
    return UploadResponse(
        filename=file.filename,
        message="uploaded; ingestion will begin via storage event",
    )


@router.get("", response_model=DocumentList)
async def list_documents():
    storage = get_storage()
    documents = await storage.list()
    return DocumentList(documents=documents)


@router.get("/{filename:path}/status", response_model=IngestionStatus)
async def get_document_status(filename: str, request: Request):
    job = Job(job_id=filename, redis=request.app.state.redis_pool)
    status = await job.status()
    if status == JobStatus.complete:
        try:
            result = await job.result(timeout=0)
            return IngestionStatus(filename=filename, status="completed", chunks=result["chunks"])
        except Exception:
            return IngestionStatus(filename=filename, status="failed")
    return IngestionStatus(filename=filename, status="pending")
