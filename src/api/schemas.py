from datetime import datetime

from pydantic import BaseModel

from src.evaluation.interfaces.evaluator import EvalResult


class UploadResponse(BaseModel):
    filename: str
    message: str


class IngestionStatus(BaseModel):
    filename: str
    status: str  # pending | completed | failed
    chunks: int | None = None


class DocumentList(BaseModel):
    documents: list[str]


class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False
    max_retries: int = 2


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    eval_result: EvalResult | None = None


class HistoryItem(BaseModel):
    id: int | None = None
    timestamp: datetime
    question: str
    answer: str
    sources: list[str]
    eval_result: EvalResult | None = None
    retries: int = 0


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
