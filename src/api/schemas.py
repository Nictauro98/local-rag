from pydantic import BaseModel


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


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    eval_result: None = None
