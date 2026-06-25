from fastapi import APIRouter

from src.api.schemas import QueryRequest, QueryResponse
from src.graphs.query_graph import run_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    state = await run_query(request.question, evaluate=request.evaluate)
    # Deduplicate sources while preserving order of first appearance.
    sources = list(dict.fromkeys(c.source_filename for c in state["chunks"]))
    return QueryResponse(answer=state["answer"], sources=sources, eval_result=None)
