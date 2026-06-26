from datetime import datetime, timezone

from fastapi import APIRouter, Query

from src.adapters.sqlite_history import get_history_store
from src.api.schemas import HistoryItem, HistoryResponse, QueryRequest, QueryResponse
from src.graphs.query_graph import run_query
from src.interfaces.history import HistoryRecord

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    state = await run_query(
        request.question,
        evaluate=request.evaluate,
        max_retries=request.max_retries,
    )
    sources = list(dict.fromkeys(c.source_filename for c in state["chunks"]))

    if request.evaluate:
        store = get_history_store()
        await store.save(
            HistoryRecord(
                timestamp=datetime.now(tz=timezone.utc),
                question=request.question,
                answer=state["answer"],
                sources=sources,
                eval_result=state.get("eval_result"),
                retries=state.get("retries", 0),
            )
        )

    return QueryResponse(
        answer=state["answer"],
        sources=sources,
        eval_result=state.get("eval_result"),
    )


@router.get("/query/history", response_model=HistoryResponse)
async def query_history(limit: int = Query(default=50, ge=1, le=500)):
    store = get_history_store()
    records = await store.list(limit=limit)
    return HistoryResponse(
        items=[
            HistoryItem(
                id=r.id,
                timestamp=r.timestamp,
                question=r.question,
                answer=r.answer,
                sources=r.sources,
                eval_result=r.eval_result,
                retries=r.retries,
            )
            for r in records
        ]
    )
