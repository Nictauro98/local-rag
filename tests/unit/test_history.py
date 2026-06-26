"""Unit tests for SQLiteHistory — save/list round-trip."""

from datetime import UTC, datetime

import pytest

from src.adapters.sqlite_history import SQLiteHistory
from src.evaluation.interfaces.evaluator import EvalResult
from src.interfaces.history import HistoryRecord


def _record(eval_result=None, retries=0):
    return HistoryRecord(
        timestamp=datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        question="What is the capital of France?",
        answer="Paris.",
        sources=["doc.pdf"],
        eval_result=eval_result,
        retries=retries,
    )


def _full_eval():
    return EvalResult(
        faithfulness=0.85,
        context_relevance=0.75,
        answer_grounded=True,
        flagged=False,
        judge_reasoning="Well supported by context.",
        tier_scores={"deterministic_faithfulness": 0.85},
        retries=1,
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteHistory(db_path=str(tmp_path / "test.db"))


@pytest.mark.asyncio
async def test_save_and_list_round_trip(store):
    await store.save(_record())
    records = await store.list()

    assert len(records) == 1
    r = records[0]
    assert r.question == "What is the capital of France?"
    assert r.answer == "Paris."
    assert r.sources == ["doc.pdf"]
    assert r.eval_result is None
    assert r.retries == 0


@pytest.mark.asyncio
async def test_eval_result_survives_round_trip(store):
    ev = _full_eval()
    await store.save(_record(eval_result=ev, retries=1))
    records = await store.list()

    r = records[0]
    assert r.eval_result is not None
    assert r.eval_result.faithfulness == pytest.approx(0.85)
    assert r.eval_result.context_relevance == pytest.approx(0.75)
    assert r.eval_result.answer_grounded is True
    assert r.eval_result.flagged is False
    assert r.eval_result.judge_reasoning == "Well supported by context."
    assert r.eval_result.tier_scores == {"deterministic_faithfulness": 0.85}
    assert r.retries == 1


@pytest.mark.asyncio
async def test_list_returns_most_recent_first(store):
    await store.save(_record())
    await store.save(
        HistoryRecord(
            timestamp=datetime(2026, 6, 25, 13, 0, 0, tzinfo=UTC),
            question="Second question?",
            answer="Second answer.",
            sources=[],
        )
    )
    records = await store.list()

    assert records[0].question == "Second question?"
    assert records[1].question == "What is the capital of France?"


@pytest.mark.asyncio
async def test_list_respects_limit(store):
    for i in range(5):
        await store.save(
            HistoryRecord(
                timestamp=datetime(2026, 6, 25, 12, i, 0, tzinfo=UTC),
                question=f"Question {i}",
                answer=f"Answer {i}",
                sources=[],
            )
        )
    records = await store.list(limit=3)

    assert len(records) == 3


@pytest.mark.asyncio
async def test_empty_store_returns_empty_list(store):
    assert await store.list() == []
