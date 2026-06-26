"""Unit tests for eval-related API routes."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.evaluation.interfaces.evaluator import EvalResult
from src.interfaces.history import HistoryRecord

# ---------------------------------------------------------------------------
# Shared fixtures (mirrors test_api.py setup)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock()
    pool.aclose = AsyncMock()
    return pool


@pytest.fixture
def client(mock_pool):
    with (
        patch("src.api.main.get_embedder") as mock_get_embedder,
        patch("src.api.main.get_vector_store") as mock_get_store,
        patch("src.api.main.create_pool", new_callable=AsyncMock, return_value=mock_pool),
    ):
        mock_embedder = AsyncMock()
        mock_embedder.embed_one.return_value = [0.1] * 384
        mock_get_embedder.return_value = mock_embedder
        mock_get_store.return_value = AsyncMock()

        from src.api.main import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_result(flagged=False):
    return EvalResult(
        faithfulness=0.85,
        context_relevance=0.75,
        answer_grounded=True,
        flagged=flagged,
        judge_reasoning="Looks good.",
        tier_scores={"deterministic_faithfulness": 0.85},
    )


def _mock_state(eval_result=None, retries=0):
    chunk = MagicMock()
    chunk.source_filename = "doc.pdf"
    return {
        "answer": "Paris.",
        "chunks": [chunk],
        "eval_result": eval_result,
        "retries": retries,
    }


def _mock_store(records=None):
    store = AsyncMock()
    store.save = AsyncMock()
    store.list = AsyncMock(return_value=records or [])
    return store


# ---------------------------------------------------------------------------
# POST /query with evaluate=true
# ---------------------------------------------------------------------------


def test_query_with_evaluate_returns_populated_eval_result(client):
    ev = _eval_result()
    state = _mock_state(eval_result=ev, retries=0)
    mock_store = _mock_store()

    with (
        patch("src.api.routes.query.run_query", new_callable=AsyncMock, return_value=state),
        patch("src.api.routes.query.get_history_store", return_value=mock_store),
    ):
        response = client.post("/query", json={"question": "Capital of France?", "evaluate": True})

    assert response.status_code == 200
    body = response.json()
    assert body["eval_result"] is not None
    assert body["eval_result"]["faithfulness"] == pytest.approx(0.85)
    assert body["eval_result"]["flagged"] is False
    assert body["eval_result"]["judge_reasoning"] == "Looks good."
    mock_store.save.assert_awaited_once()


def test_query_with_evaluate_false_does_not_persist(client):
    state = _mock_state(eval_result=None)
    mock_store = _mock_store()

    with (
        patch("src.api.routes.query.run_query", new_callable=AsyncMock, return_value=state),
        patch("src.api.routes.query.get_history_store", return_value=mock_store),
    ):
        response = client.post("/query", json={"question": "q?", "evaluate": False})

    assert response.status_code == 200
    mock_store.save.assert_not_awaited()


def test_query_flagged_result_surfaces_in_response(client):
    ev = _eval_result(flagged=True)
    state = _mock_state(eval_result=ev, retries=2)
    mock_store = _mock_store()

    with (
        patch("src.api.routes.query.run_query", new_callable=AsyncMock, return_value=state),
        patch("src.api.routes.query.get_history_store", return_value=mock_store),
    ):
        response = client.post("/query", json={"question": "q?", "evaluate": True})

    body = response.json()
    assert body["eval_result"]["flagged"] is True


# ---------------------------------------------------------------------------
# GET /query/history
# ---------------------------------------------------------------------------


def test_history_returns_saved_items(client):
    record = HistoryRecord(
        id=1,
        timestamp=datetime(2026, 6, 25, 12, 0, 0, tzinfo=UTC),
        question="Capital of France?",
        answer="Paris.",
        sources=["doc.pdf"],
        eval_result=_eval_result(),
        retries=0,
    )
    mock_store = _mock_store(records=[record])

    with patch("src.api.routes.query.get_history_store", return_value=mock_store):
        response = client.get("/query/history")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["question"] == "Capital of France?"
    assert item["eval_result"]["faithfulness"] == pytest.approx(0.85)


def test_history_empty_returns_empty_list(client):
    mock_store = _mock_store(records=[])

    with patch("src.api.routes.query.get_history_store", return_value=mock_store):
        response = client.get("/query/history")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_history_limit_param_forwarded(client):
    mock_store = _mock_store()

    with patch("src.api.routes.query.get_history_store", return_value=mock_store):
        client.get("/query/history?limit=10")

    mock_store.list.assert_awaited_once_with(limit=10)
