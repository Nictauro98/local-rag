"""Integration: full LangGraph retry loop with real DeterministicEvaluator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import get_settings
from src.evaluation.composite_evaluator import CompositeEvaluator
from src.evaluation.deterministic import DeterministicEvaluator
from src.graphs.query_graph import _build_graph


def _make_chunk(text: str, source: str = "test.txt") -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.source_filename = source
    return chunk


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def mock_embedder():
    mock = AsyncMock()
    # Aligned vectors → context_relevance stays 1.0; only token overlap drives flagging.
    vec = [1.0, 0.0, 0.0]
    mock.embed_one.return_value = vec
    mock.embed.return_value = [vec]
    return mock


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.search.return_value = [_make_chunk("The capital of France is Paris.")]
    return store


@pytest.mark.integration
async def test_retry_triggers_and_terminates(settings, mock_embedder, mock_store):
    """
    First generate returns an off-topic answer → DeterministicEvaluator flags it.
    rewrite_query runs → second generate returns a grounded answer → passes.
    Final state: retries=1, eval_result.flagged=False.
    """
    evaluator = CompositeEvaluator([DeterministicEvaluator(mock_embedder)])
    call_n = {"generate": 0}

    async def fake_generate(state, *, llm_url, llm_model):
        call_n["generate"] += 1
        if call_n["generate"] == 1:
            return {"answer": "Bananas are yellow tropical fruits grown in warm climates."}
        return {"answer": "The capital of France is Paris."}

    async def fake_rewrite(state, *, llm_url, llm_model):
        return {
            "query": "What city is the capital of France?",
            "retries": state.get("retries", 0) + 1,
        }

    with (
        patch("src.graphs.query_graph.generate", fake_generate),
        patch("src.graphs.query_graph.rewrite_query", fake_rewrite),
    ):
        graph = _build_graph(settings, mock_embedder, mock_store, evaluator)
        result = await graph.ainvoke(
            {
                "query": "What is the capital of France?",
                "evaluate": True,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["retries"] == 1
    assert result["eval_result"] is not None
    assert result["eval_result"].flagged is False
    assert result["answer"] == "The capital of France is Paris."
    assert call_n["generate"] == 2


@pytest.mark.integration
async def test_loop_stops_at_max_retries(settings, mock_embedder, mock_store):
    """Always-failing generate never clears the flag — loop exits at max_retries."""
    evaluator = CompositeEvaluator([DeterministicEvaluator(mock_embedder)])

    async def always_off_topic(state, *, llm_url, llm_model):
        return {"answer": "Bananas are yellow tropical fruits grown in warm climates."}

    async def fake_rewrite(state, *, llm_url, llm_model):
        return {"query": "rephrased?", "retries": state.get("retries", 0) + 1}

    max_retries = 2
    with (
        patch("src.graphs.query_graph.generate", always_off_topic),
        patch("src.graphs.query_graph.rewrite_query", fake_rewrite),
    ):
        graph = _build_graph(settings, mock_embedder, mock_store, evaluator)
        result = await graph.ainvoke(
            {
                "query": "What is the capital of France?",
                "evaluate": True,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": max_retries,
            }
        )

    assert result["retries"] == max_retries
    assert result["eval_result"] is not None
    assert result["eval_result"].flagged is True


@pytest.mark.integration
async def test_evaluate_false_skips_retry_loop(settings, mock_embedder, mock_store):
    """evaluate=False: graph ends at generate with no eval_result and no retries."""
    evaluator = CompositeEvaluator([DeterministicEvaluator(mock_embedder)])

    async def off_topic(state, *, llm_url, llm_model):
        return {"answer": "Bananas are yellow tropical fruits grown in warm climates."}

    with patch("src.graphs.query_graph.generate", off_topic):
        graph = _build_graph(settings, mock_embedder, mock_store, evaluator)
        result = await graph.ainvoke(
            {
                "query": "Capital of France?",
                "evaluate": False,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["eval_result"] is None
    assert result["retries"] == 0
