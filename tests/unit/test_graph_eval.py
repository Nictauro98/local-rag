"""Unit tests for the eval-extended query graph routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.composite_evaluator import CompositeEvaluator
from src.evaluation.interfaces.evaluator import EvalResult, Evaluator

# ---------------------------------------------------------------------------
# Stub evaluators
# ---------------------------------------------------------------------------


class _AlwaysPass(Evaluator):
    async def evaluate(self, query, context, answer):
        return EvalResult(
            faithfulness=0.9,
            context_relevance=0.9,
            answer_grounded=True,
            flagged=False,
            tier_scores={},
        )


class _FlagOnceThenPass(Evaluator):
    """Flags on the first call, passes on subsequent calls."""

    def __init__(self):
        self._calls = 0

    async def evaluate(self, query, context, answer):
        self._calls += 1
        flagged = self._calls == 1
        return EvalResult(
            faithfulness=0.0 if flagged else 0.9,
            context_relevance=0.0 if flagged else 0.9,
            answer_grounded=not flagged,
            flagged=flagged,
            tier_scores={},
        )


class _AlwaysFlag(Evaluator):
    async def evaluate(self, query, context, answer):
        return EvalResult(
            faithfulness=0.0,
            context_relevance=0.0,
            answer_grounded=False,
            flagged=True,
            tier_scores={},
        )


# ---------------------------------------------------------------------------
# Shared mock nodes
# ---------------------------------------------------------------------------


def _mock_settings():
    s = MagicMock()
    s.top_k = 3
    s.ollama_url = "http://ollama:11434"
    s.llm_model = "llama3.2"
    return s


async def _mock_retrieve(state, *, embedder, store, top_k):
    return {"chunks": []}


async def _mock_generate(state, *, llm_url, llm_model):
    return {"answer": "test answer", "eval_result": None}


async def _mock_rewrite(state, *, llm_url, llm_model):
    return {"query": f"{state['query']} (rephrased)", "retries": state.get("retries", 0) + 1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_false_skips_eval_node():
    """With evaluate=False the graph goes retrieve→generate→END; eval_result stays None."""
    from src.graphs.query_graph import _build_graph

    with (
        patch("src.graphs.query_graph.retrieve", _mock_retrieve),
        patch("src.graphs.query_graph.generate", _mock_generate),
        patch("src.graphs.query_graph.rewrite_query", _mock_rewrite),
    ):
        graph = _build_graph(
            _mock_settings(), AsyncMock(), AsyncMock(), CompositeEvaluator([_AlwaysFlag()])
        )
        result = await graph.ainvoke(
            {
                "query": "q?",
                "evaluate": False,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["eval_result"] is None
    assert result["answer"] == "test answer"


@pytest.mark.asyncio
async def test_evaluate_true_pass_goes_to_end():
    """evaluate=True with a passing evaluator: loop runs once, no retry."""
    from src.graphs.query_graph import _build_graph

    with (
        patch("src.graphs.query_graph.retrieve", _mock_retrieve),
        patch("src.graphs.query_graph.generate", _mock_generate),
        patch("src.graphs.query_graph.rewrite_query", _mock_rewrite),
    ):
        graph = _build_graph(
            _mock_settings(), AsyncMock(), AsyncMock(), CompositeEvaluator([_AlwaysPass()])
        )
        result = await graph.ainvoke(
            {
                "query": "q?",
                "evaluate": True,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["eval_result"] is not None
    assert result["eval_result"].flagged is False
    assert result["retries"] == 0


@pytest.mark.asyncio
async def test_single_retry_on_first_flag():
    """Evaluator flags once then passes → exactly one retry, final eval not flagged."""
    from src.graphs.query_graph import _build_graph

    with (
        patch("src.graphs.query_graph.retrieve", _mock_retrieve),
        patch("src.graphs.query_graph.generate", _mock_generate),
        patch("src.graphs.query_graph.rewrite_query", _mock_rewrite),
    ):
        graph = _build_graph(
            _mock_settings(), AsyncMock(), AsyncMock(), CompositeEvaluator([_FlagOnceThenPass()])
        )
        result = await graph.ainvoke(
            {
                "query": "q?",
                "evaluate": True,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["retries"] == 1
    assert result["eval_result"].flagged is False


@pytest.mark.asyncio
async def test_always_flagging_stops_at_max_retries():
    """Always-flagging evaluator: loop stops after max_retries, result is still flagged."""
    from src.graphs.query_graph import _build_graph

    with (
        patch("src.graphs.query_graph.retrieve", _mock_retrieve),
        patch("src.graphs.query_graph.generate", _mock_generate),
        patch("src.graphs.query_graph.rewrite_query", _mock_rewrite),
    ):
        graph = _build_graph(
            _mock_settings(), AsyncMock(), AsyncMock(), CompositeEvaluator([_AlwaysFlag()])
        )
        result = await graph.ainvoke(
            {
                "query": "q?",
                "evaluate": True,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["retries"] == 2  # max_retries reached
    assert result["eval_result"].flagged is True


@pytest.mark.asyncio
async def test_existing_no_eval_path_unchanged():
    """Backward-compat: the old test scenario still works (evaluate=False, eval_result=None)."""
    from src.graphs.query_graph import _build_graph

    with (
        patch("src.graphs.query_graph.retrieve", _mock_retrieve),
        patch("src.graphs.query_graph.generate", _mock_generate),
    ):
        graph = _build_graph(_mock_settings(), AsyncMock(), AsyncMock())
        result = await graph.ainvoke(
            {
                "query": "test?",
                "evaluate": False,
                "chunks": [],
                "answer": "",
                "eval_result": None,
                "retries": 0,
                "max_retries": 2,
            }
        )

    assert result["answer"] == "test answer"
    assert result["eval_result"] is None
