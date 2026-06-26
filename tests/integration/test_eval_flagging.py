"""Integration: real DeterministicEvaluator through CompositeEvaluator — misleading context → flagged."""

import pytest
from unittest.mock import AsyncMock

from src.evaluation.composite_evaluator import CompositeEvaluator
from src.evaluation.deterministic import DeterministicEvaluator


@pytest.fixture
def mock_embedder():
    mock = AsyncMock()
    # Aligned vectors so context_relevance stays high (1.0); only token overlap drives flagging.
    vec = [1.0, 0.0, 0.0]
    mock.embed_one.return_value = vec
    mock.embed.return_value = [vec]
    return mock


@pytest.fixture
def evaluator(mock_embedder):
    return CompositeEvaluator([DeterministicEvaluator(mock_embedder)])


@pytest.mark.integration
async def test_off_topic_answer_is_flagged(evaluator):
    """Answer shares no tokens with context → faithfulness=0 → flagged."""
    context = ["The capital of France is Paris, a city on the Seine river."]
    answer = "Bananas are yellow tropical fruits cultivated in warm humid climates."

    result = await evaluator.evaluate("What is the capital of France?", context, answer)

    assert result.flagged is True
    assert result.faithfulness == pytest.approx(0.0)


@pytest.mark.integration
async def test_grounded_answer_passes(evaluator):
    """Answer that repeats context tokens → high faithfulness → not flagged."""
    context = ["The capital of France is Paris, a city on the Seine river."]
    answer = "The capital of France is Paris."

    result = await evaluator.evaluate("What is the capital of France?", context, answer)

    assert result.flagged is False
    assert result.faithfulness > 0.5


@pytest.mark.integration
async def test_flagged_state_propagates_through_composite(evaluator):
    """Tier flagging is surfaced in composite result with per-tier scores present."""
    context = ["Qdrant is a vector database written in Rust."]
    answer = "The moon is made of cheese and orbits Neptune."

    result = await evaluator.evaluate("What is Qdrant?", context, answer)

    assert result.flagged is True
    assert "deterministic_faithfulness" in result.tier_scores
    assert "deterministic_context_relevance" in result.tier_scores
