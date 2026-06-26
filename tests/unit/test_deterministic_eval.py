"""Unit tests for DeterministicEvaluator."""

from unittest.mock import AsyncMock

import pytest

from src.evaluation.deterministic import DeterministicEvaluator


def _make_embedder(query_vec, chunk_vecs):
    """Return a mock embedder with fixed vectors."""
    mock = AsyncMock()
    mock.embed_one.return_value = query_vec
    mock.embed.return_value = chunk_vecs
    return mock


@pytest.mark.asyncio
async def test_high_overlap_answer_scores_high():
    # Query and context pointing in the same direction; answer shares most tokens.
    query_vec = [1.0, 0.0, 0.0]
    chunk_vec = [1.0, 0.0, 0.0]  # cosine=1 → context_relevance=1.0
    context = ["The sky is blue. The ocean is blue."]
    answer = "The sky and ocean are blue."

    embedder = _make_embedder(query_vec, [chunk_vec])
    ev = DeterministicEvaluator(embedder)
    result = await ev.evaluate("What color?", context, answer)

    assert result.context_relevance > 0.5
    assert result.faithfulness > 0.3
    assert result.answer_grounded is True
    assert result.flagged is False


@pytest.mark.asyncio
async def test_off_topic_answer_flagged():
    # Query vector orthogonal to context → low context_relevance.
    query_vec = [1.0, 0.0, 0.0]
    chunk_vec = [0.0, 1.0, 0.0]  # cosine=0 → context_relevance=0.5 (boundary)
    context = ["Photosynthesis converts light into energy."]
    answer = "Quantum mechanics describes particle behavior."

    embedder = _make_embedder(query_vec, [chunk_vec])
    ev = DeterministicEvaluator(embedder)
    result = await ev.evaluate("Explain photosynthesis", context, answer)

    # Answer shares essentially no tokens with context → flagged
    assert result.faithfulness < 0.1
    assert result.flagged is True


@pytest.mark.asyncio
async def test_empty_context_returns_zero_and_flagged():
    embedder = _make_embedder([1.0, 0.0], [])
    ev = DeterministicEvaluator(embedder)
    result = await ev.evaluate("anything?", [], "some answer")

    assert result.context_relevance == 0.0
    assert result.faithfulness == 0.0
    assert result.flagged is True


@pytest.mark.asyncio
async def test_tier_scores_populated():
    embedder = _make_embedder([1.0, 0.0], [[1.0, 0.0]])
    ev = DeterministicEvaluator(embedder)
    result = await ev.evaluate("q", ["ctx text"], "ctx text is here")

    assert "deterministic_context_relevance" in result.tier_scores
    assert "deterministic_faithfulness" in result.tier_scores
