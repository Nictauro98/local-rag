"""Unit tests for NLIEvaluator (model mocked — no download required)."""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.nli_evaluator import NLIEvaluator


def _mock_model(entailment_prob: float):
    """Return a mock CrossEncoder that always predicts a fixed entailment probability."""
    model = MagicMock()
    # predict returns list of [contradiction, entailment, neutral] rows
    model.predict.side_effect = lambda pairs, **_: [
        [0.1, entailment_prob, 1.0 - entailment_prob - 0.1] for _ in pairs
    ]
    return model


@pytest.mark.asyncio
async def test_entailed_answer_scores_high():
    with patch("src.evaluation.nli_evaluator._get_model", return_value=_mock_model(0.9)):
        ev = NLIEvaluator()
        result = await ev.evaluate(
            "What is the capital of France?",
            ["Paris is the capital of France."],
            "The capital of France is Paris.",
        )

    assert result.faithfulness >= 0.8
    assert result.answer_grounded is True
    assert result.flagged is False
    assert "nli_faithfulness" in result.tier_scores


@pytest.mark.asyncio
async def test_contradicted_answer_scores_low():
    with patch("src.evaluation.nli_evaluator._get_model", return_value=_mock_model(0.05)):
        ev = NLIEvaluator()
        result = await ev.evaluate(
            "What is the capital of France?",
            ["Paris is the capital of France."],
            "The capital of France is Berlin.",
        )

    assert result.faithfulness < 0.5
    assert result.flagged is True


@pytest.mark.asyncio
async def test_empty_context_returns_zero_and_flagged():
    ev = NLIEvaluator()
    result = await ev.evaluate("anything?", [], "some answer")

    assert result.faithfulness == 0.0
    assert result.flagged is True


@pytest.mark.asyncio
async def test_empty_answer_returns_zero_and_flagged():
    ev = NLIEvaluator()
    result = await ev.evaluate("question?", ["some context"], "")

    assert result.faithfulness == 0.0
    assert result.flagged is True
