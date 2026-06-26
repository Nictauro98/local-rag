"""Unit tests for CompositeEvaluator."""

from unittest.mock import AsyncMock

import pytest

from src.evaluation.composite_evaluator import CompositeEvaluator
from src.evaluation.interfaces.evaluator import EvalResult


def _stub(faithfulness=0.8, context_relevance=0.8, flagged=False, tier_key="stub"):
    """Return a mock Evaluator that returns a fixed EvalResult."""
    ev = AsyncMock()
    ev.evaluate.return_value = EvalResult(
        faithfulness=faithfulness,
        context_relevance=context_relevance,
        answer_grounded=not flagged,
        flagged=flagged,
        tier_scores={tier_key: faithfulness},
    )
    return ev


@pytest.mark.asyncio
async def test_all_pass_not_flagged():
    composite = CompositeEvaluator(
        [_stub(flagged=False, tier_key="t1"), _stub(flagged=False, tier_key="t2")]
    )
    result = await composite.evaluate("q", ["ctx"], "ans")

    assert result.flagged is False
    assert "t1" in result.tier_scores
    assert "t2" in result.tier_scores


@pytest.mark.asyncio
async def test_any_tier_flagging_marks_flagged():
    composite = CompositeEvaluator(
        [_stub(flagged=False, tier_key="t1"), _stub(flagged=True, tier_key="t2")]
    )
    result = await composite.evaluate("q", ["ctx"], "ans")

    assert result.flagged is True


@pytest.mark.asyncio
async def test_disabled_tier_omitted_from_scores():
    # Only one tier enabled — other tier key should not appear
    composite = CompositeEvaluator([_stub(flagged=False, tier_key="only_tier")])
    result = await composite.evaluate("q", ["ctx"], "ans")

    assert "only_tier" in result.tier_scores
    assert len(result.tier_scores) == 1


@pytest.mark.asyncio
async def test_empty_tiers_returns_pass():
    composite = CompositeEvaluator([])
    result = await composite.evaluate("q", ["ctx"], "ans")

    assert result.flagged is False
    assert result.faithfulness == 1.0
    assert result.tier_scores == {}


@pytest.mark.asyncio
async def test_tiers_called_in_order():
    call_order: list[str] = []

    async def _make_tier(name: str, flagged: bool):
        from src.evaluation.interfaces.evaluator import Evaluator

        class _T(Evaluator):
            async def evaluate(self, query, context, answer):
                call_order.append(name)
                return EvalResult(
                    faithfulness=0.9,
                    context_relevance=0.9,
                    answer_grounded=True,
                    flagged=flagged,
                    tier_scores={name: 0.9},
                )

        return _T()

    t1 = await _make_tier("deterministic", False)
    t2 = await _make_tier("nli", False)
    t3 = await _make_tier("judge", False)

    composite = CompositeEvaluator([t1, t2, t3])
    await composite.evaluate("q", ["ctx"], "ans")

    assert call_order == ["deterministic", "nli", "judge"]
