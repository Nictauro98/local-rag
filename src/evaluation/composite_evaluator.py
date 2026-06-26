"""Composite evaluator: runs enabled tiers cheap→expensive, merges results."""

from __future__ import annotations

from src.evaluation.interfaces.evaluator import EvalResult, Evaluator


class CompositeEvaluator(Evaluator):
    def __init__(self, tiers: list[Evaluator]) -> None:
        self._tiers = tiers

    async def evaluate(self, query: str, context: list[str], answer: str) -> EvalResult:
        if not self._tiers:
            return EvalResult(
                faithfulness=1.0,
                context_relevance=1.0,
                answer_grounded=True,
                flagged=False,
                judge_reasoning=None,
                tier_scores={},
            )

        merged_tier_scores: dict[str, float] = {}
        any_flagged = False
        last_result: EvalResult | None = None

        for tier in self._tiers:
            result = await tier.evaluate(query, context, answer)
            merged_tier_scores.update(result.tier_scores)
            if result.flagged:
                any_flagged = True
            last_result = result

        assert last_result is not None
        return EvalResult(
            faithfulness=last_result.faithfulness,
            context_relevance=last_result.context_relevance,
            answer_grounded=last_result.answer_grounded,
            flagged=any_flagged,
            judge_reasoning=last_result.judge_reasoning,
            tier_scores=merged_tier_scores,
        )
