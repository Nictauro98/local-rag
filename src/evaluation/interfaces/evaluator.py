"""Evaluator ABC and shared EvalResult model."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, field_validator


class EvalResult(BaseModel):
    faithfulness: float
    context_relevance: float
    answer_grounded: bool
    flagged: bool
    judge_reasoning: str | None = None
    tier_scores: dict[str, float] = {}
    retries: int = 0

    @field_validator("faithfulness", "context_relevance")
    @classmethod
    def _score_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Score must be in [0, 1], got {v}")
        return v


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(
        self,
        query: str,
        context: list[str],
        answer: str,
    ) -> EvalResult: ...
