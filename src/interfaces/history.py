"""HistoryStore ABC and HistoryRecord model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel

from src.evaluation.interfaces.evaluator import EvalResult


class HistoryRecord(BaseModel):
    id: int | None = None
    timestamp: datetime
    question: str
    answer: str
    sources: list[str]
    eval_result: EvalResult | None = None
    retries: int = 0


class HistoryStore(ABC):
    @abstractmethod
    async def save(self, record: HistoryRecord) -> None: ...

    @abstractmethod
    async def list(self, limit: int = 50) -> list[HistoryRecord]: ...
