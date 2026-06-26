"""Tier-1 deterministic evaluator: cosine similarity + token overlap. No LLM."""

import math
import os
import re

from src.evaluation.interfaces.evaluator import EvalResult, Evaluator
from src.interfaces.embedder import Embedder

_CONTEXT_RELEVANCE_THRESHOLD = float(os.getenv("RAG_EVAL_DET_CONTEXT_THRESHOLD", "0.3"))
_FAITHFULNESS_THRESHOLD = float(os.getenv("RAG_EVAL_DET_FAITH_THRESHOLD", "0.2"))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    mean = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            mean[i] += x
    n = len(vectors)
    return [x / n for x in mean]


def _token_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


class DeterministicEvaluator(Evaluator):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def evaluate(self, query: str, context: list[str], answer: str) -> EvalResult:
        # Embed query and each context chunk
        query_vec = await self._embedder.embed_one(query)
        context_vecs = await self._embedder.embed(context) if context else []

        if context_vecs:
            mean_ctx = _mean_vector(context_vecs)
            context_relevance = (_cosine(query_vec, mean_ctx) + 1.0) / 2.0
        else:
            context_relevance = 0.0

        full_context = " ".join(context)
        faithfulness = _token_overlap(answer, full_context)
        answer_grounded = faithfulness >= _FAITHFULNESS_THRESHOLD

        flagged = (
            context_relevance < _CONTEXT_RELEVANCE_THRESHOLD
            or faithfulness < _FAITHFULNESS_THRESHOLD
        )

        return EvalResult(
            faithfulness=faithfulness,
            context_relevance=context_relevance,
            answer_grounded=answer_grounded,
            flagged=flagged,
            tier_scores={
                "deterministic_context_relevance": context_relevance,
                "deterministic_faithfulness": faithfulness,
            },
        )
