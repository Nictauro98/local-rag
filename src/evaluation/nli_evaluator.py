"""Tier-2 NLI evaluator: cross-encoder/nli-deberta-v3-small via sentence-transformers."""

from __future__ import annotations

import asyncio
import os

from src.evaluation.interfaces.evaluator import EvalResult, Evaluator

_NLI_MODEL_NAME = os.getenv("RAG_EVAL_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
_FAITHFULNESS_THRESHOLD = float(os.getenv("RAG_EVAL_NLI_FAITH_THRESHOLD", "0.5"))

# Lazy singleton — loaded once on first use, shared across requests.
_nli_model = None


def _get_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
    return _nli_model


def _split_sentences(text: str) -> list[str]:
    """Minimal sentence splitter on . ! ? boundaries."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _compute_faithfulness_sync(context_chunks: list[str], answer: str) -> float:
    """Run NLI inference synchronously (called in a thread pool)."""
    model = _get_model()
    sentences = _split_sentences(answer)
    if not sentences or not context_chunks:
        return 0.0

    full_context = " ".join(context_chunks)
    pairs = [(full_context, sentence) for sentence in sentences]

    # scores shape: (n_pairs, 3) — columns: contradiction, entailment, neutral
    # (deberta NLI label order)
    scores = model.predict(pairs, apply_softmax=True)

    # entailment is index 1 for deberta-v3 NLI
    entailment_probs = [float(row[1]) for row in scores]
    return sum(entailment_probs) / len(entailment_probs)


class NLIEvaluator(Evaluator):
    async def evaluate(self, query: str, context: list[str], answer: str) -> EvalResult:
        loop = asyncio.get_event_loop()
        faithfulness = await loop.run_in_executor(None, _compute_faithfulness_sync, context, answer)
        faithfulness = max(0.0, min(1.0, faithfulness))
        answer_grounded = faithfulness >= _FAITHFULNESS_THRESHOLD
        flagged = not answer_grounded

        return EvalResult(
            faithfulness=faithfulness,
            context_relevance=faithfulness,  # NLI doesn't separate these; use faithfulness as proxy
            answer_grounded=answer_grounded,
            flagged=flagged,
            tier_scores={"nli_faithfulness": faithfulness},
        )
