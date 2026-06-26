"""Build a CompositeEvaluator from environment flags."""

from __future__ import annotations

import os

from src.evaluation.composite_evaluator import CompositeEvaluator
from src.evaluation.interfaces.evaluator import Evaluator


def build_composite_evaluator(settings) -> CompositeEvaluator:
    """Instantiate enabled tiers in cheap→expensive order."""
    tiers: list[Evaluator] = []

    if os.getenv("RAG_EVAL_DETERMINISTIC", "true").lower() == "true":
        from src.adapters import get_embedder
        from src.evaluation.deterministic import DeterministicEvaluator

        tiers.append(DeterministicEvaluator(get_embedder()))

    if os.getenv("RAG_EVAL_NLI", "false").lower() == "true":
        from src.evaluation.nli_evaluator import NLIEvaluator

        tiers.append(NLIEvaluator())

    if os.getenv("RAG_EVAL_JUDGE", "false").lower() == "true":
        from src.evaluation.llm_judge import LLMJudgeEvaluator

        tiers.append(LLMJudgeEvaluator(ollama_url=settings.ollama_url))

    return CompositeEvaluator(tiers)
