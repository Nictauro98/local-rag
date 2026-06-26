"""Tier-3 LLM-as-judge evaluator: calls Ollama with a distinct judge model."""

from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel, ValidationError

from src.evaluation.interfaces.evaluator import EvalResult, Evaluator

_JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "phi3")
_FAITHFULNESS_THRESHOLD = float(os.getenv("RAG_EVAL_JUDGE_FAITH_THRESHOLD", "0.6"))
_GROUNDEDNESS_THRESHOLD = float(os.getenv("RAG_EVAL_JUDGE_GROUND_THRESHOLD", "0.6"))

_PROMPT_TEMPLATE = """\
You are a strict RAG evaluator. Given a question, retrieved context, and an answer, \
score the answer on two criteria using a 1–5 integer scale:

- faithfulness: does the answer only use information from the context? (5 = fully grounded, 1 = mostly hallucinated)
- groundedness: is the answer directly supported by the context? (5 = fully supported, 1 = not supported)

Respond with ONLY valid JSON in this exact format:
{{"faithfulness": <int 1-5>, "groundedness": <int 1-5>, "reasoning": "<one sentence>"}}

Question: {query}

Context:
{context}

Answer:
{answer}"""


class _JudgeOutput(BaseModel):
    faithfulness: int
    groundedness: int
    reasoning: str


def _normalize(score: int) -> float:
    """Convert 1–5 integer to 0–1 float."""
    return (max(1, min(5, score)) - 1) / 4.0


class LLMJudgeEvaluator(Evaluator):
    def __init__(self, ollama_url: str | None = None) -> None:
        self._ollama_url = ollama_url or os.getenv("RAG_OLLAMA_URL", "http://ollama:11434")

    async def evaluate(self, query: str, context: list[str], answer: str) -> EvalResult:
        prompt = _PROMPT_TEMPLATE.format(
            query=query,
            context="\n\n".join(context),
            answer=answer,
        )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={"model": _JUDGE_MODEL, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                raw = resp.json()["response"].strip()

            # Strip markdown code fences if the model wraps the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = _JudgeOutput.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError, KeyError, httpx.HTTPError):
            return EvalResult(
                faithfulness=0.0,
                context_relevance=0.0,
                answer_grounded=False,
                flagged=True,
                judge_reasoning="unparseable judge response",
                tier_scores={"judge_faithfulness": 0.0, "judge_groundedness": 0.0},
            )

        faithfulness = _normalize(parsed.faithfulness)
        groundedness = _normalize(parsed.groundedness)
        flagged = faithfulness < _FAITHFULNESS_THRESHOLD or groundedness < _GROUNDEDNESS_THRESHOLD

        return EvalResult(
            faithfulness=faithfulness,
            context_relevance=groundedness,
            answer_grounded=groundedness >= _GROUNDEDNESS_THRESHOLD,
            flagged=flagged,
            judge_reasoning=parsed.reasoning,
            tier_scores={
                "judge_faithfulness": faithfulness,
                "judge_groundedness": groundedness,
            },
        )
