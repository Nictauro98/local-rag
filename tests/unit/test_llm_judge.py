"""Unit tests for LLMJudgeEvaluator (Ollama mocked)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.llm_judge import LLMJudgeEvaluator


def _mock_ollama(response_text: str):
    """Return a mock httpx.AsyncClient that replies with response_text."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": response_text}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
async def test_valid_json_parses_correctly():
    payload = json.dumps({"faithfulness": 5, "groundedness": 4, "reasoning": "Well grounded."})
    with patch("src.evaluation.llm_judge.httpx.AsyncClient", return_value=_mock_ollama(payload)):
        ev = LLMJudgeEvaluator(ollama_url="http://fake:11434")
        result = await ev.evaluate("q?", ["some context"], "answer")

    assert result.faithfulness == 1.0  # (5-1)/4
    assert result.context_relevance == 0.75  # (4-1)/4
    assert result.answer_grounded is True
    assert result.flagged is False
    assert result.judge_reasoning == "Well grounded."
    assert "judge_faithfulness" in result.tier_scores


@pytest.mark.asyncio
async def test_low_scores_are_flagged():
    payload = json.dumps({"faithfulness": 1, "groundedness": 1, "reasoning": "Hallucinated."})
    with patch("src.evaluation.llm_judge.httpx.AsyncClient", return_value=_mock_ollama(payload)):
        ev = LLMJudgeEvaluator(ollama_url="http://fake:11434")
        result = await ev.evaluate("q?", ["context"], "bad answer")

    assert result.faithfulness == 0.0
    assert result.flagged is True
    assert result.judge_reasoning == "Hallucinated."


@pytest.mark.asyncio
async def test_malformed_json_returns_flagged():
    with patch(
        "src.evaluation.llm_judge.httpx.AsyncClient", return_value=_mock_ollama("not json at all")
    ):
        ev = LLMJudgeEvaluator(ollama_url="http://fake:11434")
        result = await ev.evaluate("q?", ["ctx"], "answer")

    assert result.flagged is True
    assert result.judge_reasoning == "unparseable judge response"
    assert result.faithfulness == 0.0


@pytest.mark.asyncio
async def test_missing_field_returns_flagged():
    payload = json.dumps({"faithfulness": 4})  # missing groundedness and reasoning
    with patch("src.evaluation.llm_judge.httpx.AsyncClient", return_value=_mock_ollama(payload)):
        ev = LLMJudgeEvaluator(ollama_url="http://fake:11434")
        result = await ev.evaluate("q?", ["ctx"], "answer")

    assert result.flagged is True
    assert result.judge_reasoning == "unparseable judge response"


@pytest.mark.asyncio
async def test_markdown_fenced_json_parses_correctly():
    payload = (
        "```json\n"
        + json.dumps({"faithfulness": 3, "groundedness": 3, "reasoning": "OK."})
        + "\n```"
    )
    with patch("src.evaluation.llm_judge.httpx.AsyncClient", return_value=_mock_ollama(payload)):
        ev = LLMJudgeEvaluator(ollama_url="http://fake:11434")
        result = await ev.evaluate("q?", ["ctx"], "answer")

    assert result.faithfulness == 0.5  # (3-1)/4
    assert result.judge_reasoning == "OK."
