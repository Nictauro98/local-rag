"""Unit tests for retrieve/generate nodes and the query graph."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.nodes.generate import generate
from src.nodes.retrieve import retrieve

# ---------------------------------------------------------------------------
# retrieve node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_searches():
    mock_embedder = AsyncMock()
    mock_embedder.embed_one.return_value = [0.1, 0.2, 0.3]

    mock_chunk = MagicMock()
    mock_store = AsyncMock()
    mock_store.search.return_value = [mock_chunk]

    state = {
        "query": "what is X?",
        "evaluate": False,
        "chunks": [],
        "answer": "",
        "eval_result": None,
    }
    result = await retrieve(state, embedder=mock_embedder, store=mock_store, top_k=3)

    mock_embedder.embed_one.assert_awaited_once_with("what is X?")
    mock_store.search.assert_awaited_once_with([0.1, 0.2, 0.3], 3)
    assert result == {"chunks": [mock_chunk]}


@pytest.mark.asyncio
async def test_retrieve_returns_empty_chunks_when_no_results():
    mock_embedder = AsyncMock()
    mock_embedder.embed_one.return_value = [0.0] * 384
    mock_store = AsyncMock()
    mock_store.search.return_value = []

    state = {
        "query": "unknown topic",
        "evaluate": False,
        "chunks": [],
        "answer": "",
        "eval_result": None,
    }
    result = await retrieve(state, embedder=mock_embedder, store=mock_store, top_k=5)

    assert result == {"chunks": []}


# ---------------------------------------------------------------------------
# generate node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_calls_ollama_and_returns_answer():
    chunk = MagicMock()
    chunk.source_filename = "doc.pdf"
    chunk.text = "The capital of France is Paris."

    state = {
        "query": "What is the capital of France?",
        "chunks": [chunk],
        "evaluate": False,
        "answer": "",
        "eval_result": None,
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Paris"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.nodes.generate.httpx.AsyncClient", return_value=mock_client):
        result = await generate(state, llm_url="http://ollama:11434", llm_model="llama3.2")

    assert result["answer"] == "Paris"
    assert result["eval_result"] is None


@pytest.mark.asyncio
async def test_generate_includes_source_in_prompt():
    chunk = MagicMock()
    chunk.source_filename = "report.pdf"
    chunk.text = "Revenue was $1M."

    state = {
        "query": "What was the revenue?",
        "chunks": [chunk],
        "evaluate": False,
        "answer": "",
        "eval_result": None,
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "$1M"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.nodes.generate.httpx.AsyncClient", return_value=mock_client):
        await generate(state, llm_url="http://ollama:11434", llm_model="llama3.2")

    call_json = mock_client.post.call_args.kwargs["json"]
    assert "report.pdf" in call_json["prompt"]
    assert "Revenue was $1M." in call_json["prompt"]


@pytest.mark.asyncio
async def test_generate_empty_chunks_still_calls_ollama():
    state = {
        "query": "anything?",
        "chunks": [],
        "evaluate": False,
        "answer": "",
        "eval_result": None,
    }

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "I don't have that information."}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.nodes.generate.httpx.AsyncClient", return_value=mock_client):
        result = await generate(state, llm_url="http://ollama:11434", llm_model="llama3.2")

    assert result["answer"] == "I don't have that information."


# ---------------------------------------------------------------------------
# Graph-level: wiring + eval_result is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_routes_retrieve_then_generate_and_eval_result_is_none():
    from src.graphs.query_graph import _build_graph

    mock_settings = MagicMock()
    mock_settings.top_k = 3
    mock_settings.ollama_url = "http://ollama:11434"
    mock_settings.llm_model = "llama3.2"

    async def _mock_retrieve(state, *, embedder, store, top_k):
        return {"chunks": []}

    async def _mock_generate(state, *, llm_url, llm_model):
        return {"answer": "graph answer", "eval_result": None}

    with (
        patch("src.graphs.query_graph.retrieve", _mock_retrieve),
        patch("src.graphs.query_graph.generate", _mock_generate),
    ):
        graph = _build_graph(mock_settings, AsyncMock(), AsyncMock())
        result = await graph.ainvoke(
            {"query": "test?", "evaluate": False, "chunks": [], "answer": "", "eval_result": None}
        )

    assert result["answer"] == "graph answer"
    assert result["eval_result"] is None
