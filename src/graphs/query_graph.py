"""RAG query graph: retrieve → generate → [evaluate → rewrite_query →]* END."""

from __future__ import annotations

import functools
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.adapters import get_embedder, get_vector_store
from src.config import get_settings
from src.evaluation.factory import build_composite_evaluator
from src.evaluation.interfaces.evaluator import EvalResult
from src.interfaces.vector_store import RetrievedChunk
from src.nodes.evaluate import evaluate
from src.nodes.generate import generate
from src.nodes.retrieve import retrieve
from src.nodes.rewrite_query import rewrite_query


class RAGState(TypedDict):
    query: str
    evaluate: bool
    chunks: list[RetrievedChunk]
    answer: str
    eval_result: EvalResult | None
    retries: int
    max_retries: int


def _route_after_generate(state: RAGState) -> str:
    return "evaluate" if state.get("evaluate") else END


def _route_after_evaluate(state: RAGState) -> str:
    result = state.get("eval_result")
    if result is None or not result.flagged:
        return END
    if state.get("retries", 0) < state.get("max_retries", 2):
        return "rewrite_query"
    return END


def _build_graph(settings, embedder, store, evaluator=None):
    if evaluator is None:
        from src.evaluation.composite_evaluator import CompositeEvaluator

        evaluator = CompositeEvaluator([])

    g = StateGraph(RAGState)

    g.add_node(
        "retrieve",
        functools.partial(retrieve, embedder=embedder, store=store, top_k=settings.top_k),
    )
    g.add_node(
        "generate",
        functools.partial(generate, llm_url=settings.ollama_url, llm_model=settings.llm_model),
    )
    g.add_node(
        "evaluate",
        functools.partial(evaluate, evaluator=evaluator),
    )
    g.add_node(
        "rewrite_query",
        functools.partial(rewrite_query, llm_url=settings.ollama_url, llm_model=settings.llm_model),
    )

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")

    g.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"evaluate": "evaluate", END: END},
    )
    g.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"rewrite_query": "rewrite_query", END: END},
    )
    g.add_edge("rewrite_query", "retrieve")

    return g.compile()


_graph = None


async def run_query(question: str, evaluate: bool = False, max_retries: int = 2) -> RAGState:
    global _graph
    if _graph is None:
        settings = get_settings()
        _graph = _build_graph(
            settings,
            get_embedder(),
            get_vector_store(),
            build_composite_evaluator(settings),
        )
    return await _graph.ainvoke(
        {
            "query": question,
            "evaluate": evaluate,
            "chunks": [],
            "answer": "",
            "eval_result": None,
            "retries": 0,
            "max_retries": max_retries,
        }
    )
