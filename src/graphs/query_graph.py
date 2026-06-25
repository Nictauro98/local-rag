"""RAG query graph: retrieve → generate → END."""

import functools
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.adapters import get_embedder, get_vector_store
from src.config import get_settings
from src.interfaces.vector_store import RetrievedChunk
from src.nodes.generate import generate
from src.nodes.retrieve import retrieve


class RAGState(TypedDict):
    query: str
    evaluate: bool
    chunks: list[RetrievedChunk]
    answer: str
    eval_result: None  # always None in MVP 1; MVP 2 populates this


def _build_graph(settings, embedder, store):
    g = StateGraph(RAGState)
    g.add_node(
        "retrieve",
        functools.partial(retrieve, embedder=embedder, store=store, top_k=settings.top_k),
    )
    g.add_node(
        "generate",
        functools.partial(generate, llm_url=settings.ollama_url, llm_model=settings.llm_model),
    )
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = None


async def run_query(question: str, evaluate: bool = False) -> RAGState:
    global _graph
    if _graph is None:
        settings = get_settings()
        _graph = _build_graph(settings, get_embedder(), get_vector_store())
    return await _graph.ainvoke(
        {"query": question, "evaluate": evaluate, "chunks": [], "answer": "", "eval_result": None}
    )
