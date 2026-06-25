"""Retrieve node: embeds the query and searches Qdrant for the top-k chunks."""

from src.interfaces.embedder import Embedder
from src.interfaces.vector_store import VectorStore


async def retrieve(state: dict, *, embedder: Embedder, store: VectorStore, top_k: int) -> dict:
    vector = await embedder.embed_one(state["query"])
    chunks = await store.search(vector, top_k)
    return {"chunks": chunks}
