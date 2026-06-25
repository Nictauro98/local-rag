from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.interfaces.vector_store import ChunkPoint, RetrievedChunk, VectorStore


class QdrantStore(VectorStore):
    def __init__(self, url: str, collection: str):
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection

    async def ensure_collection(self, dim: int) -> None:
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def upsert(self, points: list[ChunkPoint]) -> None:
        qdrant_points = [
            PointStruct(
                id=p.id,
                vector=p.vector,
                payload={
                    "text": p.text,
                    "source_filename": p.source_filename,
                    "chunk_index": p.chunk_index,
                },
            )
            for p in points
        ]
        await self._client.upsert(collection_name=self._collection, points=qdrant_points)

    async def delete_by_source(self, filename: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_filename", match=MatchValue(value=filename))]
            ),
        )

    async def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                text=r.payload["text"],
                source_filename=r.payload["source_filename"],
                chunk_index=r.payload["chunk_index"],
                score=r.score,
            )
            for r in response.points
        ]
