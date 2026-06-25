from abc import ABC, abstractmethod

from pydantic import BaseModel


class ChunkPoint(BaseModel):
    id: str
    text: str
    source_filename: str
    chunk_index: int
    vector: list[float]


class RetrievedChunk(BaseModel):
    text: str
    source_filename: str
    chunk_index: int
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self, dim: int) -> None:
        """Create the collection with dim dimensions if it does not exist."""

    @abstractmethod
    async def upsert(self, points: list[ChunkPoint]) -> None:
        """Insert or replace chunk vectors."""

    @abstractmethod
    async def delete_by_source(self, filename: str) -> None:
        """Delete all vectors whose payload source_filename matches filename."""

    @abstractmethod
    async def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks to the query vector."""
