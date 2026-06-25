from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    @abstractmethod
    async def embed_one(self, text: str) -> list[float]:
        """Return a single embedding vector for text."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the length of the embedding vectors produced by this model."""
