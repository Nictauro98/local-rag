import httpx

from src.interfaces.embedder import Embedder


class OllamaEmbedder(Embedder):
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim: int | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            return response.json()["embeddings"]

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    def dimension(self) -> int:
        # Dimension is resolved lazily at startup via ensure_collection; callers must
        # await _resolve_dimension() before calling this in production paths.
        if self._dim is None:
            raise RuntimeError(
                "Embedder dimension not yet resolved; call await resolve_dimension() first."
            )
        return self._dim

    async def resolve_dimension(self) -> int:
        """Probe the model with a single token to determine vector size. Cached after first call."""
        if self._dim is None:
            vector = await self.embed_one("a")
            self._dim = len(vector)
        return self._dim
