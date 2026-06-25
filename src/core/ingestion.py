import hashlib

from src.core.chunking import chunk
from src.core.parsing import parse
from src.interfaces.embedder import Embedder
from src.interfaces.vector_store import ChunkPoint, VectorStore


def _chunk_id(filename: str, chunk_index: int) -> str:
    """Deterministic UUID-like ID from filename + index."""
    key = f"{filename}:{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


async def ingest(
    filename: str,
    data: bytes,
    *,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Parse → chunk → embed → delete existing → upsert. Returns chunk count."""
    text = parse(filename, data)
    chunks = chunk(text, chunk_size, chunk_overlap)

    if not chunks:
        return 0

    vectors = await embedder.embed(chunks)

    points = [
        ChunkPoint(
            id=_chunk_id(filename, i),
            text=chunks[i],
            source_filename=filename,
            chunk_index=i,
            vector=vectors[i],
        )
        for i in range(len(chunks))
    ]

    # Delete before upsert — idempotent re-ingestion
    await store.delete_by_source(filename)
    await store.upsert(points)

    return len(chunks)
