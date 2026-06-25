from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from src.adapters import get_embedder, get_vector_store
from src.api.routes import documents, events, query
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    embedder = get_embedder()
    store = get_vector_store()

    # Probe embedding dimension once; creates the Qdrant collection if absent.
    probe = await embedder.embed_one("a")
    await store.ensure_collection(len(probe))

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.redis_pool = redis_pool

    yield

    await redis_pool.aclose()


app = FastAPI(title="Local RAG API", version="0.1.0", lifespan=lifespan)

app.include_router(documents.router)
app.include_router(events.router)
app.include_router(query.router)
