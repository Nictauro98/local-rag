from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "chunks"

    ollama_url: str = "http://ollama:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"

    redis_url: str = "redis://redis:6379"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
