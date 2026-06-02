from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "ChatDoc"
    debug: bool = False

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "chatdoc"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5     # after RRF fusion
    rrf_k: int = 60           # RRF constant

    # Eval
    eval_top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
