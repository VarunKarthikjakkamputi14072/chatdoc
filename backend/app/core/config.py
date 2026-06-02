from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "ChatDoc"
    debug: bool = False

    # --- Providers (pluggable) ---
    # LLM: "openai" (default) | "groq"
    llm_provider: str = "openai"
    # Embeddings: "openai" (default) | "local" (fastembed, no API key)
    embed_provider: str = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # Generation temperature (low = grounded, terse answers)
    llm_temperature: float = 0.1

    # Groq (OpenAI-compatible; LLM inference only — no embeddings)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Local embeddings (fastembed / ONNX, runs offline)
    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    local_embed_dim: int = 384

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

    def vector_size(self) -> int:
        """Embedding dimensionality for the active embed provider."""
        return self.local_embed_dim if self.embed_provider == "local" else self.embed_dim


@lru_cache
def get_settings() -> Settings:
    return Settings()
