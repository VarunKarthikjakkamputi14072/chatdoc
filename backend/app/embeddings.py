"""
Pluggable embedding backends.

- "openai": OpenAI embeddings (text-embedding-3-small, 1536-dim) — default.
- "local":  fastembed (ONNX, BAAI/bge-small-en-v1.5, 384-dim) — no API key,
            runs offline. Used for keyless local demos.

Both ingestion (indexer) and query-time dense retrieval go through here so the
two sides can never drift onto different models.
"""
from functools import lru_cache

from app.core.config import Settings


@lru_cache(maxsize=2)
def _fastembed(model_name: str):
    # Imported lazily so the dependency is only needed when embed_provider=local.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


async def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    if settings.embed_provider == "local":
        model = _fastembed(settings.local_embed_model)
        # fastembed is synchronous and returns numpy arrays.
        return [[float(x) for x in vec] for vec in model.embed(list(texts))]

    from openai import AsyncOpenAI

    if settings.embed_provider == "transit":
        # Route through Transit (NVIDIA NIM embeddings, content-hash cached) —
        # re-embedding identical chunks costs nothing on a cache hit.
        client = AsyncOpenAI(
            api_key=settings.transit_api_key, base_url=settings.transit_base_url
        )
        resp = await client.embeddings.create(
            model=settings.transit_embed_model, input=list(texts)
        )
        return [item.embedding for item in resp.data]

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.embeddings.create(model=settings.embed_model, input=list(texts))
    return [item.embedding for item in resp.data]


async def embed_query(query: str, settings: Settings) -> list[float]:
    return (await embed_texts([query], settings))[0]
