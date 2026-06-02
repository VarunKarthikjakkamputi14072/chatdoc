"""Dense retrieval via Qdrant cosine similarity."""
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings


async def dense_search(query: str, settings: Settings, client: AsyncQdrantClient) -> list[dict]:
    from openai import AsyncOpenAI

    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await oai.embeddings.create(model=settings.embed_model, input=[query])
    query_vector = resp.data[0].embedding

    results = await client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=settings.top_k_dense,
        with_payload=True,
    )

    return [
        {"id": str(r.id), "text": r.payload["text"], "source": r.payload["source"], "score": r.score}
        for r in results
    ]
