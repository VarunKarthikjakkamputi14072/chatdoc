"""Dense retrieval via Qdrant cosine similarity."""
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings


async def dense_search(query: str, settings: Settings, client: AsyncQdrantClient) -> list[dict]:
    from app.embeddings import embed_query

    query_vector = await embed_query(query, settings)

    response = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=settings.top_k_dense,
        with_payload=True,
    )

    return [
        {"id": str(r.id), "text": r.payload["text"], "source": r.payload["source"], "score": r.score}
        for r in response.points
    ]
