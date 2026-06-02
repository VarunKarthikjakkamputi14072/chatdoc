"""
Reciprocal Rank Fusion (RRF) over dense + sparse result lists.

RRF score = sum(1 / (k + rank_i)) for each list i that contains the doc.
k=60 is the standard constant from the original Cormack et al. paper.
"""
from app.core.config import Settings
from app.retrieval.dense import dense_search
from app.retrieval.sparse import bm25_search
from qdrant_client import AsyncQdrantClient


def _rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank + 1)


def fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank, k)
        docs[doc_id] = item

    for rank, item in enumerate(sparse_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank, k)
        docs[doc_id] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in ranked
    ]


async def hybrid_search(query: str, settings: Settings, qdrant: AsyncQdrantClient) -> list[dict]:
    dense = await dense_search(query, settings, qdrant)
    sparse = bm25_search(query, settings.top_k_sparse)
    return fuse(dense, sparse, top_k=settings.top_k_final, k=settings.rrf_k)
