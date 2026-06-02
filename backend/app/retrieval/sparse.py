"""BM25 sparse retrieval over the on-disk chunk store."""
from app.ingestion.indexer import load_bm25_store


def bm25_search(query: str, top_k: int) -> list[dict]:
    from rank_bm25 import BM25Okapi

    store = load_bm25_store()
    if not store:
        return []

    tokenized_corpus = [doc["text"].lower().split() for doc in store]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(zip(scores, store), key=lambda x: x[0], reverse=True)[:top_k]
    return [
        {"id": doc["id"], "text": doc["text"], "source": doc["source"], "score": float(score)}
        for score, doc in ranked
        if score > 0
    ]
