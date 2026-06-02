"""BM25 sparse retrieval over the on-disk chunk store."""
from app.ingestion.indexer import load_bm25_store


def bm25_search(query: str, top_k: int) -> list[dict]:
    from rank_bm25 import BM25Okapi

    store = load_bm25_store()
    if not store:
        return []

    query_tokens = query.lower().split()
    query_set = set(query_tokens)
    tokenized_corpus = [doc["text"].lower().split() for doc in store]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    # Keep only chunks that share at least one query term. We can't rely on
    # `score > 0`: BM25 IDF degenerates to 0 on small corpora (e.g. a term in
    # half the docs), which would otherwise drop genuine matches.
    candidates = [
        (float(score), tokens, doc)
        for score, tokens, doc in zip(scores, tokenized_corpus, store)
        if query_set.intersection(tokens)
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)

    return [
        {"id": doc["id"], "text": doc["text"], "source": doc["source"], "score": score}
        for score, _tokens, doc in candidates[:top_k]
    ]
