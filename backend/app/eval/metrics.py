"""
Retrieval eval metrics: recall@k and MRR@k.

These run offline against a QA dataset where each question has a known
set of relevant chunk IDs (ground truth). The harness generates this
dataset synthetically from ingested documents using an LLM.
"""
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr_at_k: float
    k: int
    num_queries: int


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant result in top-k."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    results: list[dict],  # [{"retrieved_ids": [...], "relevant_ids": [...]}]
    k: int,
) -> RetrievalMetrics:
    if not results:
        return RetrievalMetrics(recall_at_k=0.0, mrr_at_k=0.0, k=k, num_queries=0)

    recalls, mrrs = [], []
    for r in results:
        relevant = set(r["relevant_ids"])
        retrieved = r["retrieved_ids"]
        recalls.append(recall_at_k(retrieved, relevant, k))
        mrrs.append(mrr_at_k(retrieved, relevant, k))

    return RetrievalMetrics(
        recall_at_k=sum(recalls) / len(recalls),
        mrr_at_k=sum(mrrs) / len(mrrs),
        k=k,
        num_queries=len(results),
    )
