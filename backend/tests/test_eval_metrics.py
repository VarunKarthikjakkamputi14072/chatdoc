"""Unit tests for recall@k and MRR@k — pure math, zero dependencies."""
from pytest import approx as pytest_approx
from app.eval.metrics import recall_at_k, mrr_at_k, compute_retrieval_metrics


def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_miss():
    assert recall_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_recall_k_cutoff():
    # relevant doc is at rank 3, but k=2 → miss
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_mrr_first_hit():
    assert mrr_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_mrr_second_hit():
    assert mrr_at_k(["x", "a", "c"], {"a"}, k=3) == pytest_approx(0.5)


def test_mrr_miss():
    assert mrr_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_compute_metrics_averages():
    results = [
        {"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]},   # recall=1.0, mrr=1.0
        {"retrieved_ids": ["x", "a"], "relevant_ids": ["a"]},   # recall=1.0, mrr=0.5
    ]
    metrics = compute_retrieval_metrics(results, k=3)
    assert metrics.recall_at_k == 1.0
    assert abs(metrics.mrr_at_k - 0.75) < 1e-6
    assert metrics.num_queries == 2


def test_compute_metrics_empty():
    metrics = compute_retrieval_metrics([], k=5)
    assert metrics.recall_at_k == 0.0
    assert metrics.num_queries == 0
