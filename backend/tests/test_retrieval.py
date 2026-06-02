"""
Tests for hybrid retrieval: RRF fusion logic and BM25.
No Qdrant server, no OpenAI — uses FakeQdrant and fixed embeddings.
"""
import pytest
from app.retrieval.hybrid import fuse
from app.retrieval.sparse import bm25_search
from app.ingestion.indexer import _append_bm25_store, BM25_STORE_PATH
from app.ingestion.chunker import Chunk


def _make_chunk(text: str, source: str = "doc.txt", idx: int = 0) -> Chunk:
    return Chunk(text=text, chunk_index=idx, source=source, doc_type="text")


# --- RRF fusion ---

def test_rrf_fuse_deduplicates():
    dense = [{"id": "a", "text": "foo", "source": "x", "score": 0.9}]
    sparse = [{"id": "a", "text": "foo", "source": "x", "score": 5.0}]
    result = fuse(dense, sparse, top_k=5)
    assert len(result) == 1
    assert result[0]["id"] == "a"


def test_rrf_fuse_ordering():
    # doc "b" appears in both lists at rank 0 → should outscore "a" which is only in dense
    dense  = [{"id": "a", "text": "t", "source": "x", "score": 1.0},
              {"id": "b", "text": "t", "source": "x", "score": 0.9}]
    sparse = [{"id": "b", "text": "t", "source": "x", "score": 3.0}]
    result = fuse(dense, sparse, top_k=2)
    assert result[0]["id"] == "b"


def test_rrf_fuse_respects_top_k():
    dense  = [{"id": str(i), "text": "t", "source": "x", "score": float(i)} for i in range(10)]
    sparse = [{"id": str(i), "text": "t", "source": "x", "score": float(i)} for i in range(10)]
    result = fuse(dense, sparse, top_k=3)
    assert len(result) == 3


# --- BM25 ---

@pytest.fixture(autouse=True)
def clean_bm25_store():
    if BM25_STORE_PATH.exists():
        BM25_STORE_PATH.unlink()
    yield
    if BM25_STORE_PATH.exists():
        BM25_STORE_PATH.unlink()


def test_bm25_search_empty_store():
    results = bm25_search("anything", top_k=5)
    assert results == []


def test_bm25_search_finds_relevant():
    chunks = [
        _make_chunk("the quick brown fox jumps", idx=0),
        _make_chunk("machine learning and neural networks", idx=1),
    ]
    _append_bm25_store(chunks, ["id-0", "id-1"])

    results = bm25_search("quick fox", top_k=5)
    assert results[0]["id"] == "id-0"


def test_bm25_search_top_k_respected():
    chunks = [_make_chunk(f"document number {i}", idx=i) for i in range(10)]
    ids = [f"id-{i}" for i in range(10)]
    _append_bm25_store(chunks, ids)

    results = bm25_search("document", top_k=3)
    assert len(results) <= 3
