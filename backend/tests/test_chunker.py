"""Tests for the chunker — uses LlamaIndex SentenceSplitter, no network calls."""
from app.ingestion.parser import ParsedDoc, DocType
from app.ingestion.chunker import chunk_doc


def _make_doc(text: str) -> ParsedDoc:
    return ParsedDoc(text=text, source="test.txt", doc_type=DocType.TEXT, metadata={})


def test_chunk_doc_produces_chunks():
    text = " ".join(["word"] * 300)
    doc = _make_doc(text)
    chunks = chunk_doc(doc, chunk_size=64, chunk_overlap=8)
    assert len(chunks) > 1


def test_chunk_doc_short_text_single_chunk():
    doc = _make_doc("Short sentence.")
    chunks = chunk_doc(doc, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "Short sentence."


def test_chunk_metadata_populated():
    doc = _make_doc("Some content here for testing.")
    chunks = chunk_doc(doc, chunk_size=512, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk.source == "test.txt"
        assert chunk.chunk_index == i
        assert chunk.doc_type == "text"


def test_empty_text_returns_no_chunks():
    doc = _make_doc("   ")
    chunks = chunk_doc(doc, chunk_size=128, chunk_overlap=16)
    assert chunks == []
