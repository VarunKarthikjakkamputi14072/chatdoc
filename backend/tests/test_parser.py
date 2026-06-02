"""Tests for document parsers — no external services needed."""
import pytest
from app.ingestion.parser import parse_text, parse_file, DocType


def test_parse_plain_text():
    data = b"Hello world. This is a test document."
    doc = parse_text(data, "test.txt")
    assert doc.text == "Hello world. This is a test document."
    assert doc.source == "test.txt"
    assert doc.doc_type == DocType.TEXT


def test_parse_file_routes_to_text_for_md():
    data = b"# Heading\n\nSome content here."
    doc = parse_file(data, "notes.md")
    assert doc.doc_type == DocType.TEXT
    assert "Heading" in doc.text


def test_parse_file_routes_to_pdf():
    """parse_file picks the right parser for .pdf extension."""
    # We just test routing without a real PDF — fitz import happens lazily.
    import pytest
    with pytest.raises(Exception):
        # Passing garbage bytes to a PDF parser will raise — confirms routing happened.
        parse_file(b"not a pdf", "doc.pdf")


def test_parse_utf8_fallback():
    # bytes with a bad utf-8 sequence should not crash
    data = b"Good text \xff\xfe bad bytes"
    doc = parse_text(data, "broken.txt")
    assert "Good text" in doc.text
