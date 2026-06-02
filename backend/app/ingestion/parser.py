"""
Parse raw bytes / URLs into plain text.
Supports: PDF, DOCX, plain text, Markdown, and web URLs.
"""
import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DocType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    URL = "url"


@dataclass
class ParsedDoc:
    text: str
    source: str          # filename or URL
    doc_type: DocType
    metadata: dict


def _detect_type(filename: str) -> DocType:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return DocType.PDF
    if ext in (".docx", ".doc"):
        return DocType.DOCX
    return DocType.TEXT


def parse_pdf(data: bytes, source: str) -> ParsedDoc:
    import fitz  # pymupdf

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    text = "\n\n".join(pages)
    return ParsedDoc(text=text, source=source, doc_type=DocType.PDF, metadata={"pages": len(doc)})


def parse_docx(data: bytes, source: str) -> ParsedDoc:
    from docx import Document

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    return ParsedDoc(text=text, source=source, doc_type=DocType.DOCX, metadata={})


def parse_text(data: bytes, source: str) -> ParsedDoc:
    text = data.decode("utf-8", errors="replace")
    return ParsedDoc(text=text, source=source, doc_type=DocType.TEXT, metadata={})


def parse_url(url: str) -> ParsedDoc:
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch URL: {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError(f"No extractable text at URL: {url}")
    return ParsedDoc(text=text, source=url, doc_type=DocType.URL, metadata={"url": url})


def parse_file(data: bytes, filename: str) -> ParsedDoc:
    doc_type = _detect_type(filename)
    if doc_type == DocType.PDF:
        return parse_pdf(data, filename)
    if doc_type == DocType.DOCX:
        return parse_docx(data, filename)
    return parse_text(data, filename)
