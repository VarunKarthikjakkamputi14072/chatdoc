"""
Splits a ParsedDoc into overlapping text chunks using LlamaIndex's
SentenceSplitter — keeps semantic boundaries at sentence level.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.parser import ParsedDoc


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)


def chunk_doc(doc: "ParsedDoc", chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_text(doc.text)

    return [
        Chunk(
            text=chunk,
            chunk_index=i,
            source=doc.source,
            doc_type=doc.doc_type.value,
            metadata={**doc.metadata, "chunk_index": i, "source": doc.source},
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]
