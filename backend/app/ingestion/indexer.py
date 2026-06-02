"""
Writes chunks into two indexes:
  1. Qdrant  — dense vectors (OpenAI text-embedding-3-small)
  2. In-process BM25 store (rank-bm25) — rebuilt on each new doc,
     persisted to disk as JSON so it survives restarts.
"""
import json
import os
import uuid
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import Settings
from app.ingestion.chunker import Chunk

# Where the BM25 store + eval dataset live. Defaults to the Docker volume
# mount (/app/data); override with CHATDOC_DATA_DIR for local dev and tests.
DATA_DIR = Path(os.getenv("CHATDOC_DATA_DIR", "/app/data"))
BM25_STORE_PATH = DATA_DIR / "bm25_store.json"


def ensure_data_dir() -> None:
    """Create the data directory on demand (not at import time)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def ensure_collection(settings: Settings) -> None:
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    existing = {c.name for c in (await client.get_collections()).collections}
    if settings.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )


async def embed_chunks(chunks: list[Chunk], settings: Settings) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    texts = [c.text for c in chunks]
    response = await client.embeddings.create(model=settings.embed_model, input=texts)
    return [item.embedding for item in response.data]


async def index_chunks(chunks: list[Chunk], settings: Settings) -> list[str]:
    """Embeds and upserts chunks into Qdrant; appends to BM25 store. Returns chunk IDs."""
    embeddings = await embed_chunks(chunks, settings)

    qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    points = []
    ids = []
    for chunk, vector in zip(chunks, embeddings):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        points.append(
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "doc_type": chunk.doc_type,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
        )

    await qdrant.upsert(collection_name=settings.qdrant_collection, points=points)

    _append_bm25_store(chunks, ids)
    return ids


def _append_bm25_store(chunks: list[Chunk], ids: list[str]) -> None:
    store: list[dict] = []
    if BM25_STORE_PATH.exists():
        store = json.loads(BM25_STORE_PATH.read_text())

    for chunk, chunk_id in zip(chunks, ids):
        store.append({"id": chunk_id, "text": chunk.text, "source": chunk.source})

    ensure_data_dir()
    BM25_STORE_PATH.write_text(json.dumps(store))


def load_bm25_store() -> list[dict]:
    if not BM25_STORE_PATH.exists():
        return []
    return json.loads(BM25_STORE_PATH.read_text())
