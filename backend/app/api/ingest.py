from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from app.core.config import Settings, get_settings
from app.ingestion.chunker import chunk_doc
from app.ingestion.indexer import index_chunks
from app.ingestion.parser import parse_file, parse_url

router = APIRouter()


class IngestResponse(BaseModel):
    source: str
    chunks_indexed: int
    chunk_ids: list[str]


class UrlIngestRequest(BaseModel):
    url: HttpUrl


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        doc = parse_file(data, file.filename or "upload")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    chunks = chunk_doc(doc, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    ids = await index_chunks(chunks, settings)
    return IngestResponse(source=doc.source, chunks_indexed=len(ids), chunk_ids=ids)


@router.post("/url", response_model=IngestResponse)
async def ingest_url(
    body: UrlIngestRequest,
    settings: Annotated[Settings, Depends(get_settings)],
):
    try:
        doc = parse_url(str(body.url))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    chunks = chunk_doc(doc, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text extracted from URL")

    ids = await index_chunks(chunks, settings)
    return IngestResponse(source=doc.source, chunks_indexed=len(ids), chunk_ids=ids)
