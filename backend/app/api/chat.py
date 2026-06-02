from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.deps import get_qdrant
from app.generation.rag import answer, answer_stream

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int


@router.post("/", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    qdrant=Depends(get_qdrant),
):
    if body.stream:
        async def token_stream() -> AsyncIterator[bytes]:
            async for token in answer_stream(body.query, settings, qdrant):
                yield token.encode()

        return StreamingResponse(token_stream(), media_type="text/plain")

    result = await answer(body.query, settings, qdrant)
    return ChatResponse(**result)
