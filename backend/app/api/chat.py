import json
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.deps import get_qdrant
from app.generation.rag import answer, stream_answer

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int


def _sse(data: dict) -> bytes:
    """Encode a dict as a single Server-Sent Events frame."""
    return f"data: {json.dumps(data)}\n\n".encode()


@router.post("/", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    qdrant=Depends(get_qdrant),
):
    if body.stream:
        async def event_stream() -> AsyncIterator[bytes]:
            # Emit answer tokens as they arrive, then a final frame carrying
            # the source citations (the non-streaming path returns these too),
            # followed by a [DONE] sentinel.
            sources, tokens = await stream_answer(body.query, settings, qdrant)
            async for token in tokens:
                yield _sse({"type": "token", "value": token})
            yield _sse({"type": "sources", "value": sources})
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await answer(body.query, settings, qdrant)
    return ChatResponse(**result)
