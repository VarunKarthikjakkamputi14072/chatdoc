from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ingest, chat, eval as eval_router
from app.core.config import get_settings
from app.ingestion.indexer import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await ensure_collection(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Document Q&A with hybrid RAG (BM25 + dense) and RAGAS eval harness",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(eval_router.router, prefix="/api/v1/eval", tags=["eval"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
