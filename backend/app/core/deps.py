from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings, get_settings


@lru_cache
def _qdrant_client(host: str, port: int) -> AsyncQdrantClient:
    return AsyncQdrantClient(host=host, port=port)


def get_qdrant(settings: Annotated[Settings, Depends(get_settings)]) -> AsyncQdrantClient:
    return _qdrant_client(settings.qdrant_host, settings.qdrant_port)
