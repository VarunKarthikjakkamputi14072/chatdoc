"""
In-memory Qdrant stand-in for tests.
Stores vectors and payloads in a plain list; supports cosine similarity search.
No qdrant-client server needed.
"""
import math
from dataclasses import dataclass, field
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class _ScoredPoint:
    id: str
    score: float
    payload: dict


@dataclass
class _QueryResponse:
    points: list


@dataclass
class FakeQdrant:
    _store: list[dict] = field(default_factory=list)

    async def get_collections(self):
        class _Resp:
            collections = []
        return _Resp()

    async def create_collection(self, _collection_name: str, _vectors_config: Any = None):
        pass

    async def upsert(self, _collection_name: str, points: list):
        for p in points:
            self._store.append({
                "id": str(p.id),
                "vector": p.vector,
                "payload": p.payload,
            })

    async def query_points(
        self,
        _collection_name: str,
        query: list[float],
        limit: int = 5,
        with_payload: bool = True,  # noqa: ARG002
    ) -> _QueryResponse:
        scored = [
            _ScoredPoint(
                id=item["id"],
                score=_cosine(query, item["vector"]),
                payload=item["payload"],
            )
            for item in self._store
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return _QueryResponse(points=scored[:limit])
