"""
Tests for the SSE streaming chat endpoint.

We monkeypatch `stream_answer` so no Qdrant/OpenAI is needed — the point is to
verify the SSE framing: token events, a trailing sources event, and [DONE].
"""
import json
from typing import AsyncIterator


def _parse_sse(body: str) -> list:
    """Parse an SSE body into a list of decoded events ('[DONE]' kept as-is)."""
    events = []
    for frame in body.split("\n\n"):
        data = "".join(
            line[len("data:"):].strip()
            for line in frame.splitlines()
            if line.startswith("data:")
        )
        if not data:
            continue
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_chat_stream_emits_tokens_sources_and_done(client, monkeypatch):
    from app.api import chat as chat_module

    sources = [{"source": "doc.txt", "chunk_index": 0}]

    async def fake_stream_answer(query, settings, qdrant):
        async def tokens() -> AsyncIterator[str]:
            for t in ["Hello", " ", "world"]:
                yield t

        return sources, tokens()

    monkeypatch.setattr(chat_module, "stream_answer", fake_stream_answer)

    res = client.post("/api/v1/chat/", json={"query": "hi", "stream": True})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(res.text)
    tokens = [e["value"] for e in events if isinstance(e, dict) and e["type"] == "token"]
    source_events = [e for e in events if isinstance(e, dict) and e["type"] == "sources"]

    assert "".join(tokens) == "Hello world"
    assert source_events and source_events[0]["value"] == sources
    assert events[-1] == "[DONE]"
