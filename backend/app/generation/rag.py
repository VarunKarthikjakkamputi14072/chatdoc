"""
RAG query engine using LlamaIndex.

We use LlamaIndex's low-level query pipeline rather than its high-level
VectorStoreIndex so our hybrid retriever (RRF fusion) stays in control of
what goes into context — LlamaIndex just handles prompt construction and
the streaming LLM call.
"""
from typing import AsyncIterator, Tuple

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.retrieval.hybrid import hybrid_search

SYSTEM_PROMPT = """\
You answer questions using only the provided document excerpts.

Rules:
- Lead with the answer. Be direct and concise — usually 1-3 sentences.
- No preamble, no restating the question, no meta-commentary about the excerpts
  (never say things like "it appears", "based on the context", or "the excerpts show").
- Use a short bulleted list only when enumerating multiple concrete items.
- If the excerpts do not contain the answer, reply exactly: "I don't know based on the provided documents." Do not speculate or add suggestions.
- Never invent facts or sources. Source citations are attached separately, so do not append them yourself.
"""

QA_TEMPLATE = PromptTemplate(
    "Use only these document excerpts to answer.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Question: {query}\n"
    "Answer directly and concisely:"
)


def _build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "groq":
        from llama_index.llms.groq import Groq

        return Groq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            system_prompt=SYSTEM_PROMPT,
            temperature=settings.llm_temperature,
        )
    if settings.llm_provider == "transit":
        # Route chat through Transit (NVIDIA NIM, metered + cached). One af_ key
        # fronts both chat and embeddings; repeated questions hit Transit's cache.
        return OpenAI(
            model=settings.transit_model,
            api_key=settings.transit_api_key,
            api_base=settings.transit_base_url,
            system_prompt=SYSTEM_PROMPT,
            temperature=settings.llm_temperature,
        )
    return OpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        system_prompt=SYSTEM_PROMPT,
        temperature=settings.llm_temperature,
    )


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(parts)


def _format_sources(chunks: list[dict]) -> list[dict]:
    return [{"source": c["source"], "chunk_index": c.get("chunk_index")} for c in chunks]


async def stream_answer(
    query: str,
    settings: Settings,
    qdrant: AsyncQdrantClient,
) -> Tuple[list[dict], AsyncIterator[str]]:
    """
    Retrieves the supporting chunks once, then returns
    (sources, token_iterator). The caller streams the tokens and can emit the
    sources whenever it likes — no shared/global state between requests.
    """
    chunks = await hybrid_search(query, settings, qdrant)
    context = _format_context(chunks)
    prompt = QA_TEMPLATE.format(context=context, query=query)
    llm = _build_llm(settings)

    async def tokens() -> AsyncIterator[str]:
        stream = await llm.astream_complete(prompt)
        async for delta in stream:
            yield delta.delta

    return _format_sources(chunks), tokens()


async def answer(
    query: str,
    settings: Settings,
    qdrant: AsyncQdrantClient,
) -> dict:
    chunks = await hybrid_search(query, settings, qdrant)
    context = _format_context(chunks)
    prompt = QA_TEMPLATE.format(context=context, query=query)

    llm = _build_llm(settings)
    response = await llm.acomplete(prompt)
    return {
        "answer": response.text,
        "sources": _format_sources(chunks),
        "chunks_used": len(chunks),
    }
