"""
RAG query engine using LlamaIndex.

We use LlamaIndex's low-level query pipeline rather than its high-level
VectorStoreIndex so our hybrid retriever (RRF fusion) stays in control of
what goes into context — LlamaIndex just handles prompt construction and
the streaming LLM call.
"""
from typing import AsyncIterator

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.retrieval.hybrid import hybrid_search

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions strictly based on the provided document excerpts.
If the answer is not in the excerpts, say you don't know — do not make up information.
Always cite the source document name at the end of your answer in brackets, e.g. [source: report.pdf].
"""

QA_TEMPLATE = PromptTemplate(
    "Context excerpts:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Question: {query}\n"
    "Answer:"
)


def _build_llm(settings: Settings) -> LLM:
    return OpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        system_prompt=SYSTEM_PROMPT,
    )


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(parts)


async def answer_stream(
    query: str,
    settings: Settings,
    qdrant: AsyncQdrantClient,
) -> AsyncIterator[str]:
    chunks = await hybrid_search(query, settings, qdrant)
    context = _format_context(chunks)
    prompt = QA_TEMPLATE.format(context=context, query=query)

    llm = _build_llm(settings)
    stream = await llm.astream_complete(prompt)
    async for delta in stream:
        yield delta.delta


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
        "sources": [{"source": c["source"], "chunk_index": c.get("chunk_index")} for c in chunks],
        "chunks_used": len(chunks),
    }
