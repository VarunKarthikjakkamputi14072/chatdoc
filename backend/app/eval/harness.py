"""
Eval harness — two phases:

1. QA generation: given ingested chunks, ask an LLM to produce question/answer
   pairs with ground-truth chunk IDs. Stored in /app/data/eval_dataset.json.

2. Eval run:
   a. Retrieval eval — recall@k and MRR@k (our own metrics, no RAGAS needed).
   b. Generation eval — RAGAS faithfulness + answer_relevancy on the RAG answers.
"""
import json

from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.eval.metrics import compute_retrieval_metrics, RetrievalMetrics
from app.ingestion.indexer import load_bm25_store
from app.retrieval.hybrid import hybrid_search

from app.ingestion.indexer import DATA_DIR
EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"


async def generate_qa_dataset(settings: Settings, num_pairs: int = 20) -> list[dict]:
    """
    Synthetically generates QA pairs from stored chunks.
    Each entry: {"question": str, "answer": str, "relevant_ids": [chunk_id, ...]}
    """
    from openai import AsyncOpenAI

    store = load_bm25_store()
    if not store:
        return []

    # Sample up to num_pairs chunks to generate questions from
    import random
    sample = random.sample(store, min(num_pairs, len(store)))

    oai = AsyncOpenAI(api_key=settings.openai_api_key)
    dataset = []

    for item in sample:
        prompt = (
            f"Given this document excerpt, write one factual question whose answer "
            f"is directly contained in the excerpt. Reply with JSON only:\n"
            f"{{\"question\": \"...\", \"answer\": \"...\"}}\n\n"
            f"Excerpt:\n{item['text'][:800]}"
        )
        resp = await oai.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        try:
            qa = json.loads(resp.choices[0].message.content)
            dataset.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "relevant_ids": [item["id"]],
                "context": item["text"],
                "source": item["source"],
            })
        except (KeyError, json.JSONDecodeError):
            continue

    EVAL_DATASET_PATH.write_text(json.dumps(dataset, indent=2))
    return dataset


def load_eval_dataset() -> list[dict]:
    if not EVAL_DATASET_PATH.exists():
        return []
    return json.loads(EVAL_DATASET_PATH.read_text())


async def run_retrieval_eval(settings: Settings, qdrant: AsyncQdrantClient) -> RetrievalMetrics:
    dataset = load_eval_dataset()
    if not dataset:
        raise ValueError("No eval dataset found. Run /eval/generate first.")

    results = []
    for item in dataset:
        chunks = await hybrid_search(item["question"], settings, qdrant)
        retrieved_ids = [c["id"] for c in chunks]
        results.append({"retrieved_ids": retrieved_ids, "relevant_ids": item["relevant_ids"]})

    return compute_retrieval_metrics(results, k=settings.eval_top_k)


async def run_ragas_eval(settings: Settings, qdrant: AsyncQdrantClient) -> dict:
    """
    Runs RAGAS faithfulness + answer_relevancy.
    Returns per-metric averages.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    from app.generation.rag import answer

    dataset_raw = load_eval_dataset()
    if not dataset_raw:
        raise ValueError("No eval dataset found. Run /eval/generate first.")

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in dataset_raw:
        result = await answer(item["question"], settings, qdrant)
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([c["text"] for c in
                                  await hybrid_search(item["question"], settings, qdrant)])
        rows["ground_truth"].append(item["answer"])

    ds = Dataset.from_dict(rows)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "num_samples": len(dataset_raw),
    }
