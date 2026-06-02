from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.deps import get_qdrant
from app.eval.harness import (
    generate_qa_dataset,
    load_eval_dataset,
    run_ragas_eval,
    run_retrieval_eval,
)

router = APIRouter()


class GenerateDatasetResponse(BaseModel):
    num_pairs: int
    message: str


class RetrievalEvalResponse(BaseModel):
    recall_at_k: float
    mrr_at_k: float
    k: int
    num_queries: int


class RagasEvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    num_samples: int


class DatasetStatusResponse(BaseModel):
    num_pairs: int
    has_dataset: bool


@router.post("/generate", response_model=GenerateDatasetResponse)
async def generate_dataset(
    settings: Annotated[Settings, Depends(get_settings)],
    num_pairs: int = Query(default=20, ge=5, le=100),
):
    pairs = await generate_qa_dataset(settings, num_pairs)
    if not pairs:
        raise HTTPException(status_code=422, detail="No chunks indexed yet. Ingest documents first.")
    return GenerateDatasetResponse(num_pairs=len(pairs), message="Dataset generated and saved.")


@router.get("/dataset", response_model=DatasetStatusResponse)
async def dataset_status():
    dataset = load_eval_dataset()
    return DatasetStatusResponse(num_pairs=len(dataset), has_dataset=bool(dataset))


@router.get("/retrieval", response_model=RetrievalEvalResponse)
async def retrieval_eval(
    settings: Annotated[Settings, Depends(get_settings)],
    qdrant=Depends(get_qdrant),
):
    try:
        metrics = await run_retrieval_eval(settings, qdrant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RetrievalEvalResponse(
        recall_at_k=metrics.recall_at_k,
        mrr_at_k=metrics.mrr_at_k,
        k=metrics.k,
        num_queries=metrics.num_queries,
    )


@router.get("/ragas", response_model=RagasEvalResponse)
async def ragas_eval(
    settings: Annotated[Settings, Depends(get_settings)],
    qdrant=Depends(get_qdrant),
):
    try:
        result = await run_ragas_eval(settings, qdrant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RagasEvalResponse(**result)
