# ChatDoc

Chat with your documents. Drop a PDF, DOCX, text file, or URL and ask questions against it.

Built on a hybrid RAG pipeline — BM25 sparse retrieval fused with dense vector search via Reciprocal Rank Fusion — with a full eval harness (recall@k, MRR@k, RAGAS faithfulness) to actually measure how well retrieval and generation are working.

---

## What makes this more than a basic RAG demo

Most document chat demos wire OpenAI + a vector DB and call it done. This one has a few things that matter when you take retrieval seriously:

**Hybrid retrieval (BM25 + dense vectors)**
Pure dense retrieval misses exact-match queries. Pure BM25 misses semantic ones. I combine both with Reciprocal Rank Fusion — each retriever ranks independently, scores are fused with `1 / (k + rank)`. No fine-tuning needed; it consistently outperforms either retriever alone.

**Eval harness that runs on your own data**
After you ingest documents, the eval pipeline auto-generates QA pairs from your chunks (using GPT-4o-mini), then measures:
- **Recall@5** — how often the right chunk appears in the top 5 results
- **MRR@5** — how high up the first relevant chunk is ranked
- **RAGAS faithfulness** — does the generated answer stay grounded in retrieved context?
- **RAGAS answer relevancy** — does the answer actually address the question?

No held-out test set needed. Works on whatever you upload.

**LlamaIndex for orchestration (not raw API calls)**
The query engine uses LlamaIndex's `PromptTemplate` + streaming `astream_complete`, keeping prompt construction and LLM calls cleanly separated from the retrieval logic.

---

## Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + LlamaIndex |
| Vector DB | Qdrant (self-hosted, Docker) |
| Sparse retrieval | rank-bm25 |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini (configurable) |
| Eval | RAGAS + custom recall@k / MRR@k |
| Document parsing | PyMuPDF · python-docx · trafilatura |
| Frontend | React 18 + Vite + TypeScript |

---

## Running locally

```bash
cp backend/.env.example backend/.env
# add your OPENAI_API_KEY to backend/.env

docker compose up
```

Frontend: http://localhost:5173  
API docs: http://localhost:8000/docs

### Dev mode (no Docker)

```bash
# Terminal 1 — Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2 — Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 3 — Frontend
cd frontend
npm install && npm run dev
```

### Tests

```bash
cd backend
pytest
```

Tests run with a fake in-memory Qdrant and no API keys. Covers: parsers, chunker, RRF fusion, BM25 search, recall@k / MRR@k math.

---

## Eval workflow

1. Upload documents via the UI or `POST /api/v1/ingest/file`
2. Generate a QA dataset: click "Generate QA Dataset" in the Eval panel (or `POST /api/v1/eval/generate`)
3. Run retrieval eval: `GET /api/v1/eval/retrieval` → recall@5, MRR@5
4. Run RAGAS eval: `GET /api/v1/eval/ragas` → faithfulness, answer relevancy

The eval dataset persists to `/tmp/chatdoc_eval_dataset.json`. Re-run any time after adding more documents.

---

## Project layout

```
chatdoc/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes (ingest, chat, eval)
│   │   ├── core/         # config, DI
│   │   ├── ingestion/    # parser → chunker → indexer
│   │   ├── retrieval/    # dense (Qdrant) + sparse (BM25) + hybrid RRF
│   │   ├── generation/   # LlamaIndex RAG query engine
│   │   └── eval/         # metrics + harness
│   ├── fake_providers/   # in-memory Qdrant for tests
│   └── tests/
└── frontend/
    └── src/
        ├── components/   # UploadPanel, ChatPanel, EvalPanel
        └── lib/api.ts    # typed API client
```
