# Personal Context-Aware RAG System

A retrieval-augmented generation system over a personal library of ML textbooks.
Answers questions with **verifiable citations** down to book, chapter, section, and page.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Foundations — `app/config.py`, `app/models.py`, logging, `.env.example` | Done |
| 1 | Ingestion pipeline — `pipeline/` | Done |
| 2 | Retrieval core — `app/retrieval/` | Built; M2 needs the eval set |
| 3 | Generation + API — `app/services/`, `app/routes/`, `app/guardrails/`, `frontend/` | Built; M3 not yet verified live |
| 4 | Evaluation — `evaluation/` | Not started |
| 5 | Agents (optional) — `app/agents/` | Not started |

## Architecture

```
PDFs → extract → parse structure → preprocess → chunk → dedup → embed → Qdrant (dense + BM25 sparse)
                                                                            ↓
query → input guard → follow-up rewrite → semantic cache → hybrid retrieve → rerank
      → grade (correct / ambiguous / none) → generate → grounding check → cited answer (SSE)
```

Answers come only from retrieved chunks. The model cites sources inline (`[S1]`), and the
output guard holds the stream back until the answer cites a real source. An answer with
no valid citation is replaced with "This is not covered in your library."
Every query writes one record to `data/query_log.jsonl`.

Ingestion is **offline and deploys independently** of the serving app. `ChunkMetadata`
in `app/models.py` is the frozen contract between the two halves — changing it requires
a full re-ingest.

## Layout

```
app/         FastAPI service: config, schemas, retrieval, generation, routes
pipeline/    Offline ingestion: PDF → structured, deduplicated, embedded chunks
evaluation/  Retrieval + answer quality metrics
infra/       Docker Compose stacks and the environment contract
frontend/    Streamlit UI (talks to the API over HTTP only)
tests/       Unit + integration tests (Qdrant local mode, fakeredis, fake models)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

cp infra/.env.example infra/.env   # then fill in LLM_API_KEY
```

`LLM_API_KEY` is the only variable without a default — startup fails loudly if it is missing.

## Running the ingestion pipeline

```bash
python -m pipeline.ingest --path books/                      # every PDF in a folder
python -m pipeline.ingest --path books/bishop.pdf --title "Bishop PRML"   # one book
```

Ingestion is idempotent per book: re-running one book replaces its chunks rather than
duplicating them, so new books can be added without reprocessing the library.

## Running the API and UI

```bash
docker compose -f infra/docker-compose-dev.yml up -d      # Qdrant + Redis
uvicorn app.main:app --reload                              # API on :8000 (docs at /docs)
streamlit run frontend/streamlit_app.py                    # UI on :8501
```

Or run the API in Docker too: `docker compose -f infra/docker-compose-dev.yml --profile app up -d`.

| Endpoint | Purpose |
|----------|---------|
| `POST /api/query` | Ask a question; SSE stream of `token` events, then one `answer` (or `error`) |
| `POST /api/query/sync` | Same, as one JSON response |
| `GET /api/library` | Indexed books and chapters |
| `GET /api/chunks/{id}` | A cited chunk's stored text, for checking a citation |
| `POST /api/ingest`, `GET /api/ingest/{job}` | Ingest a PDF from `books/` in the background |
| `GET /health`, `GET /health/live` | Readiness (Qdrant, Redis, embedding model) and liveness |

## Development

```bash
pytest -q                                          # tests
ruff check .                                       # lint + import order
black --line-length 100 app pipeline tests evaluation frontend
```

CI runs all three on every push and pull request.
