# Personal Context-Aware RAG System

A retrieval-augmented generation system over a personal library of ML textbooks.
Answers questions with **verifiable citations** down to book, chapter, section, and page.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Foundations — `app/config.py`, `app/models.py`, logging, `.env.example` | Done |
| 1 | Ingestion pipeline — `pipeline/` | Done |
| 2 | Retrieval core — `app/retrieval/` | Not started |
| 3 | Generation + API — `app/services/`, `app/routes/`, `app/guardrails/` | Not started |
| 4 | Evaluation — `evaluation/` | Not started |
| 5 | Agents (optional) — `app/agents/` | Not started |

## Architecture

```
PDFs → extract → parse structure → preprocess → chunk → dedup → embed → Qdrant + BM25
                                                                            ↓
                            query → hybrid retrieve → rerank → generate → cited answer
```

Ingestion is **offline and deploys independently** of the serving app. `ChunkMetadata`
in `app/models.py` is the frozen contract between the two halves — changing it requires
a full re-ingest.

## Layout

```
app/         FastAPI service: config, schemas, retrieval, generation, routes
pipeline/    Offline ingestion: PDF → structured, deduplicated, embedded chunks
evaluation/  Retrieval + answer quality metrics
infra/       Docker Compose stacks and the environment contract
tests/       Unit tests (pure logic first: chunker, preprocessor, dedup)
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
python -m pipeline.ingest path/to/books/          # a folder or a single PDF
```

Ingestion is idempotent per book: re-running one book replaces its chunks rather than
duplicating them, so new books can be added without reprocessing the library.

## Development

```bash
pytest -q                                          # tests
ruff check .                                       # lint + import order
black --line-length 100 app pipeline tests evaluation
```

CI runs all three on every push and pull request.
