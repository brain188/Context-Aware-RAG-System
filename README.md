# Personal Context-Aware RAG System

A retrieval-augmented generation system over a personal library of ML textbooks.
Answers questions with **verifiable citations** down to book, chapter, section, and page.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Foundations — `app/config.py`, `app/models.py`, logging, `.env.example` | Done |
| 1 | Ingestion pipeline — `pipeline/` | Done |
| 2 | Retrieval core — `app/retrieval/` | Built; M2 measured by the retrieval eval |
| 3 | Generation + API — `app/services/`, `app/routes/`, `app/guardrails/`, `frontend/` | Built; M3 not yet verified live |
| 4 | Evaluation — `evaluation/` | Built; M4 needs a first live run |
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
evaluation/  Question set, metrics, eval runner, dated reports
infra/       Docker Compose stacks and the environment contract
frontend/    Streamlit UI (talks to the API over HTTP only)
tests/       Unit + integration tests (Qdrant local mode, fakeredis, fake models)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

cp infra/.env.example infra/.env   # then fill in GEMINI_API_KEY and GROQ_API_KEY
```

The LLM layer (`app/llm/`) uses Gemini by default, falls back to Groq (`openai/gpt-oss-120b`)
after retries, and judges evals with Groq; Claude is available as `anthropic`, and
`LLM_PROVIDER=ollama` keeps everything on the machine. Startup fails
loudly if the primary or fallback provider has no API key. Pick model ids from
`python -m app.llm.list_models` and set `LLM_RPM_LIMIT` / `LLM_TPM_LIMIT` to your quota.
See [ADR-0001](docs/adr/0001-gemini-default-provider-abstraction.md).

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

## Evaluation

```bash
python -m evaluation.run_eval --retrieval-only     # retrieve + rerank only: no LLM, free
python -m evaluation.run_eval                      # full pipeline: ~2 LLM calls per question
python -m evaluation.run_eval --judge              # + faithfulness / relevance judge (1 call per answer)
python -m evaluation.run_eval --tag smoke --retrieval-only --fail-on-regression
python -m evaluation.run_eval --retrieval-only --set embedding_query_instruction=   # override any setting
python -m evaluation.compare --latest              # newest report vs the one before, per mode
```

Needs Qdrant (and Redis for full runs) with the books ingested. Eval runs bypass the
semantic cache and log to `data/eval_query_log.jsonl`. Each run writes
`evaluation/reports/<UTC time>_<mode>.json` plus a `.md` summary, and is compared with the
previous report of the same mode over the questions both share.

`evaluation/eval_questions.jsonl` holds one question per line:

```json
{"id": "dj-04", "question": "What is the difference between makemigrations and migrate?",
 "expected_sources": [{"book": "The Ultimate Django Guide", "chapter": "6. Django Models",
                       "page_start": 80, "page_end": 82}],
 "must_mention": ["makemigrations", "apply|applies"], "tags": ["django", "smoke"]}
```

Pages are PDF page numbers (the viewer's counter), not printed page labels. Set
`"answerable": false` (and no sources) for questions the library should decline. The
seed set is drafted from the two indexed books; replace it with questions you really ask.

A chunk is *relevant* when it comes from an expected source: same book (and chapter or
section, if given) with overlapping pages.

| Metric | Meaning |
|--------|---------|
| `hit@1`, `hit@k`, `mrr` | A relevant chunk is ranked first / in the reranked top k / reciprocal rank of the first |
| `precision@k`, `recall@k` | Share of the top k that is relevant / share of expected sources found in it |
| `candidate_recall` | Expected sources found among the fused candidates, before reranking |
| `answered`, `abstained` | Answerable questions answered / unanswerable ones declined |
| `citation_precision`, `citation_hit` | Share of citations on an expected source / at least one is |
| `mention_coverage` | Share of `must_mention` phrases in the answer |
| `faithfulness`, `answer_relevance` | LLM judge scores (with `--judge`) |
| `errored` | The question failed with an error (lower is better) |

CI cannot reach the books or the API, so it gates on committed reports: it fails when the
newest report of a mode is worse than the previous one by more than 0.03 on any metric.

## Development

```bash
pytest -q                                          # tests
ruff check .                                       # lint + import order
black --line-length 100 app pipeline tests evaluation frontend
```

CI runs all three on every push and pull request, plus the eval regression gate.
