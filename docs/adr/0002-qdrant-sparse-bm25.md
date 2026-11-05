# ADR-0002: Keyword search as BM25 sparse vectors inside Qdrant

- **Status:** accepted
- **Date:** 2026-11-03

## Context

Hybrid retrieval needs a keyword side next to the dense embeddings. Technical
questions often hinge on exact terms that embeddings blur: "KL divergence",
`ModelForm`, equation labels, function names. The first plan used `rank-bm25`, an
in-memory index in the API process. That has three problems (`PROJECT_BRIEF.md` §12.1):

- it is rebuilt from every chunk on each restart, and memory is the scarcest
  resource on the laptop (R7);
- it cannot be updated for one book (FR2) without rebuilding everything;
- it does not see the Qdrant metadata filters, so book, chapter and page filters
  would apply to the dense side only.

## Options

1. **`rank-bm25` in memory**: simplest code.
   - Against: all three problems above.
2. **OpenSearch / Elasticsearch next to Qdrant**: mature BM25.
   - Against: a second store to run, sync and filter consistently. Too heavy for the laptop.
3. **Qdrant sparse vectors with the IDF modifier, built with fastembed's `Qdrant/bm25`.**
   - For: stemming and stopwords included.
   - Against: an extra dependency and model download in both app and pipeline.
4. **Qdrant sparse vectors with the IDF modifier, from our own small encoder**
   (`app/retrieval/sparse_encoder.py`).

## Decision

Option 4. At ingest, each chunk stores the BM25 term-frequency half (k1, b, a fixed
average length) on a named sparse vector `bm25` whose collection config sets
`Modifier.IDF`. Qdrant computes the IDF half at query time over the live corpus, so
adding a book updates IDF with no rebuild. A query sends weight 1.0 per distinct term.
`HybridRetriever` makes **one** Query API call with a dense and a sparse prefetch,
both scoped by the same filter, and fuses them with RRF on the server.

The encoder is about 60 lines with no dependencies: lowercase `\w+` tokens hashed
with CRC32 into a 32-bit space. Pipeline and app import the same module, so both
sides tokenize the same way by construction.

## Consequences

- No index rebuild on restart and no keyword index in API memory. Per-book re-ingest
  works, and filters apply to both retrieval branches.
- Tokenization is a contract: changing `tokenize`, the hashing or the BM25 settings
  (`BM25_K1`, `BM25_B`, `BM25_AVG_DOC_LENGTH`) requires re-ingesting every book. These
  settings are part of the index fingerprint, so the semantic cache is invalidated
  automatically.
- No stemming or stopword removal, so "divergences" does not match "divergence" on the
  keyword side. The dense side usually covers it. If the eval shows keyword misses,
  switching to fastembed BM25 is a contained change (encoder + re-ingest), measured
  through the ablation in TASKLIST 1.11.
- `BM25_AVG_DOC_LENGTH` is a fixed setting, not the true corpus average. This
  approximation only shifts length normalization slightly; chunk sizes are bounded
  by the chunker anyway.
- `rank-bm25` is no longer a dependency.
