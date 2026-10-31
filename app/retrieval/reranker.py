"""Cross-encoder reranking of fused retrieval candidates.

The hybrid dense + BM25 stage optimizes recall; the cross-encoder reads query and
passage together for precision. sentence-transformers is imported lazily so
this module stays cheap to import and tests can inject a fake PairScorer.
"""

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Protocol

from app.config import Settings, get_logger
from app.models import ScoredChunk
from app.retrieval.errors import RerankerError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


class PairScorer(Protocol):
    """Anything that scores (query, passage) pairs; higher means more relevant."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderScorer:
    """PairScorer backed by a sentence-transformers CrossEncoder (e.g. bge-reranker)."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.reranker_model
        self._batch_size = settings.rerank_batch_size
        self._model: CrossEncoder | None = None
        # Executor threads may race to load the model on the first concurrent queries.
        self._lock = threading.Lock()

    def _get_model(self) -> "CrossEncoder":
        with self._lock:
            if self._model is None:
                import torch
                from sentence_transformers import CrossEncoder

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading reranker model '%s' on %s.", self._model_name, device)
                self._model = CrossEncoder(self._model_name, device=device)
            return self._model

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score each pair (sigmoid relevance in [0, 1] for bge-reranker).

        CPU/GPU-bound: call from an executor inside async code.
        """
        scores = self._get_model().predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


class Reranker:
    """Reorder candidates by cross-encoder relevance and keep the top k."""

    def __init__(self, scorer: PairScorer, settings: Settings) -> None:
        self._scorer = scorer
        self._top_k = settings.rerank_top_k

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        request_id: str,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return the best *top_k* candidates (default RERANK_TOP_K) rescored by the reranker.

        Returned scores replace the retrieval scores. Ties keep retrieval order.

        Raises:
            RerankerError: if scoring fails or returns the wrong number of scores.
        """
        if not candidates:
            return []
        top_k = top_k or self._top_k
        log_extra = {"request_id": request_id}
        start = time.perf_counter()

        pairs = [(query, c.chunk.text) for c in candidates]
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(None, self._scorer.score, pairs)
        except Exception as exc:
            raise RerankerError(f"Reranker scoring failed: {exc}") from exc
        if len(scores) != len(candidates):
            raise RerankerError(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        results = [ScoredChunk(chunk=c.chunk, score=score) for c, score in ranked]
        logger.info(
            "Rerank: in=%d out=%d top_score=%.4f latency_ms=%.0f",
            len(candidates),
            len(results),
            results[0].score,
            (time.perf_counter() - start) * 1000,
            extra=log_extra,
        )
        return results
