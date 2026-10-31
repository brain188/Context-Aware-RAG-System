"""POST /api/query (SSE stream) and POST /api/query/sync (single JSON response).

SSE events, in order:
    token   {"token": "...", "request_id": "..."}   zero or more
    answer  Answer                                   final and authoritative
    error   ErrorResponse                            instead of answer, if a stage fails
Input-guard rejections happen before the stream starts and return a plain 400.
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.config import get_logger
from app.dependencies import Services, get_services
from app.errors import error_response
from app.models import ErrorResponse, QueryRequest, QueryResponse, StreamToken
from app.services.rag_pipeline import AnswerEvent, PreparedQuery, RagPipeline, TokenEvent

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["query"])

_ERRORS = {
    400: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


async def _sse_events(
    pipeline: RagPipeline, query: PreparedQuery, show_details: bool
) -> AsyncIterator[dict[str, str]]:
    rid = query.request_id
    try:
        async for event in pipeline.run(query):
            if isinstance(event, TokenEvent):
                token = StreamToken(token=event.text, request_id=rid)
                yield {"event": "token", "data": token.model_dump_json()}
            else:
                yield {"event": "answer", "data": event.answer.model_dump_json()}
    except Exception as exc:
        # Headers (200) are already sent; report the failure as a final event.
        status, body = error_response(exc, rid, show_details)
        logger.error(
            "Query failed mid-stream: status=%d %s: %s",
            status,
            type(exc).__name__,
            exc,
            exc_info=status >= 500,
            extra={"request_id": rid},
        )
        yield {"event": "error", "data": body.model_dump_json()}


@router.post("/query", responses=_ERRORS)
async def query(
    body: QueryRequest, request: Request, services: Services = Depends(get_services)
) -> EventSourceResponse:
    """Answer a question, streaming tokens then the cited answer."""
    prepared = services.pipeline.prepare(body, request.state.request_id)
    show_details = services.settings.environment == "dev"
    return EventSourceResponse(_sse_events(services.pipeline, prepared, show_details))


@router.post("/query/sync", response_model=QueryResponse, responses=_ERRORS)
async def query_sync(
    body: QueryRequest, request: Request, services: Services = Depends(get_services)
) -> QueryResponse:
    """Answer a question in one response (evaluation and simple clients)."""
    prepared = services.pipeline.prepare(body, request.state.request_id)
    final: AnswerEvent | None = None
    async for event in services.pipeline.run(prepared):
        if isinstance(event, AnswerEvent):
            final = event
    if final is None:
        raise RuntimeError("Pipeline finished without an answer")
    return QueryResponse(answer=final.answer, request_id=prepared.request_id)
