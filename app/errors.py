"""Map internal exceptions to HTTP status codes and the ErrorResponse envelope.

Used by the FastAPI exception handlers and by the SSE stream, which can no
longer change the status code once streaming has started. 5xx details are
shown only in dev; stack traces never reach the client.
"""

from app.guardrails.input_guard import InputRejectedError
from app.llm import LLMError
from app.models import ErrorResponse
from app.prompts import PromptNotFoundError
from app.retrieval.errors import RetrievalError


class IngestRequestError(ValueError):
    """An ingest request names a file that cannot be ingested."""


class IngestUnavailableError(RuntimeError):
    """This deployment does not include the ingestion pipeline."""


class NotFoundError(LookupError):
    """The requested resource does not exist."""


# Most specific first; the first isinstance match wins.
_MAPPING: list[tuple[type[Exception], int, str]] = [
    (InputRejectedError, 400, "input_rejected"),
    (IngestRequestError, 400, "invalid_ingest_request"),
    (NotFoundError, 404, "not_found"),
    (IngestUnavailableError, 501, "ingest_unavailable"),
    (LLMError, 502, "llm_unavailable"),
    (RetrievalError, 503, "retrieval_unavailable"),
    (PromptNotFoundError, 500, "configuration_error"),
]


# Exception types with a defined status; main.py registers a handler for each.
MAPPED_EXCEPTIONS: tuple[type[Exception], ...] = tuple(t for t, _, _ in _MAPPING)


def error_response(
    exc: Exception, request_id: str | None, show_details: bool
) -> tuple[int, ErrorResponse]:
    """Return (status code, body) for *exc*."""
    for exc_type, status, code in _MAPPING:
        if isinstance(exc, exc_type):
            detail = str(exc) if status < 500 or show_details else None
            if isinstance(exc, InputRejectedError):
                code = f"{code}:{exc.reason}"
            return status, ErrorResponse(error=code, detail=detail, request_id=request_id)
    return 500, ErrorResponse(
        error="internal_error",
        detail=f"{type(exc).__name__}: {exc}" if show_details else None,
        request_id=request_id,
    )
