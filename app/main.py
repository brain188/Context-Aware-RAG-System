"""FastAPI app factory: routers, error handlers, middleware, and service lifecycle.

Run with:  uvicorn app.main:app
No business logic here, and nothing runs at import: settings are read and
services built in the lifespan hook.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_logger, get_settings
from app.dependencies import Services, build_services, warm_models
from app.errors import MAPPED_EXCEPTIONS, error_response
from app.middleware import RequestIdMiddleware
from app.models import ErrorResponse
from app.routes import health, ingest, library, query
from app.services.index_version import IndexMismatchError, check_index_embedder

logger = get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _show_details(request: Request) -> bool:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings is not None and settings.environment == "dev"


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'] if p != 'body')}: {err['msg']}"
            for err in exc.errors()
        )
        body = ErrorResponse(
            error="validation_error", detail=detail, request_id=_request_id(request)
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        body = ErrorResponse(
            error="http_error", detail=str(exc.detail), request_id=_request_id(request)
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    async def _mapped(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(request)
        status, body = error_response(exc, rid, _show_details(request))
        logger.log(
            logging.ERROR if status >= 500 else logging.WARNING,
            "Request failed: status=%d %s: %s",
            status,
            type(exc).__name__,
            exc,
            exc_info=status >= 500,
            extra={"request_id": rid or "-"},
        )
        return JSONResponse(status_code=status, content=body.model_dump())

    # Typed errors are handled inside the app; the catch-all Exception handler
    # runs in Starlette's outer error middleware, which also logs the trace.
    for exc_type in (*MAPPED_EXCEPTIONS, Exception):
        app.add_exception_handler(exc_type, _mapped)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the API. Pass *services* (and *settings*) to run against fakes in tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or (services.settings if services else get_settings())
        app.state.settings = resolved
        owned = services is None
        app.state.services = services or build_services(resolved)
        try:
            await check_index_embedder(app.state.services.qdrant, resolved)
        except IndexMismatchError:
            if owned:
                await app.state.services.aclose()
            raise
        if owned and resolved.warm_models_on_startup:
            await warm_models(app.state.services)
        logger.info(
            "API ready: env=%s collection=%s llm=%s:%s fallback=%s prompt=%s",
            resolved.environment,
            resolved.qdrant_collection,
            resolved.llm_provider,
            resolved.llm_generation_model,
            resolved.llm_fallback_provider or "none",
            resolved.prompt_version,
        )
        yield
        if owned:
            await app.state.services.aclose()

    app = FastAPI(title="Personal RAG", version="0.3.0", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    _register_exception_handlers(app)
    for module in (query, ingest, library, health):
        app.include_router(module.router)
    return app


app = create_app()
