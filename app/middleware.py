"""Request-id correlation and access logging (pure ASGI, so SSE streams pass through untouched)."""

import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RequestIdMiddleware:
    """Accept or mint a request id, expose it as request.state.request_id, echo it in responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        incoming = headers.get(REQUEST_ID_HEADER.lower().encode(), b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = request_id
        log_extra = {"request_id": request_id}
        start = time.perf_counter()
        logger.info("Request: %s %s", scope["method"], scope["path"], extra=log_extra)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
                logger.info(
                    "Response: status=%d latency_ms=%.0f",
                    message["status"],
                    (time.perf_counter() - start) * 1000,
                    extra=log_extra,
                )
            await send(message)

        await self.app(scope, receive, send_with_id)
