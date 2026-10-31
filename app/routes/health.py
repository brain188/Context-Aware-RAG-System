"""GET /health (readiness: dependencies reachable) and GET /health/live (process up).

Status: healthy when everything is reachable; degraded when only Redis is down
(sessions and the cache degrade, answers still work); unhealthy (503) when
Qdrant or the embedding model is unavailable.
"""

import asyncio
from collections.abc import Awaitable

from fastapi import APIRouter, Depends, Response

from app.dependencies import Services, get_services
from app.models import DependencyStatus, HealthResponse

router = APIRouter(tags=["health"])


async def _check(probe: Awaitable[object], timeout: float) -> DependencyStatus:
    """Run *probe*; any failure or timeout is reported, never raised."""
    try:
        await asyncio.wait_for(probe, timeout=timeout)
    except TimeoutError:
        return DependencyStatus(healthy=False, detail=f"no response within {timeout}s")
    except Exception as exc:
        return DependencyStatus(healthy=False, detail=f"{type(exc).__name__}: {exc}")
    return DependencyStatus(healthy=True)


@router.get("/health", response_model=HealthResponse)
async def health(response: Response, services: Services = Depends(get_services)) -> HealthResponse:
    """Check Qdrant (collection exists), Redis (ping), and the embedding model (encodes)."""
    timeout = services.settings.health_timeout_seconds
    loop = asyncio.get_running_loop()
    qdrant, redis, model = await asyncio.gather(
        _check(services.qdrant.get_collection(services.settings.qdrant_collection), timeout),
        _check(services.redis.ping(), timeout),
        _check(loop.run_in_executor(None, services.embedder.embed_query, "health"), timeout),
    )
    if qdrant.healthy and redis.healthy and model.healthy:
        status = "healthy"
    elif qdrant.healthy and model.healthy:
        status = "degraded"
    else:
        status = "unhealthy"
        response.status_code = 503
    return HealthResponse(status=status, qdrant=qdrant, redis=redis, embedding_model=model)


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Liveness: the process is serving requests."""
    return {"status": "alive"}
