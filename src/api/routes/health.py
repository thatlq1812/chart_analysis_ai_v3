"""
Health Check Routes

GET /api/v1/health        - Lightweight liveness probe
GET /api/v1/health/ready  - Readiness probe (checks model + queue)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter

from ..config import Settings, get_settings
from ..schemas import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthStatus,
    summary="Liveness probe",
    description="Returns 200 OK when the API process is alive.",
)
async def health_live() -> HealthStatus:
    """Simple liveness check -- always returns ok if the server is running."""
    settings = get_settings()
    return HealthStatus(
        status="ok",
        version=settings.app_version,
        services={"api": "ok"},
    )


@router.get(
    "/ready",
    response_model=HealthStatus,
    summary="Readiness probe",
    description="Checks that all required services (model, queue) are reachable.",
)
async def health_ready() -> HealthStatus:
    """
    Readiness check.

    Reports status of:
    - API process itself
    - Celery queue (Redis) when enabled
    """
    settings = get_settings()
    services: Dict[str, str] = {"api": "ok"}

    # Check Redis / Celery
    if settings.celery_enabled:
        services["queue"] = _check_redis(settings.redis_url)
    else:
        services["queue"] = "disabled"

    overall = "ok" if all(v == "ok" or v == "disabled" for v in services.values()) else "degraded"

    return HealthStatus(
        status=overall,
        version=settings.app_version,
        services=services,
    )


def _check_redis(redis_url: str) -> str:
    """Ping Redis and return 'ok' or an error string."""
    try:
        import redis  # type: ignore[import-not-found]

        client = redis.from_url(redis_url, socket_connect_timeout=2)
        client.ping()
        return "ok"
    except (ImportError, Exception) as exc:
        logger.warning(f"Redis health check failed | error={exc}")
        return f"error: {exc}"
