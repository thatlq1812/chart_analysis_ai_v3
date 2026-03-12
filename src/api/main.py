"""
FastAPI Application Factory

Entry point for the Geo-SLM Chart Analysis REST API.

Usage:
    # Development
    uvicorn src.api.main:app --reload --port 8000

    # Production
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

    # Via Makefile
    make serve

API Documentation:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)

Architecture:
    All pipeline processing is dispatched through background tasks.
    In development mode: FastAPI BackgroundTasks (in-process).
    In production: swap for Celery tasks (set CELERY_ENABLED=true).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import get_settings
from .middleware import register_middleware
from .routes import documents_router, health_router, jobs_router

# Configure root logger for the API module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


# =============================================================================
# Lifespan (startup / shutdown)
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Runs startup logic before the server accepts requests
    and shutdown logic when the server is stopping.
    """
    settings = get_settings()

    # --- Startup ---
    settings.setup_directories()
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} | "
        f"host={settings.host} | port={settings.port} | "
        f"debug={settings.debug}"
    )

    yield  # server is running

    # --- Shutdown ---
    logger.info(f"Shutting down {settings.app_name}")


# =============================================================================
# App factory
# =============================================================================


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Geo-SLM Chart Analysis API -- extract structured data from chart images "
            "embedded in PDF, DOCX, Markdown, and image files."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Middleware
    register_middleware(app)

    # Routes
    app.include_router(health_router, prefix=API_PREFIX)
    app.include_router(documents_router, prefix=API_PREFIX)
    app.include_router(jobs_router, prefix=API_PREFIX)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "service": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "health": f"{API_PREFIX}/health",
            }
        )

    logger.info(
        f"API routes registered: "
        f"health={API_PREFIX}/health | "
        f"documents={API_PREFIX}/documents | "
        f"jobs={API_PREFIX}/jobs"
    )

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
