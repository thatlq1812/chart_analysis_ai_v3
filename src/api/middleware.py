"""
FastAPI Middleware

Configures CORS, request logging, and global error handling.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .schemas import ErrorDetail

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    """
    Register all middleware and exception handlers on the FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    settings = get_settings()

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request ID + timing ----
    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable,
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            f"Request start | id={request_id} | "
            f"method={request.method} | path={request.url.path}"
        )

        response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Request end | id={request_id} | "
            f"status={response.status_code} | elapsed={elapsed_ms}ms"
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        return response

    # ---- Global exception handler (RFC 7807) ----
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            f"Unhandled exception | id={request_id} | "
            f"path={request.url.path} | error={exc}"
        )
        error = ErrorDetail(
            title="Internal Server Error",
            status=500,
            detail=str(exc) if settings.debug else "An unexpected error occurred.",
            instance=str(request.url),
        )
        return JSONResponse(status_code=500, content=error.model_dump())
