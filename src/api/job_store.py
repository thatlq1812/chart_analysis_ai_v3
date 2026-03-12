"""
In-Memory Job Store

Lightweight job state management when Celery is not configured.
In production, replace with a SQLAlchemy-backed repository.

Thread-safety: uses asyncio.Lock -- suitable for single-process deployments.

Roadmap:
  Phase 3 migration -> swap for JobRepository (SQLAlchemy + PostgreSQL).
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Dict, Optional

from .schemas import AnalysisResult, JobResultResponse, JobStatus, JobStatusResponse


class InMemoryJobStore:
    """
    In-process job registry backed by a plain dict + asyncio.Lock.

    Stores both the status and the final result for every submitted job.
    TTL enforcement is lazy (checked on lookup, not background-swept).
    """

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        # Records: job_id -> {"status": ..., "result": ..., "error": ..., "ts": float, "elapsed_ms": int}
        self._store: Dict[str, dict] = {}

    async def create(self, job_id: str) -> None:
        """Register a new job in QUEUED state."""
        async with self._lock:
            self._store[job_id] = {
                "status": JobStatus.QUEUED,
                "result": None,
                "error": None,
                "ts": time.time(),
                "elapsed_ms": None,
                "current_stage": None,
                "progress_pct": None,
            }

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        current_stage: Optional[str] = None,
        progress_pct: Optional[float] = None,
    ) -> None:
        """Update job status and optional stage/progress info."""
        async with self._lock:
            if job_id in self._store:
                self._store[job_id]["status"] = status
                if current_stage is not None:
                    self._store[job_id]["current_stage"] = current_stage
                if progress_pct is not None:
                    self._store[job_id]["progress_pct"] = progress_pct

    async def complete(
        self,
        job_id: str,
        result: AnalysisResult,
        elapsed_ms: int,
    ) -> None:
        """Mark job COMPLETED and store result."""
        async with self._lock:
            if job_id in self._store:
                self._store[job_id].update(
                    status=JobStatus.COMPLETED,
                    result=result,
                    elapsed_ms=elapsed_ms,
                    progress_pct=100.0,
                    current_stage=None,
                )

    async def fail(self, job_id: str, error_message: str) -> None:
        """Mark job FAILED with an error message."""
        async with self._lock:
            if job_id in self._store:
                self._store[job_id].update(
                    status=JobStatus.FAILED,
                    error=error_message,
                    current_stage=None,
                )

    async def get_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Return current job status.

        Args:
            job_id: Job identifier.

        Returns:
            JobStatusResponse or None if not found / expired.
        """
        async with self._lock:
            record = self._store.get(job_id)
            if record is None:
                return None
            if self._is_expired(record):
                del self._store[job_id]
                return None
            return JobStatusResponse(
                job_id=job_id,
                status=record["status"],
                progress_pct=record["progress_pct"],
                current_stage=record["current_stage"],
                error_message=record["error"],
            )

    async def get_result(self, job_id: str) -> Optional[JobResultResponse]:
        """
        Return the final result for a completed job.

        Args:
            job_id: Job identifier.

        Returns:
            JobResultResponse or None if not found / still processing.
        """
        async with self._lock:
            record = self._store.get(job_id)
            if record is None:
                return None
            if self._is_expired(record):
                del self._store[job_id]
                return None
            return JobResultResponse(
                job_id=job_id,
                status=record["status"],
                result=record["result"],
                error_message=record["error"],
                processing_time_ms=record["elapsed_ms"],
            )

    def _is_expired(self, record: dict) -> bool:
        return time.time() - record["ts"] > self._ttl


@lru_cache(maxsize=1)
def get_job_store() -> InMemoryJobStore:
    """Return the singleton InMemoryJobStore instance."""
    from .config import get_settings
    settings = get_settings()
    return InMemoryJobStore(ttl_seconds=settings.job_ttl_seconds)
