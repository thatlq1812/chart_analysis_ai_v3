"""
Job Management Routes

GET /api/v1/jobs/{job_id}/status  - Poll job progress
GET /api/v1/jobs/{job_id}/result  - Retrieve completed result
DELETE /api/v1/jobs/{job_id}      - Cancel and delete job record
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..job_store import InMemoryJobStore, get_job_store
from ..schemas import JobResultResponse, JobStatus, JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Poll the current processing status of a submitted job.",
)
async def get_job_status(
    job_id: str,
    job_store: InMemoryJobStore = Depends(get_job_store),
) -> JobStatusResponse:
    """
    Return the current status of a job.

    Args:
        job_id:    Job identifier (returned by /documents/analyze or /documents/ingest).
        job_store: Job state store (injected).

    Returns:
        JobStatusResponse with status and optional progress info.

    Raises:
        HTTPException 404: Job not found or expired.
    """
    status = await job_store.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return status


@router.get(
    "/{job_id}/result",
    response_model=JobResultResponse,
    summary="Get job result",
    description=(
        "Retrieve the final analysis result once a job has completed. "
        "Returns 409 if the job is still processing."
    ),
)
async def get_job_result(
    job_id: str,
    job_store: InMemoryJobStore = Depends(get_job_store),
) -> JobResultResponse:
    """
    Return the final result for a completed job.

    Args:
        job_id:    Job identifier.
        job_store: Job state store (injected).

    Returns:
        JobResultResponse with result payload.

    Raises:
        HTTPException 404: Job not found or expired.
        HTTPException 409: Job has not completed yet.
    """
    record = await job_store.get_result(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if record.status == JobStatus.QUEUED or record.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is still {record.status.value}. Try again later.",
        )

    return record


@router.delete(
    "/{job_id}",
    status_code=204,
    summary="Delete job record",
    description="Remove a job record from the store. Does not cancel in-progress jobs.",
)
async def delete_job(
    job_id: str,
    job_store: InMemoryJobStore = Depends(get_job_store),
) -> None:
    """
    Delete a job record from the in-memory store.

    For in-progress jobs this removes the record but the background task
    continues running until completion (no hard cancellation in this mode).
    Celery-based workers support proper task revocation.

    Args:
        job_id:    Job identifier.
        job_store: Job state store (injected).

    Raises:
        HTTPException 404: Job not found.
    """
    status = await job_store.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    async with job_store._lock:
        job_store._store.pop(job_id, None)

    logger.info(f"Job deleted | job_id={job_id}")
