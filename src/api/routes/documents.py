"""
Document Analysis Routes

POST /api/v1/documents/analyze   - Submit document for full pipeline analysis
POST /api/v1/documents/ingest    - Run Stage 1 only (ingestion preview)

File uploads are accepted as multipart/form-data.
Supported formats: PDF, DOCX, MD, PNG, JPG, JPEG, WebP, TIFF, BMP.

Processing is dispatched to a background task to keep the API non-blocking.
In production, swap BackgroundTasks for Celery (set CELERY_ENABLED=true).
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..config import Settings, get_settings
from ..job_store import InMemoryJobStore, get_job_store
from ..schemas import (
    AnalysisResult,
    AnalyzeOptions,
    IngestionPageInfo,
    IngestionResult,
    JobStatus,
    SubmitJobResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Accepted MIME types and extensions
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".md",
    ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp",
}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/markdown",
    "text/plain",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
    "image/bmp",
}


@router.post(
    "/analyze",
    response_model=SubmitJobResponse,
    status_code=202,
    summary="Submit document for full pipeline analysis",
    description=(
        "Upload a document (PDF, DOCX, MD, or image) for the complete 5-stage "
        "chart analysis pipeline. Returns a job_id immediately; poll "
        "/api/v1/jobs/{job_id}/status for progress."
    ),
)
async def analyze_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document to analyze."),
    pdf_dpi: int = 150,
    extract_context: bool = True,
    settings: Settings = Depends(get_settings),
    job_store: InMemoryJobStore = Depends(get_job_store),
) -> SubmitJobResponse:
    """
    Accept a file upload and queue the full pipeline for async execution.

    Args:
        file:            Uploaded file (multipart/form-data).
        pdf_dpi:         DPI for PDF rendering (passed to Stage 1).
        extract_context: Whether to extract surrounding text from documents.
        settings:        Application settings (injected).
        job_store:       Job state store (injected).

    Returns:
        SubmitJobResponse with job_id and QUEUED status.
    """
    options = AnalyzeOptions(pdf_dpi=pdf_dpi, extract_context=extract_context)
    saved_path = await _save_upload(file, settings)
    job_id = _new_job_id()

    await job_store.create(job_id)

    background_tasks.add_task(
        _run_full_pipeline,
        job_id=job_id,
        file_path=saved_path,
        options=options,
        job_store=job_store,
    )

    logger.info(
        f"Job queued | job_id={job_id} | file={file.filename} | "
        f"pipeline=full"
    )

    return SubmitJobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message=f"Job {job_id} queued. Poll /api/v1/jobs/{job_id}/status for progress.",
    )


@router.post(
    "/ingest",
    response_model=SubmitJobResponse,
    status_code=202,
    summary="Run Stage 1 ingestion only",
    description=(
        "Extract and normalize images from a document without running "
        "chart detection or analysis. Useful for previewing parsed pages "
        "and validating document context extraction."
    ),
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document to ingest."),
    pdf_dpi: int = 150,
    extract_context: bool = True,
    settings: Settings = Depends(get_settings),
    job_store: InMemoryJobStore = Depends(get_job_store),
) -> SubmitJobResponse:
    """
    Accept a file upload and run Stage 1 (ingestion) only.

    Returns extracted page images and document context without running YOLO
    detection or downstream stages.
    """
    options = AnalyzeOptions(
        pdf_dpi=pdf_dpi,
        extract_context=extract_context,
        stages=["s1"],
    )
    saved_path = await _save_upload(file, settings)
    job_id = _new_job_id()

    await job_store.create(job_id)

    background_tasks.add_task(
        _run_ingestion_only,
        job_id=job_id,
        file_path=saved_path,
        options=options,
        job_store=job_store,
    )

    logger.info(
        f"Job queued | job_id={job_id} | file={file.filename} | "
        f"pipeline=s1_only"
    )

    return SubmitJobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message=f"Ingestion job {job_id} queued. Poll /api/v1/jobs/{job_id}/status for result.",
    )


# =============================================================================
# Background task runners
# =============================================================================


async def _run_ingestion_only(
    job_id: str,
    file_path: Path,
    options: AnalyzeOptions,
    job_store: InMemoryJobStore,
) -> None:
    """Background task: run Stage 1 ingestion and store result."""
    start_ms = int(time.time() * 1000)

    try:
        await job_store.update_status(
            job_id, JobStatus.PROCESSING, current_stage="s1_ingestion", progress_pct=0.0
        )

        from src.core_engine.stages.s1_ingestion import IngestionConfig, Stage1Ingestion

        config = IngestionConfig(
            pdf_dpi=options.pdf_dpi,
            extract_context=options.extract_context,
        )
        stage = Stage1Ingestion(config)
        output = stage.process(file_path)

        ingestion_result = _build_ingestion_result(output, file_path)
        analysis_result = AnalysisResult(ingestion=ingestion_result)

        elapsed = int(time.time() * 1000) - start_ms
        await job_store.complete(job_id, analysis_result, elapsed_ms=elapsed)

        logger.info(
            f"Ingestion job done | job_id={job_id} | "
            f"pages={output.total_images} | elapsed={elapsed}ms"
        )

    except Exception as exc:
        logger.exception(f"Ingestion job failed | job_id={job_id} | error={exc}")
        await job_store.fail(job_id, error_message=str(exc))
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink(missing_ok=True)


async def _run_full_pipeline(
    job_id: str,
    file_path: Path,
    options: AnalyzeOptions,
    job_store: InMemoryJobStore,
) -> None:
    """
    Background task: run the full 5-stage pipeline.

    Currently wired to run Stage 1 only; Stages 2-5 will be integrated
    as the serving module matures (Phase 3 roadmap).
    """
    start_ms = int(time.time() * 1000)

    try:
        # Stage 1
        await job_store.update_status(
            job_id, JobStatus.PROCESSING, current_stage="s1_ingestion", progress_pct=10.0
        )
        from src.core_engine.stages.s1_ingestion import IngestionConfig, Stage1Ingestion

        config = IngestionConfig(
            pdf_dpi=options.pdf_dpi,
            extract_context=options.extract_context,
        )
        stage1 = Stage1Ingestion(config)
        s1_output = stage1.process(file_path)
        ingestion_result = _build_ingestion_result(s1_output, file_path)

        await job_store.update_status(
            job_id, JobStatus.PROCESSING, current_stage="s2_detection", progress_pct=25.0
        )

        # Stages 2-5: placeholder until full pipeline wired
        # TODO: wire Stage2Detection, Stage3Extraction, Stage4Reasoning, Stage5Reporting

        analysis_result = AnalysisResult(
            ingestion=ingestion_result,
            extra={"note": "Stages 2-5 pending Phase 3 integration."},
        )

        elapsed = int(time.time() * 1000) - start_ms
        await job_store.complete(job_id, analysis_result, elapsed_ms=elapsed)

        logger.info(
            f"Pipeline job done | job_id={job_id} | "
            f"s1_pages={s1_output.total_images} | elapsed={elapsed}ms"
        )

    except Exception as exc:
        logger.exception(f"Pipeline job failed | job_id={job_id} | error={exc}")
        await job_store.fail(job_id, error_message=str(exc))
    finally:
        if file_path.exists():
            file_path.unlink(missing_ok=True)


# =============================================================================
# Helpers
# =============================================================================


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


async def _save_upload(file: UploadFile, settings: Settings) -> Path:
    """
    Validate and persist the uploaded file to the uploads directory.

    Args:
        file:     FastAPI UploadFile object.
        settings: App settings for upload_dir and size limits.

    Returns:
        Path to the saved file.

    Raises:
        HTTPException 400: Unsupported file format.
        HTTPException 413: File exceeds size limit.
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"

    size_bytes = 0
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    with dest.open("wb") as out:
        while chunk := await file.read(8192):
            size_bytes += len(chunk)
            if size_bytes > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File exceeds maximum upload size of "
                        f"{settings.max_upload_size_mb} MB."
                    ),
                )
            out.write(chunk)

    logger.debug(
        f"Upload saved | filename={filename} | "
        f"size={size_bytes // 1024}KB | dest={dest.name}"
    )
    return dest


def _build_ingestion_result(output: "Stage1Output", source: Path) -> IngestionResult:  # type: ignore[name-defined]
    """Map Stage1Output -> IngestionResult API schema."""
    pages = [
        IngestionPageInfo(
            page_number=img.page_number,
            image_path=str(img.image_path),
            width=img.width,
            height=img.height,
            is_grayscale=img.is_grayscale,
            source_format=img.source_format,
            is_scanned=img.is_scanned,
            has_context=img.surrounding_text is not None,
            has_caption=img.figure_caption is not None,
            figure_caption=img.figure_caption,
            document_title=img.document_title,
        )
        for img in output.images
    ]
    return IngestionResult(
        session_id=output.session.session_id,
        source_file=source.name,
        total_pages=output.total_images,
        pages=pages,
        warnings=output.warnings,
    )
