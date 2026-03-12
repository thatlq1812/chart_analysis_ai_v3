"""
API Request/Response Schemas

Pydantic models for FastAPI endpoints.
All public API shapes are defined here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class JobStatus(str, Enum):
    """Lifecycle states of an analysis job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Request models
# =============================================================================


class AnalyzeOptions(BaseModel):
    """
    Optional parameters for a document analysis request.

    All fields have sensible defaults; callers can omit the entire body.
    """

    pdf_dpi: int = Field(
        default=150,
        ge=72,
        le=300,
        description="DPI for PDF rendering (ignored for image inputs).",
    )
    extract_context: bool = Field(
        default=True,
        description=(
            "Extract surrounding text / figure captions from document sources. "
            "Disable for faster processing when context is not required."
        ),
    )
    stages: Optional[List[str]] = Field(
        default=None,
        description=(
            "Subset of pipeline stages to run. "
            "Defaults to all stages: ['s1', 's2', 's3', 's4', 's5']. "
            "Pass ['s1'] to run ingestion only."
        ),
    )
    output_format: str = Field(
        default="PNG",
        description="Output image format for Stage 1: 'PNG' or 'JPEG'.",
    )


# =============================================================================
# Response models
# =============================================================================


class IngestionPageInfo(BaseModel):
    """Summary of a single page extracted by Stage 1."""

    page_number: int
    image_path: str
    width: int
    height: int
    is_grayscale: bool
    source_format: str
    is_scanned: bool
    has_context: bool = Field(
        description="True when surrounding_text was extracted."
    )
    has_caption: bool = Field(
        description="True when a figure caption was extracted."
    )
    figure_caption: Optional[str] = None
    document_title: Optional[str] = None


class IngestionResult(BaseModel):
    """Structured result from Stage 1 (Ingestion)."""

    session_id: str
    source_file: str
    total_pages: int
    pages: List[IngestionPageInfo]
    warnings: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """
    Complete pipeline analysis result.

    Currently contains Stage 1 result; extended by Stages 2-5 in production.
    """

    ingestion: Optional[IngestionResult] = None
    pipeline_version: str = "3.0.0"
    extra: Dict[str, Any] = Field(default_factory=dict)


class SubmitJobResponse(BaseModel):
    """Response returned immediately after submitting a document for analysis."""

    job_id: str = Field(description="Unique identifier for this analysis job.")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    message: str = Field(default="Job queued for processing.")


class JobStatusResponse(BaseModel):
    """Current status of an analysis job."""

    job_id: str
    status: JobStatus
    progress_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Processing progress percentage (0-100), if available.",
    )
    current_stage: Optional[str] = Field(
        default=None,
        description="Currently executing pipeline stage (e.g., 's1_ingestion').",
    )
    error_message: Optional[str] = None


class JobResultResponse(BaseModel):
    """Final result of a completed analysis job."""

    job_id: str
    status: JobStatus
    result: Optional[AnalysisResult] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None


class HealthStatus(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "3.0.0"
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    """RFC 7807 Problem Details error response."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: Optional[str] = None
