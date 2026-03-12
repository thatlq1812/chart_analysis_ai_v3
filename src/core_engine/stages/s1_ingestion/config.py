"""
Stage 1 Ingestion Configuration

Pydantic configuration model for the ingestion stage.
All settings have production-suitable defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """
    Configuration for Stage 1: Ingestion & Sanitation.

    All fields have production-suitable defaults.
    Override via IngestionConfig(pdf_dpi=200, ...) or pass a dict.
    """

    # --- Rendering ---
    pdf_dpi: int = Field(
        default=150,
        ge=72,
        le=300,
        description="DPI for PDF page rasterization (72=screen, 150=default, 300=print).",
    )

    # --- Image size limits ---
    max_image_size: int = Field(
        default=4096,
        gt=0,
        description="Maximum allowed dimension in pixels; larger images are downscaled.",
    )
    min_image_size: int = Field(
        default=100,
        gt=0,
        description="Minimum allowed dimension in pixels; smaller images are rejected.",
    )

    # --- Quality thresholds ---
    min_blur_threshold: float = Field(
        default=100.0,
        ge=0.0,
        description=(
            "Laplacian variance threshold for blur detection. "
            "Images below this score are flagged (but not rejected)."
        ),
    )

    # --- Output ---
    output_format: str = Field(
        default="PNG",
        description="Output image format written to disk: 'PNG' or 'JPEG'.",
    )
    preserve_color: bool = Field(
        default=True,
        description="Keep original colour channels. False = convert to grayscale.",
    )

    # --- Storage ---
    output_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Root directory for saved images. "
            "Defaults to 'data/processed' when None."
        ),
    )

    # --- Context extraction ---
    extract_context: bool = Field(
        default=True,
        description=(
            "Whether to extract surrounding_text and figure_caption from "
            "document sources (PDF, DOCX, MD). "
            "Disable for faster processing when context is not needed."
        ),
    )
