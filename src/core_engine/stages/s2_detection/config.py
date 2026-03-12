"""
Stage 2 Detection Configuration

Pydantic config model used by all detection adapters.
Fields cover the superset of settings across YOLO and future adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DetectionConfig(BaseModel):
    """
    Configuration for Stage 2: Detection & Localization.

    Used by all detection adapters (YOLOv8, YOLOv11, Mock, etc.).
    Adapters may ignore fields that are not applicable to their backend.
    """

    # --- Model identity ---
    model_path: Optional[Path] = Field(
        default=None,
        description=(
            "Path to model weights file. "
            "Required by YOLO adapters; ignored by Mock adapter."
        ),
    )
    device: str = Field(
        default="auto",
        description="Inference device: 'cpu', 'cuda', 'mps', or 'auto'.",
    )

    # --- Inference parameters ---
    conf_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to accept a detection.",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for Non-Maximum Suppression.",
    )
    imgsz: int = Field(
        default=640,
        gt=0,
        description="Inference image size (longer dimension in pixels).",
    )

    # --- Detection filtering ---
    min_area_pixels: int = Field(
        default=100,
        ge=0,
        description="Minimum bounding-box area in pixels to keep a detection.",
    )
    max_detections_per_image: int = Field(
        default=50,
        ge=1,
        description="Maximum number of detections kept per input image.",
    )

    # --- Output settings ---
    save_cropped_images: bool = Field(
        default=True,
        description="Persist cropped chart images to disk.",
    )
    save_annotations: bool = Field(
        default=True,
        description="Save detection annotations (bounding boxes + confidence).",
    )
    crop_padding: int = Field(
        default=10,
        ge=0,
        description="Extra pixels added to each side when cropping.",
    )

    # --- Adapter selection ---
    adapter: str = Field(
        default="yolov8",
        description=(
            "Detection backend adapter name. "
            "Registered options: 'yolov8', 'yolov11', 'mock'. "
            "Must match a key in AdapterRegistry for stage 's2_detection'."
        ),
    )
