"""
Base Detection Adapter

ABC that all Stage 2 adapters must implement.
Defines the data contract between Stage 2 and its concrete backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ....schemas.common import BoundingBox


@dataclass
class RawDetection:
    """
    Single detection result from a backend model.

    Returned by `BaseDetectionAdapter.detect()` and consumed by
    `Stage2Detection` for post-processing, cropping, and persistence.
    """

    bbox: BoundingBox
    """Bounding box in pixel coordinates of the source image."""

    label: str = "chart"
    """Detected class label (most detectors use a single 'chart' class)."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Adapter-specific metadata (e.g. class probabilities, track ID)."""


class BaseDetectionAdapter(ABC):
    """
    Abstract base class for all Stage 2 detection backends.

    Subclasses:
        YOLOv8Adapter  - Ultralytics YOLOv8 (default)
        YOLOv11Adapter - Ultralytics YOLO11 (new architecture)
        MockAdapter    - Deterministic mock for unit testing

    Contract:
        - `detect()` receives a BGR numpy array and returns `List[RawDetection]`.
        - The adapter is responsible for loading its own model in `__init__`.
        - Adapters must NOT perform cropping or I/O -- that is Stage2Detection's job.
        - Adapters must catch backend-specific exceptions and re-raise as
          standard Python exceptions with descriptive messages.
    """

    ADAPTER_NAME: str = "base"
    """Canonical name used in adapter registry and config files."""

    def __init__(self, config: "DetectionConfig") -> None:  # type: ignore[name-defined]
        """
        Initialize the adapter and load model weights.

        Args:
            config: Stage 2 configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(
            f"stage.s2_detection.{self.__class__.__name__}"
        )
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load model weights and prepare for inference.

        Called once during `__init__`. Raise a descriptive exception
        (ImportError, FileNotFoundError, RuntimeError) on failure.
        """

    @abstractmethod
    def detect(self, image: np.ndarray, page_num: int) -> List[RawDetection]:
        """
        Run inference on a single image and return raw detections.

        Args:
            image:    BGR numpy array (H, W, 3).
            page_num: Source page number (for logging only).

        Returns:
            List of RawDetection, already filtered by conf_threshold.
            May be empty.

        Raises:
            RuntimeError: If inference fails.
        """

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        Return adapter / model metadata for metrics and reporting.

        Override to include model name, weight path, parameter count, etc.
        """
        return {"adapter": self.ADAPTER_NAME}
