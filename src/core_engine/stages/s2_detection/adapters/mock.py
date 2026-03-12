"""
Mock Detection Adapter

Deterministic stub adapter for unit tests and CI environments where
YOLO model weights are not available.

Returns synthetic detections based on the image dimensions:
  - One detection covering the centre 60% of the image (conf=0.95)
  - One small detection in the top-left quadrant (conf=0.75)

Usage in pipeline.yaml:
    s2_detection:
      adapter: mock
      config: {}           # no model_path needed
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ....registry import register
from ....schemas.common import BoundingBox
from ..config import DetectionConfig
from .base import BaseDetectionAdapter, RawDetection


@register("s2_detection", "mock")
class MockDetectionAdapter(BaseDetectionAdapter):
    """
    Deterministic mock detection adapter for testing.

    Produces consistent synthetic detections so that downstream stage
    tests (Stage 3, Stage 4) can run without real YOLO weights.

    Registered name: 'mock'
    """

    ADAPTER_NAME = "mock"

    def _load_model(self) -> None:
        """No model to load for mock adapter."""
        self.logger.debug("MockDetectionAdapter: no model to load.")

    def detect(self, image: np.ndarray, page_num: int) -> List[RawDetection]:
        """
        Return synthetic detections proportional to image dimensions.

        Two detections are produced:
          1. A large central detection (60% of image area, conf=0.95).
          2. A small top-left detection (20% of image area, conf=0.75).

        Args:
            image:    BGR numpy array.
            page_num: Ignored (present for API compatibility).

        Returns:
            List of two RawDetection objects, or fewer if image is too small.
        """
        h, w = image.shape[:2]
        detections: List[RawDetection] = []

        # Detection 1: large central region
        cx1 = int(w * 0.2)
        cy1 = int(h * 0.2)
        cx2 = int(w * 0.8)
        cy2 = int(h * 0.8)

        if (cx2 - cx1) * (cy2 - cy1) >= self.config.min_area_pixels:
            detections.append(RawDetection(
                bbox=BoundingBox(
                    x_min=cx1, y_min=cy1, x_max=cx2, y_max=cy2, confidence=0.95
                ),
                label="chart",
                extra={"mock": True},
            ))

        # Detection 2: smaller top-left region
        sx2 = int(w * 0.4)
        sy2 = int(h * 0.35)
        if sx2 * sy2 >= self.config.min_area_pixels:
            detections.append(RawDetection(
                bbox=BoundingBox(
                    x_min=10, y_min=10, x_max=sx2, y_max=sy2, confidence=0.75
                ),
                label="chart",
                extra={"mock": True},
            ))

        return detections[: self.config.max_detections_per_image]

    @property
    def model_info(self) -> Dict[str, Any]:
        return {"adapter": self.ADAPTER_NAME, "mock": True}
