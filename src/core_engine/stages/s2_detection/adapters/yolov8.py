"""
YOLOv8 Detection Adapter

Wraps Ultralytics YOLOv8 for chart region detection.
Also compatible with YOLO11 weights (same API, different architecture).

Usage in pipeline.yaml:
    s2_detection:
      adapter: yolov8
      config:
        model_path: models/weights/chart_detector_v3.pt
        conf_threshold: 0.5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ....exceptions import ModelNotLoadedError
from ....schemas.common import BoundingBox
from ..config import DetectionConfig
from .base import BaseDetectionAdapter, RawDetection
from ....registry import register

logger = logging.getLogger(__name__)


@register("s2_detection", "yolov8")
class YOLOv8Adapter(BaseDetectionAdapter):
    """
    Ultralytics YOLOv8 detection adapter.

    Supports YOLOv8 and YOLO11 weight files (same Ultralytics API).
    The adapter name 'yolov8' is the canonical name for any Ultralytics YOLO.
    To use v11 weights, just point model_path at a v11 .pt file.

    Adapter names registered:
        'yolov8'  - default Ultralytics YOLO adapter
        'yolov11' - alias for the same adapter (different weights expected)
    """

    ADAPTER_NAME = "yolov8"

    def _load_model(self) -> None:
        """Load YOLO model from model_path."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            ) from exc

        if self.config.model_path is None:
            raise ModelNotLoadedError("DetectionConfig.model_path is required for YOLOv8Adapter.")

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise ModelNotLoadedError(
                f"YOLO model weights not found: {model_path}. "
                "Check models.yaml or download the weights."
            )

        self._yolo = YOLO(str(model_path))

        if self.config.device not in ("auto", ""):
            self._yolo.to(self.config.device)

        logger.info(
            f"YOLOv8Adapter model loaded | path={model_path.name} | "
            f"device={self.config.device}"
        )

    def detect(self, image: np.ndarray, page_num: int) -> List[RawDetection]:
        """
        Run YOLOv8 inference on a BGR image.

        Args:
            image:    BGR numpy array from OpenCV.
            page_num: Source page number (for log context only).

        Returns:
            List[RawDetection] filtered by conf_threshold and min_area_pixels.
        """
        results = self._yolo.predict(
            image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        detections = self._parse(results[0], page_num)
        return detections

    def _parse(self, result: Any, page_num: int) -> List[RawDetection]:
        """Convert YOLO result boxes to RawDetection list."""
        detections: List[RawDetection] = []

        if not hasattr(result, "boxes") or result.boxes is None:
            return detections

        for box in result.boxes:
            try:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                x_min, y_min, x_max, y_max = (
                    int(coords[0]), int(coords[1]),
                    int(coords[2]), int(coords[3]),
                )
                area = (x_max - x_min) * (y_max - y_min)

                if conf < self.config.conf_threshold:
                    continue
                if area < self.config.min_area_pixels:
                    logger.debug(
                        f"Detection skipped (too small) | page={page_num} | "
                        f"area={area} | conf={conf:.3f}"
                    )
                    continue

                bbox = BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=conf,
                )
                detections.append(RawDetection(bbox=bbox))

            except Exception as exc:
                logger.warning(
                    f"Failed to parse YOLO box | page={page_num} | error={exc}"
                )

        detections.sort(key=lambda d: d.bbox.confidence, reverse=True)
        return detections[: self.config.max_detections_per_image]

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "adapter": self.ADAPTER_NAME,
            "model_path": str(self.config.model_path),
            "device": self.config.device,
            "conf_threshold": self.config.conf_threshold,
        }


# Register 'yolov11' as an alias pointing to the same class.
# Users can set adapter: yolov11 in config to signal v11 weights
# while using identical inference code.
@register("s2_detection", "yolov11")
class YOLOv11Adapter(YOLOv8Adapter):
    """
    YOLO11 detection adapter (alias of YOLOv8Adapter for v11 weights).

    Ultralytics YOLO11 uses the same Python API; the only difference is
    the weight file architecture. Set model_path to a v11 .pt file.

    Usage in pipeline.yaml:
        s2_detection:
          adapter: yolov11
          config:
            model_path: models/weights/chart_detector_yolov11.pt
    """

    ADAPTER_NAME = "yolov11"
