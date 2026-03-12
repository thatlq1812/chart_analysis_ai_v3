"""
Stage 2: Detection & Localization

Orchestrates chart region detection using a pluggable adapter backend.
Delegates inference to the configured adapter (YOLOv8, Mock, etc.)
and handles cropping, persistence, and schema construction.

Registered adapters:
    'yolov8'  - Ultralytics YOLOv8 / YOLOv11
    'mock'    - Deterministic stub for testing

Adapter selection via pipeline.yaml:
    s2_detection:
      adapter: yolov8    # or yolov11 / mock
      config:
        model_path: models/weights/chart_detector_v3.pt
        conf_threshold: 0.5
"""

from __future__ import annotations

import uuid
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ...schemas.stage_outputs import DetectedChart, Stage1Output, Stage2Output
from ...schemas.common import BoundingBox, SessionInfo
from ...exceptions import StageProcessingError
from ..base import BaseStage
from .config import DetectionConfig
from .adapters.base import BaseDetectionAdapter, RawDetection

logger = logging.getLogger("stage.s2_detection")


class Stage2Detection(BaseStage[Stage1Output, Stage2Output]):
    """
    Stage 2: Detection & Localization.

    Accepts a Stage1Output (list of clean images) and runs each image
    through the configured detection adapter to produce cropped chart
    images with associated bounding boxes.

    The adapter is resolved at construction time via the DetectionConfig
    adapter field (defaults to 'yolov8').
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        output_dir: Optional[Path] = None,
        adapter: Optional[BaseDetectionAdapter] = None,
    ) -> None:
        """
        Initialize Stage 2.

        Args:
            config:     Detection configuration. Defaults to DetectionConfig().
            output_dir: Directory for cropped images. Falls back to
                        config cache dir or a 'detected_charts' subdirectory
                        relative to the first processed image.
            adapter:    Pre-built adapter instance. When None, the adapter is
                        resolved from AdapterRegistry using config.adapter.
        """
        if config is None:
            config = DetectionConfig()

        super().__init__(config)

        self._output_dir = output_dir
        self._adapter = adapter  # injected or lazily resolved in process()
        self.config: DetectionConfig = config

    # ------------------------------------------------------------------
    # BaseStage API
    # ------------------------------------------------------------------

    def process(self, input_data: Stage1Output) -> Stage2Output:
        """
        Run detection on all images in Stage1Output.

        Args:
            input_data: Output from Stage 1 (list of clean images).

        Returns:
            Stage2Output with all detected charts.

        Raises:
            StageProcessingError: On unrecoverable inference or I/O failure.
        """
        session_id = input_data.session.session_id
        logger.info(
            f"Stage 2 started | session={session_id} | "
            f"images={input_data.total_images}"
        )

        adapter = self._get_adapter()
        output_dir = self._resolve_output_dir(input_data)

        all_charts: List[DetectedChart] = []
        skipped = 0

        for clean_img in input_data.images:
            page_num = clean_img.page_number
            try:
                raw_dets, n_skipped = self._detect_on_image(
                    adapter, clean_img.image_path, page_num
                )
                skipped += n_skipped

                charts = self._build_detected_charts(
                    raw_dets, clean_img.image_path, page_num, output_dir
                )
                all_charts.extend(charts)

            except StageProcessingError:
                raise
            except Exception as exc:
                raise StageProcessingError(
                    message=str(exc),
                    stage="s2_detection",
                    recoverable=False,
                ) from exc

        logger.info(
            f"Stage 2 complete | session={session_id} | "
            f"charts_found={len(all_charts)} | skipped_low_conf={skipped}"
        )

        return Stage2Output(
            session=input_data.session,
            charts=all_charts,
            total_detected=len(all_charts),
            skipped_low_confidence=skipped,
        )

    def validate_input(self, input_data: Stage1Output) -> bool:
        """Validate Stage1Output before processing."""
        return (
            isinstance(input_data, Stage1Output)
            and input_data.session is not None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_adapter(self) -> BaseDetectionAdapter:
        """Return existing adapter or resolve one from the registry."""
        if self._adapter is not None:
            return self._adapter

        from ...registry import AdapterRegistry

        adapter_name = getattr(self.config, "adapter", "yolov8")
        try:
            adapter_cls = AdapterRegistry.resolve("s2_detection", adapter_name)
        except KeyError as exc:
            raise StageProcessingError(
                message=f"Unknown detection adapter '{adapter_name}'. "
                        f"Available: {list(AdapterRegistry.list_adapters('s2_detection').keys())}",
                stage="s2_detection",
                recoverable=False,
            ) from exc

        self._adapter = adapter_cls(self.config)
        return self._adapter

    def _resolve_output_dir(self, input_data: Stage1Output) -> Path:
        """Determine where to save cropped chart images."""
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            return self._output_dir

        if input_data.images:
            base = Path(input_data.images[0].image_path).parent.parent
        else:
            base = Path("data/cache")

        out = base / "detected_charts"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _detect_on_image(
        self,
        adapter: BaseDetectionAdapter,
        image_path: Path,
        page_num: int,
    ):
        """Load an image and run the adapter."""
        import cv2

        img_path = Path(image_path)
        if not img_path.exists():
            logger.warning(
                f"Stage 2 skipping missing image | page={page_num} | path={img_path}"
            )
            return [], 0

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning(
                f"Stage 2 could not read image | page={page_num} | path={img_path}"
            )
            return [], 0

        raw_dets = adapter.detect(bgr, page_num)

        # Count detections that were filtered due to conf < threshold
        # (adapter already filters, but we record a nominal zero here)
        skipped = 0
        logger.debug(
            f"Adapter returned | page={page_num} | dets={len(raw_dets)}"
        )
        return raw_dets, skipped

    def _build_detected_charts(
        self,
        raw_dets: List[RawDetection],
        source_path: Path,
        page_num: int,
        output_dir: Path,
    ) -> List[DetectedChart]:
        """Convert RawDetection list to DetectedChart list, saving crops."""
        charts: List[DetectedChart] = []

        source_img = None  # lazy load once per image

        for detection in raw_dets:
            chart_id = f"chart_{uuid.uuid4().hex[:8]}"
            bbox = detection.bbox

            crop_path = self._save_crop(
                source_path, bbox, output_dir, chart_id, source_img
            )
            if crop_path is None:
                continue

            charts.append(DetectedChart(
                chart_id=chart_id,
                source_image=source_path,
                cropped_path=crop_path,
                bbox=bbox,
                page_number=page_num,
            ))

            logger.debug(
                f"Chart saved | chart_id={chart_id} | page={page_num} | "
                f"conf={bbox.confidence:.3f} | path={crop_path.name}"
            )

        return charts

    def _save_crop(
        self,
        source_path: Path,
        bbox: BoundingBox,
        output_dir: Path,
        chart_id: str,
        cached_img: Optional[Image.Image],
    ) -> Optional[Path]:
        """Crop the bounding box region from the source image and save it."""
        if not self.config.save_cropped_images:
            # Return None path - caller must handle
            return output_dir / f"{chart_id}.png"

        try:
            pil_img = Image.open(source_path).convert("RGB")
            pad = self.config.crop_padding

            x1 = max(0, bbox.x_min - pad)
            y1 = max(0, bbox.y_min - pad)
            x2 = min(pil_img.width, bbox.x_max + pad)
            y2 = min(pil_img.height, bbox.y_max + pad)

            crop = pil_img.crop((x1, y1, x2, y2))
            crop_path = output_dir / f"{chart_id}.png"
            crop.save(crop_path, format="PNG")
            return crop_path

        except Exception as exc:
            logger.error(
                f"Failed to save crop | chart_id={chart_id} | error={exc}"
            )
            return None
