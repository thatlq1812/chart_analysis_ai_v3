"""
Stage 3: Structural Analysis (Extraction)

VLM-based chart-to-table extraction orchestrator.

This stage replaces the legacy geometry pipeline (OCR, skeletonization,
RANSAC axis calibration, pixel-to-value mapping) with a pluggable VLM
extractor that converts chart images directly to structured data tables.

Supported backends (via ExtractionConfig.extractor_backend):
    - deplot     : google/deplot (recommended default)
    - pix2struct : google/pix2struct-base (ablation baseline)
    - matcha     : google/matcha-base (enhanced math+chart reasoning)
    - svlm       : Qwen/Qwen2-VL-2B-Instruct (zero-shot large VLM)

Per-chart processing flow:
    1. Load cropped chart image from disk
    2. Classify chart type (EfficientNet-B0, optional)
    3. Run VLM extractor -> Pix2StructResult (headers + rows)
    4. Assemble RawMetadata for Stage 4

Reference: docs/architecture/STAGE3_EXTRACTION.md
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ...exceptions import StageProcessingError
from ...schemas.enums import ChartType
from ...schemas.stage_outputs import (
    ExtractionConfidence,
    RawMetadata,
    Stage2Output,
    Stage3Output,
)
from ..base import BaseStage
from .extractors import BackendType, BaseChartExtractor, create_extractor
from .resnet_classifier import (
    EfficientNetClassifier,
    create_efficientnet_classifier,
)

logger = logging.getLogger(__name__)


class ExtractionConfig(BaseModel):
    """Configuration for Stage 3: VLM-based chart extraction."""

    # --- VLM extractor backend ---
    extractor_backend: str = Field(
        default="deplot",
        description=(
            "VLM extraction backend. Options: "
            "deplot (google/deplot, default), "
            "pix2struct (google/pix2struct-base, ablation baseline), "
            "matcha (google/matcha-base, enhanced math reasoning), "
            "svlm (Qwen/Qwen2-VL-2B-Instruct, zero-shot large VLM)."
        ),
    )
    extractor_model: Optional[str] = Field(
        default=None,
        description=(
            "Override the default HuggingFace model ID for the selected backend. "
            "Useful for local model paths or fine-tuned variants. "
            "None = use the backend built-in default."
        ),
    )
    extractor_max_patches: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description=(
            "Max image patches for Pix2Struct-family backends "
            "(deplot, matcha, pix2struct). Higher = better quality but more VRAM. "
            "1024 is the recommended default for 12 GB VRAM."
        ),
    )
    extractor_device: str = Field(
        default="auto",
        description="Compute device for the extractor: auto, cuda, or cpu.",
    )

    # --- Chart type classifier ---
    use_efficientnet_classifier: bool = Field(
        default=True,
        description=(
            "Use EfficientNet-B0 classifier (97.54% accuracy, 3-class: bar/line/pie) "
            "to annotate chart type metadata. When False, chart_type defaults to UNKNOWN."
        ),
    )
    efficientnet_model_path: Optional[Path] = Field(
        default=None,
        description=(
            "Path to EfficientNet-B0 weights. "
            "Defaults to models/weights/efficientnet_b0_3class_v1_best.pt"
        ),
    )
    efficientnet_classes: Optional[list] = Field(
        default=None,
        description="Class list for EfficientNet. None = [bar, line, pie] (3-class default).",
    )
    efficientnet_confidence_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept EfficientNet classification.",
    )

    # --- Debug ---
    save_debug_images: bool = Field(
        default=False,
        description="Save intermediate images for debugging.",
    )
    debug_output_dir: Optional[Path] = Field(
        default=None,
        description="Directory for debug images. Defaults to data/cache/s3_debug/.",
    )


class Stage3Extraction(BaseStage):
    """
    Stage 3 Orchestrator: VLM-based Chart Extraction.

    Replaces the legacy geometry pipeline with a pluggable VLM extractor
    that converts chart images directly to structured data tables.

    Per-chart processing flow:
        1. Load cropped chart image
        2. Classify chart type (EfficientNet-B0, optional)
        3. Run VLM extractor (DePlot / MatCha / Pix2Struct / SVLM)
        4. Build RawMetadata for Stage 4

    Example:
        config = ExtractionConfig(extractor_backend="deplot")
        stage = Stage3Extraction(config)
        result = stage.process(stage2_output)
    """

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        config = config or ExtractionConfig()
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # VLM extractor - model loaded lazily on first .extract() call
        self._extractor: BaseChartExtractor = create_extractor(
            backend=config.extractor_backend,
            model_override=config.extractor_model,
            device=config.extractor_device,
            max_patches=config.extractor_max_patches,
        )
        self.logger.info(
            f"Stage3Extraction initialized | "
            f"backend={config.extractor_backend} | "
            f"model={self._extractor.model_name}"
        )

        # EfficientNet-B0 classifier (optional chart-type annotation)
        self._efficientnet: Optional[EfficientNetClassifier] = None
        if config.use_efficientnet_classifier:
            try:
                self._efficientnet = create_efficientnet_classifier(
                    model_path=config.efficientnet_model_path,
                    class_names=config.efficientnet_classes,
                    confidence_threshold=config.efficientnet_confidence_threshold,
                )
                self.logger.info(
                    f"EfficientNet-B0 classifier loaded | accuracy=97.54% | "
                    f"path={self._efficientnet.model_path.name}"
                )
            except Exception as exc:
                self.logger.warning(
                    f"EfficientNet classifier unavailable: {exc}. "
                    "chart_type will default to UNKNOWN."
                )

    # -------------------------------------------------------------------------
    # BaseStage interface
    # -------------------------------------------------------------------------

    def process(self, stage2_output: Stage2Output) -> Stage3Output:
        """
        Process all charts from Stage 2 output.

        Args:
            stage2_output: Detected charts from Stage 2.

        Returns:
            Stage3Output with RawMetadata for each successfully processed chart.
        """
        session = stage2_output.session
        metadata_list: List[RawMetadata] = []

        self.logger.info(
            f"Stage3 started | session={session.session_id} | "
            f"charts={len(stage2_output.charts)} | "
            f"backend={self.config.extractor_backend}"
        )

        for chart in stage2_output.charts:
            try:
                meta = self._process_single_chart(chart)
                if meta is not None:
                    metadata_list.append(meta)
            except Exception as exc:
                self.logger.error(
                    f"Chart processing failed | "
                    f"chart_id={chart.chart_id} | error={exc}"
                )

        self.logger.info(
            f"Stage3 complete | session={session.session_id} | "
            f"processed={len(metadata_list)}/{len(stage2_output.charts)}"
        )

        return Stage3Output(session=session, metadata=metadata_list)

    def validate_input(self, input_data) -> bool:
        """Validate input is Stage2Output."""
        return isinstance(input_data, Stage2Output)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _process_single_chart(self, chart) -> Optional[RawMetadata]:
        """
        Extract structured data from a single detected chart.

        Args:
            chart: DetectedChart from Stage 2.

        Returns:
            RawMetadata or None if the image cannot be loaded.
        """
        chart_id = chart.chart_id

        image_bgr = cv2.imread(str(chart.cropped_path))
        if image_bgr is None:
            self.logger.error(
                f"Image load failed | chart_id={chart_id} | path={chart.cropped_path}"
            )
            return None

        chart_type, classification_confidence = self._classify_chart(
            image_bgr, chart_id
        )

        vlm_result = self._extractor.extract(image_bgr, chart_id=chart_id)

        extraction_confidence = ExtractionConfidence(
            classification_confidence=classification_confidence,
            overall_confidence=classification_confidence,
        )

        return RawMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            texts=[],
            elements=[],
            axis_info=None,
            confidence=extraction_confidence,
            pix2struct_table=vlm_result,
        )

    def _classify_chart(
        self,
        image_bgr: np.ndarray,
        chart_id: str,
    ) -> Tuple[ChartType, float]:
        """
        Classify chart type using EfficientNet-B0.

        Returns:
            (chart_type, confidence). Returns (ChartType.UNKNOWN, 0.0) when
            the classifier is unavailable or confidence falls below threshold.
        """
        if self._efficientnet is None:
            return ChartType.UNKNOWN, 0.0

        try:
            chart_type_str, confidence = self._efficientnet.predict_with_confidence(
                image_bgr
            )
            try:
                chart_type = ChartType(chart_type_str)
            except ValueError:
                chart_type = ChartType.UNKNOWN

            self.logger.debug(
                f"Chart classified | chart_id={chart_id} | "
                f"type={chart_type} | confidence={confidence:.3f}"
            )
            return chart_type, confidence

        except Exception as exc:
            self.logger.warning(
                f"Classification failed | chart_id={chart_id} | error={exc}"
            )
            return ChartType.UNKNOWN, 0.0
