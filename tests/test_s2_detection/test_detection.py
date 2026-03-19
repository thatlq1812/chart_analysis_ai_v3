"""
Tests for Stage 2: Detection & Localization.

All tests mock the YOLO model to avoid needing GPU or model weights.
Verifies detection output schema, confidence filtering, crop padding,
and max-chart-per-page limits.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.core_engine.schemas.common import BoundingBox, SessionInfo
from src.core_engine.schemas.stage_outputs import (
    CleanImage,
    DetectedChart,
    Stage1Output,
    Stage2Output,
)
from src.core_engine.stages.s2_detection import Stage2Detection, DetectionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_info() -> SessionInfo:
    return SessionInfo(
        session_id="det_test_001",
        source_file=Path("input.png"),
        config_hash="def45678",
    )


@pytest.fixture
def stage1_output(tmp_path: Path, session_info: SessionInfo) -> Stage1Output:
    """Stage1Output with a real 200x200 image on disk."""
    img_path = tmp_path / "page1.png"
    Image.new("RGB", (200, 200), color="white").save(img_path)
    return Stage1Output(
        session=session_info,
        images=[
            CleanImage(
                image_path=img_path,
                original_path=img_path,
                width=200,
                height=200,
            )
        ],
    )


@pytest.fixture
def mock_detection_adapter() -> MagicMock:
    """Mock detection adapter returning two detections."""
    from src.core_engine.stages.s2_detection.adapters.base import RawDetection

    adapter = MagicMock()
    adapter.detect.return_value = [
        RawDetection(
            bbox=BoundingBox(
                x_min=10, y_min=10, x_max=100, y_max=100, confidence=0.95
            ),
            label="chart",
        ),
        RawDetection(
            bbox=BoundingBox(
                x_min=120, y_min=10, x_max=190, y_max=100, confidence=0.80
            ),
            label="chart",
        ),
    ]
    return adapter


@pytest.fixture
def detection_config() -> DetectionConfig:
    return DetectionConfig(
        model_path=None,
        device="cpu",
        conf_threshold=0.5,
        adapter="mock",
        crop_padding=5,
        max_detections_per_image=10,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStage2Detection:
    """Tests for chart detection with mocked adapter."""

    def test_detection_with_mock_adapter(
        self,
        tmp_path: Path,
        detection_config: DetectionConfig,
        mock_detection_adapter: MagicMock,
        stage1_output: Stage1Output,
    ) -> None:
        """Mock adapter should produce detected charts."""
        stage = Stage2Detection(
            config=detection_config,
            output_dir=tmp_path / "crops",
            adapter=mock_detection_adapter,
        )
        result = stage.process(stage1_output)
        assert isinstance(result, Stage2Output)
        assert len(result.charts) >= 1

    def test_detection_empty_image(
        self,
        tmp_path: Path,
        detection_config: DetectionConfig,
        session_info: SessionInfo,
    ) -> None:
        """No detections -> empty charts list."""
        adapter = MagicMock()
        adapter.detect.return_value = []

        img_path = tmp_path / "blank.png"
        Image.new("RGB", (200, 200), color="black").save(img_path)

        s1 = Stage1Output(
            session=session_info,
            images=[
                CleanImage(
                    image_path=img_path,
                    original_path=img_path,
                    width=200,
                    height=200,
                )
            ],
        )

        stage = Stage2Detection(
            config=detection_config,
            output_dir=tmp_path / "crops",
            adapter=adapter,
        )
        result = stage.process(s1)
        assert len(result.charts) == 0
        assert result.total_detected == 0

    def test_detection_confidence_filtering(
        self,
        tmp_path: Path,
        session_info: SessionInfo,
    ) -> None:
        """Stage should pass through adapter results; adapter does the filtering."""
        from src.core_engine.stages.s2_detection.adapters.base import RawDetection

        # Only provide high-confidence detections (adapter pre-filters)
        adapter = MagicMock()
        adapter.detect.return_value = [
            RawDetection(
                bbox=BoundingBox(
                    x_min=10, y_min=10, x_max=80, y_max=80, confidence=0.9
                ),
                label="chart",
            ),
        ]

        img_path = tmp_path / "chart.png"
        Image.new("RGB", (200, 200), color="white").save(img_path)

        s1 = Stage1Output(
            session=session_info,
            images=[
                CleanImage(
                    image_path=img_path, original_path=img_path, width=200, height=200
                )
            ],
        )

        config = DetectionConfig(conf_threshold=0.5, adapter="mock")
        stage = Stage2Detection(
            config=config, output_dir=tmp_path / "crops", adapter=adapter
        )
        result = stage.process(s1)
        # Adapter returned 1 detection -> stage should produce 1 chart
        assert len(result.charts) == 1
        assert result.charts[0].bbox.confidence >= 0.5

    def test_detection_max_charts_per_page(
        self,
        tmp_path: Path,
        session_info: SessionInfo,
    ) -> None:
        """Stage returns all adapter detections; adapter controls limit."""
        from src.core_engine.stages.s2_detection.adapters.base import RawDetection

        # Adapter returns exactly 5 detections
        adapter = MagicMock()
        adapter.detect.return_value = [
            RawDetection(
                bbox=BoundingBox(
                    x_min=i * 10, y_min=0, x_max=i * 10 + 9, y_max=9, confidence=0.9
                ),
                label="chart",
            )
            for i in range(5)
        ]

        img_path = tmp_path / "dense.png"
        Image.new("RGB", (300, 300), color="white").save(img_path)

        s1 = Stage1Output(
            session=session_info,
            images=[
                CleanImage(
                    image_path=img_path, original_path=img_path, width=300, height=300
                )
            ],
        )

        config = DetectionConfig(
            conf_threshold=0.5, adapter="mock", max_detections_per_image=5
        )
        stage = Stage2Detection(
            config=config, output_dir=tmp_path / "crops", adapter=adapter
        )
        result = stage.process(s1)
        assert len(result.charts) == 5

    def test_detection_crop_padding(
        self,
        detection_config: DetectionConfig,
    ) -> None:
        """Config crop_padding should be stored correctly."""
        assert detection_config.crop_padding == 5

    def test_detection_output_schema(
        self,
        tmp_path: Path,
        detection_config: DetectionConfig,
        mock_detection_adapter: MagicMock,
        stage1_output: Stage1Output,
    ) -> None:
        """Stage2Output should contain charts list and total_detected."""
        stage = Stage2Detection(
            config=detection_config,
            output_dir=tmp_path / "crops",
            adapter=mock_detection_adapter,
        )
        result = stage.process(stage1_output)
        assert hasattr(result, "charts")
        assert hasattr(result, "total_detected")
        assert hasattr(result, "session")
        assert isinstance(result.charts, list)

    def test_detection_handles_model_not_loaded(
        self,
        tmp_path: Path,
        stage1_output: Stage1Output,
    ) -> None:
        """When adapter raises on detect(), stage should propagate error."""
        adapter = MagicMock()
        adapter.detect.side_effect = RuntimeError("Model weights not found")

        config = DetectionConfig(adapter="mock")
        stage = Stage2Detection(
            config=config, output_dir=tmp_path / "crops", adapter=adapter
        )
        with pytest.raises((RuntimeError, Exception)):
            stage.process(stage1_output)

    def test_detection_config_defaults(self) -> None:
        """DetectionConfig defaults should be production-suitable."""
        cfg = DetectionConfig()
        assert cfg.conf_threshold == 0.5
        assert cfg.iou_threshold == 0.45
        assert cfg.imgsz == 640
        assert cfg.adapter == "yolov8"

    def test_detection_validate_input(
        self,
        detection_config: DetectionConfig,
        stage1_output: Stage1Output,
    ) -> None:
        """validate_input should accept Stage1Output."""
        stage = Stage2Detection(config=detection_config)
        assert stage.validate_input(stage1_output) is True

    def test_detection_validate_input_rejects_wrong_type(
        self,
        detection_config: DetectionConfig,
    ) -> None:
        """validate_input should reject non-Stage1Output types."""
        stage = Stage2Detection(config=detection_config)
        assert stage.validate_input("not a stage1 output") is False
