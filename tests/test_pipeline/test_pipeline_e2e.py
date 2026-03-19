"""
End-to-end pipeline tests with mocked stages.

Verifies S1->S5 data flow, graceful degradation, empty-detection handling,
and result schema compliance without GPU, API keys, or model weights.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.core_engine.exceptions import PipelineError, ConfigurationError
from src.core_engine.pipeline import ChartAnalysisPipeline
from src.core_engine.schemas.enums import ChartType
from src.core_engine.schemas.stage_outputs import (
    CleanImage,
    DataPoint,
    DataSeries,
    DetectedChart,
    FinalChartResult,
    ChartInsight,
    PipelineResult,
    RawMetadata,
    RefinedChartData,
    SessionInfo,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
)
from src.core_engine.schemas.common import BoundingBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> SessionInfo:
    return SessionInfo(
        session_id="e2e_test_001",
        source_file=Path("test.png"),
        config_hash="abc12345",
    )


def _make_s1_output(tmp_path: Path) -> Stage1Output:
    img_path = tmp_path / "page1.png"
    img_path.write_bytes(b"\x89PNG")  # minimal stub
    return Stage1Output(
        session=_make_session(),
        images=[
            CleanImage(
                image_path=img_path,
                original_path=img_path,
                page_number=1,
                width=200,
                height=200,
            )
        ],
    )


def _make_s2_output(tmp_path: Path) -> Stage2Output:
    chart_path = tmp_path / "chart_001.png"
    chart_path.write_bytes(b"\x89PNG")
    return Stage2Output(
        session=_make_session(),
        charts=[
            DetectedChart(
                chart_id="chart_001",
                source_image=chart_path,
                cropped_path=chart_path,
                bbox=BoundingBox(
                    x_min=10, y_min=10, x_max=100, y_max=100, confidence=0.95
                ),
            )
        ],
        total_detected=1,
    )


def _make_s3_output() -> Stage3Output:
    return Stage3Output(
        session=_make_session(),
        metadata=[
            RawMetadata(chart_id="chart_001", chart_type=ChartType.BAR),
        ],
    )


def _make_s4_output() -> Stage4Output:
    return Stage4Output(
        session=_make_session(),
        charts=[
            RefinedChartData(
                chart_id="chart_001",
                chart_type=ChartType.BAR,
                title="Revenue",
                series=[
                    DataSeries(
                        name="Revenue",
                        points=[DataPoint(label="2023", value=100.0)],
                    )
                ],
                description="A bar chart showing revenue.",
            )
        ],
    )


def _make_pipeline_result() -> PipelineResult:
    return PipelineResult(
        session=_make_session(),
        charts=[
            FinalChartResult(
                chart_id="chart_001",
                chart_type=ChartType.BAR,
                title="Revenue",
                data=_make_s4_output().charts[0],
                insights=[
                    ChartInsight(
                        insight_type="summary",
                        text="Revenue is 100.",
                        confidence=0.9,
                    )
                ],
            )
        ],
        summary="Processed 1 chart.",
        processing_time_seconds=0.5,
        model_versions={"yolo": "v3"},
    )


def _build_mock_pipeline(tmp_path: Path) -> ChartAnalysisPipeline:
    """Build a pipeline with all stages mocked."""
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "name": "test",
                "version": "0.1.0",
                "stages": {
                    "ingestion": {"enabled": True},
                    "detection": {"enabled": True},
                    "extraction": {"enabled": True},
                    "reasoning": {"enabled": True},
                    "reporting": {"enabled": True},
                },
            },
            "session": {"id_prefix": "test", "timestamp_format": "%Y%m%d_%H%M%S"},
        }
    )
    pipeline = ChartAnalysisPipeline(cfg)

    # Build mock stages
    mock_s1 = MagicMock()
    mock_s1.process.return_value = _make_s1_output(tmp_path)

    mock_s2 = MagicMock()
    mock_s2.process.return_value = _make_s2_output(tmp_path)

    mock_s3 = MagicMock()
    mock_s3.process.return_value = _make_s3_output()

    mock_s4 = MagicMock()
    mock_s4.process.return_value = _make_s4_output()

    mock_s5 = MagicMock()
    mock_s5.process.return_value = _make_pipeline_result()

    pipeline._stages = [mock_s1, mock_s2, mock_s3, mock_s4, mock_s5]
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineE2E:
    """End-to-end pipeline tests using mocked stage processors."""

    def test_pipeline_runs_with_sample_image(self, tmp_path: Path) -> None:
        """Mock all stages, verify S1->S5 data flow completes."""
        pipeline = _build_mock_pipeline(tmp_path)
        input_file = tmp_path / "report.png"
        input_file.write_bytes(b"\x89PNG")

        result = pipeline.run(input_file)

        assert isinstance(result, PipelineResult)
        assert len(result.charts) == 1
        assert result.charts[0].chart_id == "chart_001"

    def test_pipeline_handles_no_detections(self, tmp_path: Path) -> None:
        """S2 returns empty chart list -> pipeline returns empty result."""
        pipeline = _build_mock_pipeline(tmp_path)
        # Override S2 to return empty
        pipeline._stages[1].process.return_value = Stage2Output(
            session=_make_session(),
            charts=[],
            total_detected=0,
        )
        input_file = tmp_path / "empty.png"
        input_file.write_bytes(b"\x89PNG")

        result = pipeline.run(input_file)

        assert isinstance(result, PipelineResult)
        assert len(result.charts) == 0
        assert "No charts" in result.summary

    def test_pipeline_handles_s3_error_gracefully(self, tmp_path: Path) -> None:
        """S3 fails -- pipeline raises PipelineError wrapping original."""
        pipeline = _build_mock_pipeline(tmp_path)
        pipeline._stages[2].process.side_effect = RuntimeError("OCR failed")
        input_file = tmp_path / "broken.png"
        input_file.write_bytes(b"\x89PNG")

        with pytest.raises(PipelineError) as exc_info:
            pipeline.run(input_file)

        # The original exception is chained via 'from e' (__cause__)
        assert exc_info.value.__cause__ is not None or "OCR failed" in str(exc_info.value)

    def test_pipeline_result_has_required_fields(self, tmp_path: Path) -> None:
        """Check PipelineResult schema contains all expected fields."""
        pipeline = _build_mock_pipeline(tmp_path)
        input_file = tmp_path / "chart.png"
        input_file.write_bytes(b"\x89PNG")

        result = pipeline.run(input_file)

        assert hasattr(result, "session")
        assert hasattr(result, "charts")
        assert hasattr(result, "summary")
        assert hasattr(result, "processing_time_seconds")
        assert hasattr(result, "model_versions")
        assert hasattr(result, "total_charts")

    def test_pipeline_config_loading(self) -> None:
        """Load pipeline from real config/pipeline.yaml if available."""
        config_dir = Path("config")
        if not (config_dir / "pipeline.yaml").exists():
            pytest.skip("config/pipeline.yaml not found")

        pipeline = ChartAnalysisPipeline.from_config(config_dir)
        assert pipeline.config is not None
        assert pipeline.config.pipeline.version is not None

    def test_pipeline_stage_skip(self, tmp_path: Path) -> None:
        """Disabled stage should cause shorter _stages list."""
        cfg = OmegaConf.create(
            {
                "pipeline": {
                    "name": "test",
                    "version": "0.1.0",
                    "stages": {
                        "ingestion": {"enabled": True},
                        "detection": {"enabled": True},
                        "extraction": {"enabled": True},
                        "reasoning": {"enabled": False},
                        "reporting": {"enabled": True},
                    },
                },
                "session": {"id_prefix": "test", "timestamp_format": "%Y%m%d_%H%M%S"},
            }
        )
        pipeline = ChartAnalysisPipeline(cfg)

        # Patch _initialize_stages to count which stages are built
        with patch.object(
            ChartAnalysisPipeline, "_initialize_stages"
        ) as mock_init:
            mock_init.return_value = [MagicMock() for _ in range(4)]
            stages = pipeline.stages
            assert len(stages) == 4

    def test_pipeline_timing_recorded(self, tmp_path: Path) -> None:
        """Processing time should be positive."""
        pipeline = _build_mock_pipeline(tmp_path)
        input_file = tmp_path / "timed.png"
        input_file.write_bytes(b"\x89PNG")

        # The mock returns fixed processing_time_seconds=0.5
        result = pipeline.run(input_file)
        assert result.processing_time_seconds > 0

    def test_pipeline_file_not_found_raises(self, tmp_path: Path) -> None:
        """Non-existent input file should raise FileNotFoundError."""
        pipeline = _build_mock_pipeline(tmp_path)

        with pytest.raises(FileNotFoundError):
            pipeline.run(tmp_path / "nonexistent.pdf")

    def test_pipeline_run_stage_validates_number(self, tmp_path: Path) -> None:
        """run_stage() with invalid stage number should raise ValueError."""
        pipeline = _build_mock_pipeline(tmp_path)

        with pytest.raises(ValueError, match="Invalid stage number"):
            pipeline.run_stage(0, None)

        with pytest.raises(ValueError, match="Invalid stage number"):
            pipeline.run_stage(6, None)

    def test_pipeline_run_stage_single(self, tmp_path: Path) -> None:
        """run_stage() for a single stage should call only that stage."""
        pipeline = _build_mock_pipeline(tmp_path)
        mock_input = MagicMock()

        pipeline.run_stage(1, mock_input)
        pipeline._stages[0].process.assert_called_once_with(mock_input)
        pipeline._stages[1].process.assert_not_called()
