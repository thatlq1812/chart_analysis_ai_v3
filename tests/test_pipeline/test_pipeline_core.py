"""
Tests for PipelineBuilder and ChartAnalysisPipeline initialization.

Verifies builder creates all stages, respects toggles, and handles
invalid configuration without GPU or model weights.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from src.core_engine.builder import PipelineBuilder
from src.core_engine.exceptions import ConfigurationError
from src.core_engine.pipeline import ChartAnalysisPipeline
from src.core_engine.stages.base import BaseStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_config() -> OmegaConf:
    """Return a minimal valid merged config for all 5 stages."""
    return OmegaConf.create(
        {
            "pipeline": {
                "name": "test-pipeline",
                "version": "1.0.0",
                "stages": {
                    "ingestion": {"enabled": True},
                    "detection": {"enabled": True, "adapter": "mock"},
                    "extraction": {"enabled": True},
                    "reasoning": {"enabled": True},
                    "reporting": {"enabled": True},
                },
            },
            "session": {"id_prefix": "test", "timestamp_format": "%Y%m%d_%H%M%S"},
            "ingestion": {
                "pdf": {"dpi": 150},
                "image": {"max_size": 4096, "min_size": 100},
                "quality": {"blur_threshold": 100},
            },
            "yolo": {
                "path": None,
                "device": "cpu",
                "iou_threshold": 0.45,
                "input_size": 640,
            },
            "detection": {"confidence_threshold": 0.5},
            "ai_routing": {
                "providers": {
                    "gemini": {"model": "gemini-2.0-flash"},
                },
            },
            "reporting": {"insights": {"enabled": True}},
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineBuilder:
    """Tests for the PipelineBuilder factory."""

    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s4",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s3",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s2",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s1",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s5",
        return_value=MagicMock(spec=BaseStage),
    )
    def test_builder_creates_all_stages(
        self, _s5, _s1, _s2, _s3, _s4
    ) -> None:
        """5 stages should be created from a valid full config."""
        cfg = _full_config()
        stages = PipelineBuilder.build_stages(cfg)
        assert len(stages) == 5

    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s4",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s3",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s2",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s1",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s5",
        return_value=MagicMock(spec=BaseStage),
    )
    def test_builder_respects_stage_toggle(
        self, _s5, _s1, _s2, _s3, _s4
    ) -> None:
        """Disabled reasoning stage should reduce stage count to 4."""
        cfg = _full_config()
        cfg.pipeline.stages.reasoning.enabled = False
        stages = PipelineBuilder.build_stages(cfg)
        assert len(stages) == 4

    def test_pipeline_init_from_config(self) -> None:
        """ChartAnalysisPipeline.__init__ should accept OmegaConf config."""
        cfg = _full_config()
        pipeline = ChartAnalysisPipeline(cfg)
        assert pipeline.config is not None
        assert pipeline.config.pipeline.name == "test-pipeline"

    def test_pipeline_factory_method(self) -> None:
        """from_config() with real config dir should work if files exist."""
        config_dir = Path("config")
        if not (config_dir / "pipeline.yaml").exists():
            pytest.skip("config/pipeline.yaml not found")

        pipeline = ChartAnalysisPipeline.from_config(config_dir)
        assert pipeline.config is not None

    def test_pipeline_invalid_config_raises(self, tmp_path: Path) -> None:
        """Bad config dir should raise ConfigurationError."""
        with pytest.raises(ConfigurationError):
            ChartAnalysisPipeline.from_config(config_dir=tmp_path / "nonexistent")

    def test_pipeline_stage_access(self) -> None:
        """pipeline.stages should return a list."""
        cfg = _full_config()
        pipeline = ChartAnalysisPipeline(cfg)

        with patch.object(
            ChartAnalysisPipeline, "_initialize_stages"
        ) as mock_init:
            mock_init.return_value = [MagicMock() for _ in range(5)]
            assert isinstance(pipeline.stages, list)
            assert len(pipeline.stages) == 5

    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s4",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s3",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s2",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s1",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s5",
        return_value=MagicMock(spec=BaseStage),
    )
    def test_builder_all_disabled_returns_empty(
        self, _s5, _s1, _s2, _s3, _s4
    ) -> None:
        """All stages disabled should produce an empty list."""
        cfg = _full_config()
        cfg.pipeline.stages.ingestion.enabled = False
        cfg.pipeline.stages.detection.enabled = False
        cfg.pipeline.stages.extraction.enabled = False
        cfg.pipeline.stages.reasoning.enabled = False
        cfg.pipeline.stages.reporting.enabled = False
        stages = PipelineBuilder.build_stages(cfg)
        assert len(stages) == 0

    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s1",
        return_value=MagicMock(spec=BaseStage),
    )
    @patch(
        "src.core_engine.builder.PipelineBuilder._build_s5",
        return_value=MagicMock(spec=BaseStage),
    )
    def test_builder_only_ingestion_and_reporting(
        self, _s5, _s1
    ) -> None:
        """Only ingestion + reporting enabled -> 2 stages."""
        cfg = _full_config()
        cfg.pipeline.stages.detection.enabled = False
        cfg.pipeline.stages.extraction.enabled = False
        cfg.pipeline.stages.reasoning.enabled = False
        stages = PipelineBuilder.build_stages(cfg)
        assert len(stages) == 2

    def test_pipeline_version_from_config(self) -> None:
        """Pipeline version should be read from config."""
        cfg = _full_config()
        pipeline = ChartAnalysisPipeline(cfg)
        assert pipeline.config.pipeline.version == "1.0.0"
