"""
Pipeline Builder

Constructs a ChartAnalysisPipeline from OmegaConf configuration,
resolving stage adapters via AdapterRegistry.

Usage:
    from omegaconf import OmegaConf
    from src.core_engine.builder import PipelineBuilder

    config = OmegaConf.load("config/pipeline.yaml")
    stages = PipelineBuilder.build_stages(config)

This module decouples stage construction from ChartAnalysisPipeline,
making it easy to swap adapters per stage via pipeline.yaml without
touching pipeline orchestration code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from .exceptions import ConfigurationError, StageProcessingError
from .registry import AdapterRegistry
from .stages.base import BaseStage

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Factory for assembling pipeline stages from OmegaConf config.

    Stage construction order mirrors the pipeline data flow:
        S1 Ingestion -> S2 Detection -> S3 Extraction -> S4 Reasoning -> S5 Reporting

    Each stage with an 'adapter' key in pipeline.yaml will have its
    backend resolved through AdapterRegistry. Stages without adapters
    continue to use their direct constructor (legacy behaviour).
    """

    @classmethod
    def build_stages(cls, config: DictConfig) -> List[BaseStage]:
        """
        Construct all enabled pipeline stages from merged OmegaConf config.

        Args:
            config: Fully-merged OmegaConf config
                    (base.yaml + models.yaml + pipeline.yaml).

        Returns:
            List of initialized BaseStage instances in execution order.

        Raises:
            ConfigurationError: If a required config key is missing.
            StageProcessingError: If an adapter cannot be instantiated.
        """
        stages: List[BaseStage] = []
        pipeline_cfg = config.pipeline.stages

        # Stage 1: Ingestion
        if OmegaConf.select(pipeline_cfg, "ingestion.enabled", default=True):
            stages.append(cls._build_s1(config))
            logger.info("Stage 1 (Ingestion) initialized via PipelineBuilder")

        # Stage 2: Detection (adapter-driven)
        if OmegaConf.select(pipeline_cfg, "detection.enabled", default=True):
            stages.append(cls._build_s2(config))
            logger.info("Stage 2 (Detection) initialized via PipelineBuilder")

        # Stage 3: Extraction
        if OmegaConf.select(pipeline_cfg, "extraction.enabled", default=True):
            stages.append(cls._build_s3(config))
            logger.info("Stage 3 (Extraction) initialized via PipelineBuilder")

        # Stage 4: Reasoning
        if OmegaConf.select(pipeline_cfg, "reasoning.enabled", default=True):
            stages.append(cls._build_s4(config))
            logger.info("Stage 4 (Reasoning) initialized via PipelineBuilder")

        # Stage 5: Reporting
        if OmegaConf.select(pipeline_cfg, "reporting.enabled", default=True):
            stages.append(cls._build_s5(config))
            logger.info("Stage 5 (Reporting) initialized via PipelineBuilder")

        logger.info(
            f"PipelineBuilder complete | active_stages={len(stages)}"
        )
        return stages

    # ------------------------------------------------------------------
    # Per-stage construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_s1(cls, config: DictConfig) -> BaseStage:
        """Build Stage 1: Ingestion."""
        from .stages.s1_ingestion import Stage1Ingestion, IngestionConfig

        s1_config = IngestionConfig(
            pdf_dpi=OmegaConf.select(config, "ingestion.pdf.dpi", default=150),
            max_image_size=OmegaConf.select(
                config, "ingestion.image.max_size", default=4096
            ),
            min_image_size=OmegaConf.select(
                config, "ingestion.image.min_size", default=100
            ),
            min_blur_threshold=OmegaConf.select(
                config, "ingestion.quality.blur_threshold", default=100
            ),
        )
        return Stage1Ingestion(s1_config)

    @classmethod
    def _build_s2(cls, config: DictConfig) -> BaseStage:
        """
        Build Stage 2: Detection.

        Resolves the adapter via AdapterRegistry using the
        pipeline.stages.detection.adapter key from config.
        Falls back to 'yolov8' if key is absent.
        """
        from .stages.s2_detection import Stage2Detection, DetectionConfig

        adapter_name: str = OmegaConf.select(
            config, "pipeline.stages.detection.adapter", default="yolov8"
        )
        s2_config = DetectionConfig(
            model_path=OmegaConf.select(config, "yolo.path", default=None),
            device=OmegaConf.select(config, "yolo.device", default="auto"),
            conf_threshold=OmegaConf.select(
                config, "detection.confidence_threshold", default=0.5
            ),
            iou_threshold=OmegaConf.select(
                config, "yolo.iou_threshold", default=0.45
            ),
            imgsz=OmegaConf.select(config, "yolo.input_size", default=640),
            adapter=adapter_name,
        )
        logger.info(
            f"PipelineBuilder | Stage 2 adapter resolved | adapter={adapter_name}"
        )
        return Stage2Detection(s2_config)

    @classmethod
    def _build_s3(cls, config: DictConfig) -> BaseStage:
        """Build Stage 3: Extraction."""
        from .stages.s3_extraction import Stage3Extraction

        return Stage3Extraction()

    @classmethod
    def _build_s4(cls, config: DictConfig) -> BaseStage:
        """Build Stage 4: Reasoning."""
        from .stages.s4_reasoning import Stage4Reasoning, ReasoningConfig
        from .stages.s4_reasoning import GeminiConfig

        gemini_cfg = GeminiConfig(
            model_name=OmegaConf.select(
                config,
                "ai_routing.providers.gemini.model",
                default="gemini-2.0-flash",
            )
        )
        s4_config = ReasoningConfig(
            engine="router",
            gemini=gemini_cfg,
        )
        return Stage4Reasoning(s4_config)

    @classmethod
    def _build_s5(cls, config: DictConfig) -> BaseStage:
        """Build Stage 5: Reporting."""
        from .stages.s5_reporting import Stage5Reporting, ReportingConfig

        s5_config = ReportingConfig(
            enable_insights=OmegaConf.select(
                config, "reporting.insights.enabled", default=True
            ),
            save_json=True,
            save_report=True,
        )
        return Stage5Reporting(s5_config)
