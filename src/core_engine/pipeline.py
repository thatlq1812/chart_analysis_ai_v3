"""
Pipeline Orchestrator

Main entry point for the chart analysis pipeline.
Coordinates all stages and handles data flow.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import hashlib
import time

from omegaconf import OmegaConf, DictConfig

from .exceptions import PipelineError, ConfigurationError
from .schemas.stage_outputs import (
    SessionInfo,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
    PipelineResult,
)

logger = logging.getLogger(__name__)


class ChartAnalysisPipeline:
    """
    Main orchestrator for chart analysis pipeline.
    
    Coordinates the 5-stage processing:
    1. Ingestion: Load and normalize input files
    2. Detection: Detect chart regions using YOLO
    3. Extraction: OCR + element detection + geometric analysis
    4. Reasoning: SLM-based correction and value mapping
    5. Reporting: Generate final output
    
    Attributes:
        config: Pipeline configuration
        stages: List of initialized stage processors
        
    Example:
        pipeline = ChartAnalysisPipeline.from_config()
        result = pipeline.run("report.pdf")
        
        # Access results
        for chart in result.charts:
            print(f"Chart: {chart.title}")
            for series in chart.data.series:
                print(f"  Series: {series.name}")
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: OmegaConf configuration object
        """
        self.config = config
        self._stages = None
        self._start_time: Optional[float] = None
        
        logger.info(f"Initialized ChartAnalysisPipeline v{config.pipeline.version}")
    
    @classmethod
    def from_config(
        cls,
        config_dir: Path = Path("config"),
        overrides: Optional[dict] = None,
    ) -> "ChartAnalysisPipeline":
        """
        Create pipeline from configuration files.
        
        Args:
            config_dir: Directory containing YAML config files
            overrides: Optional dict of config overrides
            
        Returns:
            Initialized ChartAnalysisPipeline instance
            
        Raises:
            ConfigurationError: If config files are missing or invalid
        """
        try:
            # Load config files
            base = OmegaConf.load(config_dir / "base.yaml")
            models = OmegaConf.load(config_dir / "models.yaml")
            pipeline = OmegaConf.load(config_dir / "pipeline.yaml")
            
            # Merge configs
            config = OmegaConf.merge(base, models, pipeline)
            
            # Apply overrides
            if overrides:
                config = OmegaConf.merge(config, OmegaConf.create(overrides))
            
            # Resolve interpolations
            OmegaConf.resolve(config)
            
            return cls(config)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}") from e
    
    @property
    def stages(self) -> List:
        """Lazy-load stage processors."""
        if self._stages is None:
            self._stages = self._initialize_stages()
        return self._stages
    
    def _initialize_stages(self) -> List:
        """
        Initialize all pipeline stages from configuration.

        Returns:
            List of stage processor instances in execution order
        """
        from .stages import (
            Stage1Ingestion,
            IngestionConfig,
            Stage2Detection,
            DetectionConfig,
            Stage3Extraction,
            ExtractionConfig,
            Stage4Reasoning,
            ReasoningConfig,
            Stage5Reporting,
            ReportingConfig,
        )
        from .stages.s4_reasoning import GeminiConfig

        stages = []
        cfg = self.config

        # Stage 1: Ingestion
        if cfg.pipeline.stages.ingestion.enabled:
            s1_config = IngestionConfig(
                pdf_dpi=cfg.ingestion.pdf.dpi,
                max_image_size=cfg.ingestion.image.max_size,
                min_image_size=cfg.ingestion.image.min_size,
                min_blur_threshold=cfg.ingestion.quality.blur_threshold,
            )
            stages.append(Stage1Ingestion(s1_config))
            logger.info("Stage 1 (Ingestion) initialized")

        # Stage 2: Detection
        if cfg.pipeline.stages.detection.enabled:
            s2_config = DetectionConfig(
                model_path=cfg.yolo.path,
                device=cfg.yolo.device,
                conf_threshold=cfg.detection.confidence_threshold,
                iou_threshold=cfg.yolo.iou_threshold,
                imgsz=cfg.yolo.input_size,
            )
            stages.append(Stage2Detection(s2_config))
            logger.info("Stage 2 (Detection) initialized")

        # Stage 3: Extraction
        if cfg.pipeline.stages.extraction.enabled:
            stages.append(Stage3Extraction())
            logger.info("Stage 3 (Extraction) initialized")

        # Stage 4: Reasoning -- uses AIRouter by default (multi-provider)
        if cfg.pipeline.stages.reasoning.enabled:
            gemini_cfg = GeminiConfig(
                model_name=cfg.ai_routing.providers.gemini.model,
            )
            s4_config = ReasoningConfig(
                engine="router",  # Use multi-provider AIRouter
                gemini=gemini_cfg,
            )
            stages.append(Stage4Reasoning(s4_config))
            logger.info("Stage 4 (Reasoning) initialized | engine=router")

        # Stage 5: Reporting
        if cfg.pipeline.stages.reporting.enabled:
            s5_config = ReportingConfig(
                enable_insights=cfg.reporting.insights.enabled,
                save_json=True,
                save_report=True,
            )
            stages.append(Stage5Reporting(s5_config))
            logger.info("Stage 5 (Reporting) initialized")

        logger.info(f"Pipeline initialized | active_stages={len(stages)}")
        return stages
    
    def _create_session(self, input_path: Path) -> SessionInfo:
        """
        Create session info for this pipeline run.
        
        Args:
            input_path: Path to input file
            
        Returns:
            SessionInfo with unique ID and metadata
        """
        timestamp = datetime.now().strftime(self.config.session.timestamp_format)
        session_id = f"{self.config.session.id_prefix}_{timestamp}"
        
        # Compute config hash for reproducibility
        config_str = OmegaConf.to_yaml(self.config)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return SessionInfo(
            session_id=session_id,
            source_file=input_path,
            config_hash=config_hash,
        )
    
    def run(self, input_path: str | Path) -> PipelineResult:
        """
        Run the full analysis pipeline on input file.
        
        Args:
            input_path: Path to PDF, DOCX, or image file
            
        Returns:
            PipelineResult with extracted chart data
            
        Raises:
            PipelineError: If pipeline execution fails
            FileNotFoundError: If input file doesn't exist
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self._start_time = time.time()
        session = self._create_session(input_path)
        
        logger.info(f"Pipeline run started | session={session.session_id}")
        logger.info(f"Input: {input_path}")

        try:
            from .schemas.stage_outputs import Stage4Output

            # Stage 1: Ingestion (critical - must succeed)
            stage1_output = self.stages[0].process(input_path)
            logger.info(f"Stage 1 complete | images={stage1_output.total_images}")

            # Stage 2: Detection
            stage2_output = self.stages[1].process(stage1_output)
            logger.info(f"Stage 2 complete | charts={len(stage2_output.charts)}")

            if not stage2_output.charts:
                processing_time = time.time() - self._start_time
                logger.info("No charts detected -- returning empty result")
                return PipelineResult(
                    session=session,
                    charts=[],
                    summary="No charts detected in input document.",
                    processing_time_seconds=round(processing_time, 3),
                    model_versions={},
                )

            # Stage 3: Extraction
            stage3_output = self.stages[2].process(stage2_output)
            logger.info(f"Stage 3 complete | metadata={len(stage3_output.metadata)}")

            # Stage 4: Reasoning
            stage4_output = self.stages[3].process(stage3_output)
            logger.info(f"Stage 4 complete | refined={len(stage4_output.charts)}")

            # Stage 5: Reporting
            result = self.stages[4].process(stage4_output)

            processing_time = time.time() - self._start_time
            logger.info(
                f"Pipeline complete | session={session.session_id} | "
                f"charts={len(result.charts)} | elapsed={processing_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.exception(f"Pipeline failed | session={session.session_id} | error={e}")
            raise PipelineError(
                message=str(e),
                stage="unknown",
                recoverable=False,
            ) from e
    
    def run_stage(
        self,
        stage_number: int,
        input_data,
        session: Optional[SessionInfo] = None,
    ):
        """
        Run a single pipeline stage.
        
        Useful for debugging or partial processing.
        
        Args:
            stage_number: Stage to run (1-5)
            input_data: Input for that stage
            session: Optional session info
            
        Returns:
            Stage output
        """
        if stage_number < 1 or stage_number > 5:
            raise ValueError(f"Invalid stage number: {stage_number}")
        
        stage = self.stages[stage_number - 1]
        return stage.process(input_data)
