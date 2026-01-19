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
        Initialize all pipeline stages.
        
        Returns:
            List of stage processor instances
        """
        # TODO: Import and initialize actual stage classes
        # For now, return empty list as placeholder
        stages = []
        
        if self.config.pipeline.stages.ingestion.enabled:
            # from .stages import Stage1Ingestion
            # stages.append(Stage1Ingestion(self.config.ingestion))
            pass
            
        if self.config.pipeline.stages.detection.enabled:
            # from .stages import Stage2Detection
            # stages.append(Stage2Detection(self.config.detection))
            pass
            
        if self.config.pipeline.stages.extraction.enabled:
            # from .stages import Stage3Extraction
            # stages.append(Stage3Extraction(self.config.extraction))
            pass
            
        if self.config.pipeline.stages.reasoning.enabled:
            # from .stages import Stage4Reasoning
            # stages.append(Stage4Reasoning(self.config.reasoning))
            pass
            
        if self.config.pipeline.stages.reporting.enabled:
            # from .stages import Stage5Reporting
            # stages.append(Stage5Reporting(self.config.reporting))
            pass
        
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
        
        logger.info(f"Starting pipeline run: {session.session_id}")
        logger.info(f"Input: {input_path}")
        
        try:
            # Stage 1: Ingestion
            # stage1_output = self._run_stage1(input_path, session)
            
            # Stage 2: Detection
            # stage2_output = self._run_stage2(stage1_output)
            
            # Stage 3: Extraction
            # stage3_output = self._run_stage3(stage2_output)
            
            # Stage 4: Reasoning
            # stage4_output = self._run_stage4(stage3_output)
            
            # Stage 5: Reporting
            # result = self._run_stage5(stage4_output)
            
            # TODO: Implement actual stage execution
            # For now, return placeholder result
            processing_time = time.time() - self._start_time
            
            result = PipelineResult(
                session=session,
                charts=[],
                summary="Pipeline execution placeholder - stages not yet implemented",
                processing_time_seconds=processing_time,
                model_versions={
                    "yolo": "placeholder",
                    "ocr": "placeholder",
                    "slm": "placeholder",
                },
            )
            
            logger.info(f"Pipeline completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
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
