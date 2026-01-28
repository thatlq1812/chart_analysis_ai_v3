"""
Stage 4: Semantic Reasoning

Main orchestrator for the reasoning stage.
Combines geometric mapping, OCR correction, and SLM reasoning.
"""

import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from ...exceptions import StageProcessingError
from ...schemas.common import Color
from ...schemas.enums import ChartType
from ...schemas.stage_outputs import (
    DataPoint,
    DataSeries,
    RawMetadata,
    RefinedChartData,
    Stage3Output,
    Stage4Output,
)
from ..base import BaseStage
from .gemini_engine import GeminiConfig, GeminiReasoningEngine
from .reasoning_engine import ReasoningEngine, ReasoningResult

logger = logging.getLogger(__name__)


class ReasoningConfig(BaseModel):
    """Configuration for Stage 4: Reasoning."""
    
    # Engine selection
    engine: str = Field(
        default="gemini",
        description="Reasoning engine: 'gemini', 'local_slm', or 'rule_based'"
    )
    
    # Gemini config (if using gemini engine)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    
    # Local SLM config (for future use)
    local_slm_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model ID for local SLM"
    )
    local_slm_device: str = Field(default="auto")
    
    # Processing options
    enable_ocr_correction: bool = Field(
        default=True,
        description="Enable OCR error correction"
    )
    enable_value_mapping: bool = Field(
        default=True,
        description="Enable geometric value mapping"
    )
    enable_description: bool = Field(
        default=True,
        description="Enable description generation"
    )
    
    # Fallback options
    use_fallback_on_error: bool = Field(
        default=True,
        description="Use rule-based fallback if engine fails"
    )


class Stage4Reasoning(BaseStage):
    """
    Stage 4 Orchestrator: Semantic Reasoning
    
    Applies SLM-based reasoning to refine chart data:
    1. OCR error correction
    2. Value mapping (geometric -> actual)
    3. Legend-color association
    4. Description generation
    
    Example:
        config = ReasoningConfig()
        stage = Stage4Reasoning(config)
        result = stage.process(stage3_output)
    """
    
    def __init__(self, config: Optional[ReasoningConfig] = None):
        """
        Initialize Stage 4: Reasoning.
        
        Args:
            config: Reasoning configuration
        """
        config = config or ReasoningConfig()
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize reasoning engine based on config
        self.engine: Optional[ReasoningEngine] = None
        self._initialize_engine()
        
        self.logger.info(
            f"Stage4Reasoning initialized | "
            f"engine={config.engine}"
        )
    
    def _initialize_engine(self):
        """Initialize the reasoning engine based on config."""
        engine_type = self.config.engine.lower()
        
        if engine_type == "gemini":
            self.engine = GeminiReasoningEngine(self.config.gemini)
        elif engine_type == "local_slm":
            # TODO: Implement local SLM engine
            self.logger.warning("Local SLM not yet implemented, using Gemini fallback")
            self.engine = GeminiReasoningEngine(self.config.gemini)
        elif engine_type == "rule_based":
            # No engine needed - use pure rule-based fallback
            self.engine = None
        else:
            self.logger.warning(f"Unknown engine: {engine_type}, using Gemini")
            self.engine = GeminiReasoningEngine(self.config.gemini)
    
    def process(self, input_data: Stage3Output) -> Stage4Output:
        """
        Process Stage 3 output and apply semantic reasoning.
        
        Args:
            input_data: Stage3Output with raw metadata
        
        Returns:
            Stage4Output with refined chart data
        
        Raises:
            StageProcessingError: If processing fails critically
        """
        if not self.validate_input(input_data):
            raise StageProcessingError(
                "Stage4Reasoning",
                f"Invalid input: expected Stage3Output, got {type(input_data)}"
            )
        
        session = input_data.session
        self.logger.info(
            f"Reasoning started | session={session.session_id} | "
            f"charts={len(input_data.metadata)}"
        )
        
        refined_charts = []
        
        for metadata in input_data.metadata:
            try:
                refined = self._process_single_chart(metadata)
                refined_charts.append(refined)
                
            except Exception as e:
                self.logger.error(
                    f"Chart reasoning failed | chart_id={metadata.chart_id} | "
                    f"error={str(e)}"
                )
                
                if self.config.use_fallback_on_error:
                    fallback = self._create_fallback_result(metadata)
                    refined_charts.append(fallback)
                else:
                    raise StageProcessingError(
                        "Stage4Reasoning",
                        f"Failed to process chart {metadata.chart_id}: {e}"
                    ) from e
        
        self.logger.info(
            f"Reasoning complete | session={session.session_id} | "
            f"processed={len(refined_charts)}"
        )
        
        return Stage4Output(
            session=session,
            charts=refined_charts,
        )
    
    def validate_input(self, input_data) -> bool:
        """Validate input is Stage3Output."""
        return isinstance(input_data, Stage3Output)
    
    def _process_single_chart(self, metadata: RawMetadata) -> RefinedChartData:
        """
        Process a single chart through reasoning pipeline.
        
        Args:
            metadata: Raw metadata from Stage 3
        
        Returns:
            RefinedChartData with semantic understanding
        """
        chart_id = metadata.chart_id
        self.logger.debug(f"Processing chart | chart_id={chart_id}")
        
        # Use engine if available
        if self.engine is not None and self.engine.is_available():
            result = self.engine.reason(metadata)
            
            if result.success:
                return RefinedChartData(
                    chart_id=chart_id,
                    chart_type=metadata.chart_type,
                    title=result.title,
                    x_axis_label=result.x_axis_label,
                    y_axis_label=result.y_axis_label,
                    series=result.series,
                    description=result.description,
                    correction_log=[
                        f"{c.get('original', '?')} -> {c.get('corrected', '?')}"
                        for c in result.corrections
                    ],
                )
        
        # Fallback to rule-based extraction
        return self._create_fallback_result(metadata)
    
    def _create_fallback_result(self, metadata: RawMetadata) -> RefinedChartData:
        """
        Create fallback result using rule-based extraction.
        
        Args:
            metadata: Raw metadata from Stage 3
        
        Returns:
            RefinedChartData with basic extraction
        """
        self.logger.debug(f"Using fallback for chart | chart_id={metadata.chart_id}")
        
        # Extract text by role
        title = None
        x_label = None
        y_label = None
        
        for text in metadata.texts:
            if text.role == "title" and not title:
                title = text.text
            elif text.role == "xlabel" and not x_label:
                x_label = text.text
            elif text.role == "ylabel" and not y_label:
                y_label = text.text
        
        # Create basic series from elements
        series = []
        if metadata.elements:
            # Group by color
            color_groups: dict = {}
            for elem in metadata.elements:
                if elem.color:
                    key = (elem.color.r, elem.color.g, elem.color.b)
                else:
                    key = (0, 0, 0)
                
                if key not in color_groups:
                    color_groups[key] = []
                color_groups[key].append(elem)
            
            # Create series for each color group
            for i, (color_key, elements) in enumerate(color_groups.items()):
                points = [
                    DataPoint(
                        label=f"Point {j+1}",
                        value=float(elem.center.y),  # Use y-coordinate as raw value
                        confidence=0.5,
                    )
                    for j, elem in enumerate(elements)
                ]
                
                series.append(DataSeries(
                    name=f"Series {i+1}",
                    color=Color(r=color_key[0], g=color_key[1], b=color_key[2]),
                    points=points,
                ))
        
        # Generate basic description
        description = (
            f"This {metadata.chart_type.value} chart"
            + (f' titled "{title}"' if title else "")
            + f" contains {len(metadata.elements)} elements"
            + f" and {len(metadata.texts)} text regions."
        )
        
        return RefinedChartData(
            chart_id=metadata.chart_id,
            chart_type=metadata.chart_type,
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            series=series,
            description=description,
            correction_log=["Used rule-based fallback"],
        )
    
    def process_single(
        self,
        metadata: RawMetadata,
        image_path: Optional[Path] = None,
    ) -> RefinedChartData:
        """
        Process a single chart directly (for testing/standalone use).
        
        Args:
            metadata: Raw metadata from Stage 3
            image_path: Optional path to image for vision models
        
        Returns:
            RefinedChartData with semantic understanding
        """
        return self._process_single_chart(metadata)
