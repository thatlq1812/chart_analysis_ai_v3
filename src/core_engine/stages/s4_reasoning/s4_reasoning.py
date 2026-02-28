"""
Stage 4: Semantic Reasoning

Main orchestrator for the reasoning stage.
Combines geometric mapping, OCR correction, and SLM reasoning.

Pipeline:
1. GeometricValueMapper: Convert pixel → actual values
2. GeminiPromptBuilder: Build structured context
3. GeminiReasoningEngine: Call Gemini API for reasoning
4. Post-processing: Apply corrections and generate output
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .prompt_builder import GeminiPromptBuilder, PromptConfig, CanonicalContext
from .reasoning_engine import ReasoningEngine, ReasoningResult
from .router_engine import AIRouterEngine
from .value_mapper import GeometricValueMapper, ValueMapperConfig, MappingResult

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
    
    # Value Mapper config
    value_mapper: ValueMapperConfig = Field(default_factory=ValueMapperConfig)
    
    # Prompt Builder config
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    
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
        
        # Initialize components
        self.value_mapper = GeometricValueMapper(config.value_mapper)
        self.prompt_builder = GeminiPromptBuilder(config.prompt)
        
        # Initialize reasoning engine based on config
        self.engine: Optional[ReasoningEngine] = None
        self._initialize_engine()
        
        self.logger.info(
            f"Stage4Reasoning initialized | "
            f"engine={config.engine} | "
            f"value_mapping={config.enable_value_mapping}"
        )
    
    def _initialize_engine(self) -> None:
        """Initialize the reasoning engine based on config."""
        engine_type = self.config.engine.lower()

        if engine_type == "router":
            # Multi-provider routing with automatic fallback (recommended)
            self.engine = AIRouterEngine()
            self.logger.info("Stage4Reasoning | using AIRouterEngine (multi-provider)")

        elif engine_type == "gemini":
            self.engine = GeminiReasoningEngine(self.config.gemini)

        elif engine_type == "local_slm":
            # Use router to handle local_slm primary with gemini fallback
            self.logger.info(
                "Stage4Reasoning | local_slm requested, routing via AIRouterEngine"
            )
            self.engine = AIRouterEngine()

        elif engine_type == "rule_based":
            # No LLM -- pure rule-based fallback only
            self.engine = None

        else:
            self.logger.warning(
                f"Stage4Reasoning | unknown engine '{engine_type}' | "
                "falling back to AIRouterEngine"
            )
            self.engine = AIRouterEngine()
    
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
        
        Pipeline:
        1. Geometric Value Mapping (pixel → actual values)
        2. Build Canonical Context for prompting
        3. SLM Reasoning (Gemini API)
        4. Post-processing and validation
        
        Args:
            metadata: Raw metadata from Stage 3
        
        Returns:
            RefinedChartData with semantic understanding
        """
        chart_id = metadata.chart_id
        self.logger.debug(f"Processing chart | chart_id={chart_id}")
        
        # Step 1: Geometric Value Mapping
        mapped_series: List[DataSeries] = []
        if self.config.enable_value_mapping:
            # Reset mapper for new chart
            self.value_mapper = GeometricValueMapper(self.config.value_mapper)
            
            # Map metadata to series
            mapped_series = self.value_mapper.map_metadata_to_series(metadata)
            
            calibration = self.value_mapper.get_calibration_summary()
            self.logger.debug(
                f"Value mapping complete | chart_id={chart_id} | "
                f"calibrated={calibration['is_calibrated']} | "
                f"series={len(mapped_series)}"
            )
        
        # Step 2: Build Canonical Context
        context = self.prompt_builder.build_canonical_context(
            metadata=metadata,
            mapped_series=mapped_series,
        )
        
        # Step 3: Use engine if available
        if self.engine is not None and self.engine.is_available():
            # Build prompt with context
            prompt = self.prompt_builder.build_reasoning_prompt(
                metadata=metadata,
                mapped_series=mapped_series,
            )
            
            # Call reasoning engine
            result = self.engine.reason(metadata)
            
            if result.success:
                # Merge mapped series with reasoning result
                final_series = self._merge_series(mapped_series, result.series)
                
                return RefinedChartData(
                    chart_id=chart_id,
                    chart_type=metadata.chart_type,
                    title=result.title,
                    x_axis_label=result.x_axis_label,
                    y_axis_label=result.y_axis_label,
                    series=final_series,
                    description=result.description,
                    correction_log=[
                        f"{c.get('original', '?')} -> {c.get('corrected', '?')}"
                        for c in result.corrections
                    ],
                )
        
        # Fallback to rule-based extraction with mapped values
        return self._create_fallback_result(metadata, mapped_series)
    
    def _merge_series(
        self,
        mapped_series: List[DataSeries],
        reasoned_series: List[DataSeries],
    ) -> List[DataSeries]:
        """
        Merge mapped series with reasoned series.
        
        Uses reasoned names/labels but mapped values if available.
        
        Args:
            mapped_series: Series from geometric mapping
            reasoned_series: Series from SLM reasoning
        
        Returns:
            Merged series list
        """
        if not mapped_series:
            return reasoned_series
        if not reasoned_series:
            return mapped_series
        
        # Use reasoned series as base, but update values from mapped
        merged = []
        for i, r_series in enumerate(reasoned_series):
            if i < len(mapped_series):
                # Merge points: use reasoned labels, mapped values
                m_series = mapped_series[i]
                merged_points = []
                
                for j, r_point in enumerate(r_series.points):
                    if j < len(m_series.points):
                        # Use mapped value if it has higher confidence
                        m_point = m_series.points[j]
                        if m_point.confidence > r_point.confidence:
                            merged_points.append(DataPoint(
                                label=r_point.label or m_point.label,
                                value=m_point.value,
                                confidence=m_point.confidence,
                            ))
                        else:
                            merged_points.append(r_point)
                    else:
                        merged_points.append(r_point)
                
                merged.append(DataSeries(
                    name=r_series.name,
                    color=r_series.color or m_series.color,
                    points=merged_points,
                ))
            else:
                merged.append(r_series)
        
        return merged
    
    def _create_fallback_result(
        self,
        metadata: RawMetadata,
        mapped_series: Optional[List[DataSeries]] = None,
    ) -> RefinedChartData:
        """
        Create fallback result using rule-based extraction.
        
        Uses mapped series if available, otherwise extracts from elements.
        
        Args:
            metadata: Raw metadata from Stage 3
            mapped_series: Optional pre-mapped series from ValueMapper
        
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
        
        # Use mapped series if available
        series = mapped_series if mapped_series else []
        
        # If no mapped series, create basic series from elements
        if not series and metadata.elements:
            # Group by color
            color_groups: Dict[tuple, List] = {}
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
        description = self._generate_fallback_description(
            metadata.chart_type,
            title,
            series,
            x_label,
            y_label,
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
    
    def _generate_fallback_description(
        self,
        chart_type: ChartType,
        title: Optional[str],
        series: List[DataSeries],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> str:
        """Generate simple description without API."""
        parts = [f"This {chart_type.value} chart"]
        
        if title:
            parts[0] += f' titled "{title}"'
        
        # Count data points
        total_points = sum(len(s.points) for s in series)
        if total_points > 0:
            parts.append(f"contains {total_points} data points across {len(series)} series")
        
        if x_label or y_label:
            axis_parts = []
            if x_label:
                axis_parts.append(f"x-axis showing {x_label}")
            if y_label:
                axis_parts.append(f"y-axis showing {y_label}")
            parts.append(f"with {' and '.join(axis_parts)}")
        
        return ". ".join(parts) + "."
    
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
