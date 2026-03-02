"""
Stage Output Schemas

Pydantic models for data flowing between pipeline stages.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .common import BoundingBox, Color, Point, SessionInfo
from .enums import ChartType, InsightType, TextRole, ElementType  # Single Source of Truth


# ============================================================================
# Stage 1: Ingestion Output
# ============================================================================

class CleanImage(BaseModel):
    """Single cleaned and normalized image from Stage 1."""
    
    model_config = ConfigDict(frozen=True)
    
    image_path: Path = Field(..., description="Path to normalized image file")
    original_path: Path = Field(..., description="Path to source file")
    page_number: int = Field(default=1, ge=1, description="Page number in source")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    is_grayscale: bool = Field(default=False, description="Whether image is grayscale")


class Stage1Output(BaseModel):
    """Output from Stage 1: Ingestion & Sanitation."""
    
    session: SessionInfo
    images: List[CleanImage] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def total_images(self) -> int:
        """Total number of processed images."""
        return len(self.images)


# ============================================================================
# Stage 2: Detection Output
# ============================================================================

class DetectedChart(BaseModel):
    """Single detected chart from Stage 2."""
    
    model_config = ConfigDict(frozen=True)
    
    chart_id: str = Field(..., description="Unique chart identifier")
    source_image: Path = Field(..., description="Source image path")
    cropped_path: Path = Field(..., description="Path to cropped chart image")
    bbox: BoundingBox = Field(..., description="Detection bounding box")
    page_number: int = Field(default=1, ge=1, description="Source page number")


class Stage2Output(BaseModel):
    """Output from Stage 2: Detection & Localization."""
    
    session: SessionInfo
    charts: List[DetectedChart] = Field(default_factory=list)
    total_detected: int = Field(default=0, ge=0)
    skipped_low_confidence: int = Field(default=0, ge=0)
    
    @computed_field
    @property
    def has_charts(self) -> bool:
        """Whether any charts were detected."""
        return len(self.charts) > 0


# ============================================================================
# Stage 3: Extraction Output
# ============================================================================

class OCRText(BaseModel):
    """Extracted text element from OCR."""
    
    text: str = Field(..., description="Extracted text content")
    bbox: BoundingBox = Field(..., description="Text location")
    confidence: float = Field(..., ge=0, le=1, description="OCR confidence")
    role: Optional[str] = Field(
        None,
        description="Detected role: title, xlabel, ylabel, legend, value",
    )


class ChartElement(BaseModel):
    """Detected chart element (bar, point, slice, etc.)."""
    
    element_type: str = Field(
        ...,
        description="Element type: bar, point, slice, line, area",
    )
    bbox: BoundingBox = Field(..., description="Element bounding box")
    center: Point = Field(..., description="Element center point")
    color: Optional[Color] = Field(None, description="Dominant color")
    area_pixels: Optional[int] = Field(None, ge=0, description="Area in pixels")


class AxisInfo(BaseModel):
    """Detected axis information."""
    
    x_axis_detected: bool = Field(default=False)
    y_axis_detected: bool = Field(default=False)
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_scale_factor: Optional[float] = None  # pixels per unit
    y_scale_factor: Optional[float] = None
    x_calibration_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Confidence in X-axis calibration (R-squared)"
    )
    y_calibration_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Confidence in Y-axis calibration (R-squared)"
    )
    x_outliers_removed: int = Field(
        default=0, ge=0,
        description="Number of outlier tick labels removed during calibration"
    )
    y_outliers_removed: int = Field(
        default=0, ge=0,
        description="Number of outlier tick labels removed during calibration"
    )


class ExtractionConfidence(BaseModel):
    """Confidence scores for Stage 3 extraction components."""
    
    classification_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Chart type classification confidence"
    )
    ocr_mean_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Mean OCR confidence across all texts"
    )
    axis_calibration_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Combined axis calibration confidence"
    )
    element_detection_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Confidence in element detection"
    )
    overall_confidence: float = Field(
        default=0.0, ge=0, le=1,
        description="Weighted overall extraction confidence"
    )
    
    @classmethod
    def compute_overall(
        cls,
        classification: float,
        ocr: float,
        axis: float,
        elements: float,
        weights: Tuple[float, float, float, float] = (0.3, 0.25, 0.25, 0.2),
    ) -> "ExtractionConfidence":
        """Compute overall confidence from components."""
        overall = (
            weights[0] * classification +
            weights[1] * ocr +
            weights[2] * axis +
            weights[3] * elements
        )
        return cls(
            classification_confidence=classification,
            ocr_mean_confidence=ocr,
            axis_calibration_confidence=axis,
            element_detection_confidence=elements,
            overall_confidence=overall,
        )


class RawMetadata(BaseModel):
    """Raw extracted data from Stage 3."""
    
    chart_id: str = Field(..., description="Chart identifier from Stage 2")
    chart_type: ChartType = Field(..., description="Classified chart type")
    texts: List[OCRText] = Field(default_factory=list)
    elements: List[ChartElement] = Field(default_factory=list)
    axis_info: Optional[AxisInfo] = None
    confidence: Optional[ExtractionConfidence] = Field(
        default=None,
        description="Extraction confidence scores for Stage 4 reasoning"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Extraction warnings (low confidence, missing data, etc.)"
    )


class Stage3Output(BaseModel):
    """Output from Stage 3: Structural Analysis."""
    
    session: SessionInfo
    metadata: List[RawMetadata] = Field(default_factory=list)


# ============================================================================
# Stage 4: Reasoning Output
# ============================================================================

class DataPoint(BaseModel):
    """Single data point with actual value."""
    
    label: str = Field(..., description="X-axis label or category")
    value: float = Field(..., description="Numeric value")
    unit: Optional[str] = Field(None, description="Value unit if detected")
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Extraction confidence",
    )


class DataSeries(BaseModel):
    """A series of data points (corresponds to one legend item)."""
    
    name: str = Field(..., description="Series name from legend")
    color: Optional[Color] = Field(None, description="Series color")
    points: List[DataPoint] = Field(default_factory=list)
    
    @computed_field
    @property
    def count(self) -> int:
        """Number of data points in series."""
        return len(self.points)


class RefinedChartData(BaseModel):
    """Refined data after SLM reasoning."""
    
    chart_id: str = Field(..., description="Chart identifier")
    chart_type: ChartType = Field(..., description="Chart type")
    title: Optional[str] = Field(None, description="Chart title")
    x_axis_label: Optional[str] = Field(None, description="X-axis label")
    y_axis_label: Optional[str] = Field(None, description="Y-axis label")
    series: List[DataSeries] = Field(default_factory=list)
    description: str = Field(..., description="Academic-style description")
    correction_log: List[str] = Field(
        default_factory=list,
        description="OCR corrections applied",
    )


class Stage4Output(BaseModel):
    """Output from Stage 4: Semantic Reasoning."""
    
    session: SessionInfo
    charts: List[RefinedChartData] = Field(default_factory=list)


# ============================================================================
# Stage 5: Reporting Output (Final)
# ============================================================================

class ChartInsight(BaseModel):
    """Generated insight about the chart."""
    
    insight_type: str = Field(
        ...,
        description="Type: trend, comparison, anomaly, summary",
    )
    text: str = Field(..., description="Insight text")
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Insight confidence",
    )


class FinalChartResult(BaseModel):
    """Final output for a single chart."""
    
    chart_id: str = Field(..., description="Chart identifier")
    chart_type: ChartType = Field(..., description="Chart type")
    title: Optional[str] = Field(None, description="Chart title")
    data: RefinedChartData = Field(..., description="Extracted chart data")
    insights: List[ChartInsight] = Field(default_factory=list)
    source_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Traceability information",
    )


class PipelineResult(BaseModel):
    """
    Final pipeline output from Stage 5.
    
    Contains all extracted charts and metadata.
    """
    
    session: SessionInfo
    charts: List[FinalChartResult] = Field(default_factory=list)
    summary: str = Field(..., description="Overall processing summary")
    processing_time_seconds: float = Field(..., ge=0)
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Versions of models used",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings from data checks",
    )
    
    @computed_field
    @property
    def total_charts(self) -> int:
        """Total number of charts extracted."""
        return len(self.charts)
    
    @computed_field
    @property
    def chart_types_summary(self) -> Dict[str, int]:
        """Count of each chart type."""
        counts: Dict[str, int] = {}
        for chart in self.charts:
            chart_type = chart.chart_type.value
            counts[chart_type] = counts.get(chart_type, 0) + 1
        return counts
