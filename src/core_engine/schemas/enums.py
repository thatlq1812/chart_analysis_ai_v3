"""
Shared Enumerations and Constants

THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR ALL ENUMS AND CONSTANTS.
All agents (Core, Interface, Tests) MUST import from this file.
NEVER hard-code string literals in processing logic.

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-19 | That Le | Initial shared vocabulary |
"""

from enum import Enum


# =============================================================================
# CHART TYPES
# =============================================================================

class ChartType(str, Enum):
    """
    Supported chart types for detection and analysis.
    
    Usage:
        from core_engine.schemas.enums import ChartType
        
        if chart.chart_type == ChartType.BAR:
            process_bar_chart(chart)
    """
    
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    DONUT = "donut"
    UNKNOWN = "unknown"


# =============================================================================
# PIPELINE STATUS
# =============================================================================

class StageStatus(str, Enum):
    """
    Status of a pipeline stage execution.
    
    State transitions:
        PENDING -> PROCESSING -> COMPLETED
                             -> FAILED
                             -> SKIPPED
    """
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(str, Enum):
    """Overall pipeline execution status."""
    
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Some stages failed but got partial results
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# OCR TEXT ROLES
# =============================================================================

class TextRole(str, Enum):
    """
    Role/purpose of detected text in a chart.
    
    Used by Stage 3 (Extraction) to classify OCR results.
    """
    
    TITLE = "title"
    SUBTITLE = "subtitle"
    X_AXIS_LABEL = "x_axis_label"
    Y_AXIS_LABEL = "y_axis_label"
    X_TICK = "x_tick"           # Tick labels on X axis
    Y_TICK = "y_tick"           # Tick labels on Y axis
    LEGEND = "legend"
    DATA_LABEL = "data_label"   # Value labels on data points
    ANNOTATION = "annotation"
    UNKNOWN = "unknown"


# =============================================================================
# CHART ELEMENTS
# =============================================================================

class ElementType(str, Enum):
    """
    Type of visual element detected in a chart.
    
    Used by Stage 3 (Extraction) for element detection.
    """
    
    BAR = "bar"
    LINE = "line"
    POINT = "point"
    SLICE = "slice"         # Pie/donut slice
    AREA = "area"
    GRID_LINE = "grid_line"
    AXIS = "axis"
    LEGEND_ITEM = "legend_item"


# =============================================================================
# INSIGHT TYPES
# =============================================================================

class InsightType(str, Enum):
    """
    Type of insight generated in Stage 5 (Reporting).
    
    Each insight has a specific structure and purpose.
    """
    
    TREND = "trend"             # Increasing/decreasing patterns
    COMPARISON = "comparison"   # Max/min/relative comparisons
    ANOMALY = "anomaly"         # Outliers, unusual values
    SUMMARY = "summary"         # General description
    CORRELATION = "correlation" # Relationship between variables


# =============================================================================
# FILE TYPES
# =============================================================================

class InputFileType(str, Enum):
    """Supported input file types for Stage 1 (Ingestion)."""
    
    PDF = "pdf"
    DOCX = "docx"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"


class OutputFormat(str, Enum):
    """Supported output formats for Stage 5 (Reporting)."""
    
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


# =============================================================================
# QUALITY INDICATORS
# =============================================================================

class QualityLevel(str, Enum):
    """Image or detection quality level."""
    
    HIGH = "high"           # Confidence >= 0.8
    MEDIUM = "medium"       # Confidence 0.5 - 0.8
    LOW = "low"             # Confidence < 0.5
    INVALID = "invalid"     # Failed validation


class ConfidenceThreshold:
    """
    Standard confidence thresholds used across the pipeline.
    
    Usage:
        from core_engine.schemas.enums import ConfidenceThreshold
        
        if detection.confidence >= ConfidenceThreshold.DETECTION_MIN:
            accept_detection(detection)
    """
    
    DETECTION_MIN: float = 0.5      # Minimum for chart detection
    DETECTION_HIGH: float = 0.8     # High confidence detection
    OCR_MIN: float = 0.6            # Minimum for OCR text
    OCR_HIGH: float = 0.9           # High confidence OCR
    VALUE_EXTRACTION: float = 0.7   # Value mapping confidence


# =============================================================================
# ERROR CODES
# =============================================================================

class ErrorCode(str, Enum):
    """
    Standardized error codes for pipeline failures.
    
    Format: STAGE_ERROR_TYPE
    """
    
    # Stage 1 errors
    S1_FILE_NOT_FOUND = "s1_file_not_found"
    S1_UNSUPPORTED_FORMAT = "s1_unsupported_format"
    S1_CORRUPTED_FILE = "s1_corrupted_file"
    S1_LOW_QUALITY = "s1_low_quality"
    
    # Stage 2 errors
    S2_MODEL_NOT_LOADED = "s2_model_not_loaded"
    S2_NO_DETECTIONS = "s2_no_detections"
    S2_INFERENCE_FAILED = "s2_inference_failed"
    
    # Stage 3 errors
    S3_OCR_FAILED = "s3_ocr_failed"
    S3_CLASSIFICATION_FAILED = "s3_classification_failed"
    S3_ELEMENT_DETECTION_FAILED = "s3_element_detection_failed"
    
    # Stage 4 errors
    S4_SLM_NOT_LOADED = "s4_slm_not_loaded"
    S4_REASONING_TIMEOUT = "s4_reasoning_timeout"
    S4_VALUE_MAPPING_FAILED = "s4_value_mapping_failed"
    
    # Stage 5 errors
    S5_VALIDATION_FAILED = "s5_validation_failed"
    S5_REPORT_GENERATION_FAILED = "s5_report_generation_failed"
    
    # General errors
    UNKNOWN_ERROR = "unknown_error"
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT = "timeout"


# =============================================================================
# AXIS TYPES
# =============================================================================

class AxisType(str, Enum):
    """Type of chart axis."""
    
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    LOGARITHMIC = "logarithmic"


class AxisPosition(str, Enum):
    """Position of axis in chart."""
    
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
