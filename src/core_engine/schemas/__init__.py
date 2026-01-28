"""
Schemas Package

Pydantic models for data validation and serialization.

NOTE: All enums are defined in enums.py - the Single Source of Truth.
"""

from .enums import (
    ChartType,
    StageStatus,
    PipelineStatus,
    TextRole,
    ElementType,
    InsightType,
    InputFileType,
    OutputFormat,
    QualityLevel,
    ConfidenceThreshold,
    ErrorCode,
    AxisType,
    AxisPosition,
)
from .common import (
    BoundingBox,
    Point,
    Color,
    SessionInfo,
)
from .stage_outputs import (
    CleanImage,
    Stage1Output,
    DetectedChart,
    Stage2Output,
    OCRText,
    ChartElement,
    RawMetadata,
    Stage3Output,
    DataPoint,
    DataSeries,
    RefinedChartData,
    Stage4Output,
    ChartInsight,
    FinalChartResult,
    PipelineResult,
)
from .qa_schemas import (
    QuestionType,
    ReasoningMethod,
    ChartRegion,
    ConfidenceLevel,
    PointReference,
    VisualGrounding,
    ReasoningStep,
    InferenceInfo,
    QAPairV2,
    ChartQASampleV2,
    ShardMetadataV2,
    ShardV2,
    QUESTION_TEMPLATES,
    QUESTION_DIFFICULTY,
)

__all__ = [
    # Enums (Single Source of Truth)
    "ChartType",
    "StageStatus",
    "PipelineStatus",
    "TextRole",
    "ElementType",
    "InsightType",
    "InputFileType",
    "OutputFormat",
    "QualityLevel",
    "ConfidenceThreshold",
    "ErrorCode",
    "AxisType",
    "AxisPosition",
    # Common types
    "BoundingBox",
    "Point",
    "Color",
    "SessionInfo",
    # Stage outputs
    "CleanImage",
    "Stage1Output",
    "DetectedChart",
    "Stage2Output",
    "OCRText",
    "ChartElement",
    "RawMetadata",
    "Stage3Output",
    "DataPoint",
    "DataSeries",
    "RefinedChartData",
    "Stage4Output",
    "ChartInsight",
    "FinalChartResult",
    "PipelineResult",
    # QA Schemas v2
    "QuestionType",
    "ReasoningMethod",
    "ChartRegion",
    "ConfidenceLevel",
    "PointReference",
    "VisualGrounding",
    "ReasoningStep",
    "InferenceInfo",
    "QAPairV2",
    "ChartQASampleV2",
    "ShardMetadataV2",
    "ShardV2",
    "QUESTION_TEMPLATES",
    "QUESTION_DIFFICULTY",
]
