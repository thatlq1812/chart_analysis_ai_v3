"""
Pipeline Stages Package

Each stage is a self-contained processor that transforms input to output.

Core Pipeline Stages (runtime):
- Stage1Ingestion: PDF/Image -> Normalized Images
- Stage2Detection: Images -> Detected Chart Regions (YOLO)
- Stage3Extraction: Chart Images -> Raw Metadata (VLM extraction + chart type classification)
- Stage4Reasoning: Raw Metadata -> Refined Data (SLM)

Note: QA Generation is NOT a pipeline stage. It's a data factory tool
for generating SLM training data. See `core_engine.data_factory` module.
"""

from .base import BaseStage
from .s1_ingestion import Stage1Ingestion, IngestionConfig
from .s2_detection import Stage2Detection, DetectionConfig
from .s3_extraction import (
    Stage3Extraction,
    ExtractionConfig,
    BaseChartExtractor,
    BackendType,
    DeplotExtractor,
    MatchaExtractor,
    Pix2StructBaselineExtractor,
    SVLMExtractor,
    create_extractor,
    EfficientNetClassifier,
    create_efficientnet_classifier,
)
from .s4_reasoning import (
    Stage4Reasoning,
    ReasoningConfig,
    GeminiReasoningEngine,
    GeminiConfig,
    AIRouterEngine,
)
from .s5_reporting import Stage5Reporting, ReportingConfig

__all__ = [
    # Base
    "BaseStage",
    # Stage 1: Ingestion
    "Stage1Ingestion",
    "IngestionConfig",
    # Stage 2: Detection
    "Stage2Detection",
    "DetectionConfig",
    # Stage 3: Extraction
    "Stage3Extraction",
    "ExtractionConfig",
    "BaseChartExtractor",
    "BackendType",
    "DeplotExtractor",
    "MatchaExtractor",
    "Pix2StructBaselineExtractor",
    "SVLMExtractor",
    "create_extractor",
    "EfficientNetClassifier",
    "create_efficientnet_classifier",
    # Stage 4: Reasoning
    "Stage4Reasoning",
    "ReasoningConfig",
    "GeminiReasoningEngine",
    "GeminiConfig",
    "AIRouterEngine",
    # Stage 5: Reporting
    "Stage5Reporting",
    "ReportingConfig",
]
