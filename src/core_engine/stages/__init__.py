"""
Pipeline Stages Package

Each stage is a self-contained processor that transforms input to output.

Stages:
- Stage1Ingestion: PDF/Image -> Normalized Images
- Stage2Detection: Images -> Detected Chart Regions (YOLO)
- Stage3Extraction: Chart Images -> Raw Metadata (OCR + Geometry + Vectorization)
- Stage4Reasoning: Raw Metadata -> Refined Data (SLM)
- Stage5Reporting: Refined Data -> Final JSON Report
"""

from .base import BaseStage
from .s1_ingestion import Stage1Ingestion, IngestionConfig
from .s2_detection import Stage2Detection, DetectionConfig
from .s3_extraction import (
    Stage3Extraction,
    ExtractionConfig,
    ImagePreprocessor,
    PreprocessConfig,
    Skeletonizer,
    SkeletonConfig,
    Vectorizer,
    VectorizeConfig,
    OCREngine,
    OCRConfig,
    GeometricMapper,
    MapperConfig,
    ElementDetector,
    ElementDetectorConfig,
    ChartClassifier,
    ClassifierConfig,
)

__all__ = [
    # Base
    "BaseStage",
    # Stage 1
    "Stage1Ingestion",
    "IngestionConfig",
    # Stage 2
    "Stage2Detection",
    "DetectionConfig",
    # Stage 3
    "Stage3Extraction",
    "ExtractionConfig",
    "ImagePreprocessor",
    "PreprocessConfig",
    "Skeletonizer",
    "SkeletonConfig",
    "Vectorizer",
    "VectorizeConfig",
    "OCREngine",
    "OCRConfig",
    "GeometricMapper",
    "MapperConfig",
    "ElementDetector",
    "ElementDetectorConfig",
    "ChartClassifier",
    "ClassifierConfig",
]
