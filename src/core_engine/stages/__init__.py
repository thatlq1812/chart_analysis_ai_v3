"""
Pipeline Stages Package

Each stage is a self-contained processor that transforms input to output.

Stages:
- Stage1Ingestion: PDF/Image -> Normalized Images
- Stage2Detection: Images -> Detected Chart Regions (YOLO)
- Stage3Extraction: Chart Images -> Raw Metadata (OCR + Geometry)
- Stage4Reasoning: Raw Metadata -> Refined Data (SLM)
- Stage5Reporting: Refined Data -> Final JSON Report
"""

from .base import BaseStage
from .s1_ingestion import Stage1Ingestion, IngestionConfig
from .s2_detection import Stage2Detection, DetectionConfig

__all__ = [
    "BaseStage",
    "Stage1Ingestion",
    "IngestionConfig",
    "Stage2Detection",
    "DetectionConfig",
]
