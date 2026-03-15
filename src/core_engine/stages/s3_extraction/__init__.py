"""
Stage 3: Structural Analysis (Extraction)

VLM-based pluggable chart-to-table extraction with four backends:
    - DeplotExtractor          : google/deplot (recommended default)
    - MatchaExtractor          : google/matcha-base (enhanced math reasoning)
    - Pix2StructBaselineExtractor : google/pix2struct-base (ablation baseline)
    - SVLMExtractor            : Qwen/Qwen2-VL-2B-Instruct (zero-shot large VLM)

Reference: docs/architecture/STAGE3_EXTRACTION.md
"""

from .s3_extraction import ExtractionConfig, Stage3Extraction
from .extractors import (
    BackendType,
    BaseChartExtractor,
    DeplotExtractor,
    MatchaExtractor,
    Pix2StructBaselineExtractor,
    SVLMExtractor,
    create_extractor,
)
from .resnet_classifier import (
    EfficientNetClassifier,
    ResNet18Classifier,
    create_efficientnet_classifier,
    create_resnet_classifier,
)

# Backward compatibility: Pix2StructExtractor is an alias for DeplotExtractor
from .pix2struct_extractor import Pix2StructExtractor

__all__ = [
    # Main stage
    "Stage3Extraction",
    "ExtractionConfig",
    # Extractor backends
    "BaseChartExtractor",
    "BackendType",
    "DeplotExtractor",
    "MatchaExtractor",
    "Pix2StructBaselineExtractor",
    "SVLMExtractor",
    "create_extractor",
    # Deep learning classifiers
    "EfficientNetClassifier",
    "create_efficientnet_classifier",
    "ResNet18Classifier",
    "create_resnet_classifier",
    # Backward compat
    "Pix2StructExtractor",
]
