"""
Stage 3: Structural Analysis (Extraction)

This module implements the Geo-SLM hybrid extraction approach:
1. Negative Image Preprocessing - Enhance structural contrast
2. Skeletonization - Topology-preserving thinning
3. Vectorization - RDP algorithm for piecewise linear representation
4. OCR Engine - Text extraction with role classification
5. Geometric Mapper - Coordinate space transformation
6. Element Detector - Bar, marker, pie slice detection
7. Chart Classifier - Type classification from features
8. ResNet18 Classifier - Deep learning based classification (94.66% accuracy)

Reference: docs/instruction_p2_research.md
"""

from .s3_extraction import Stage3Extraction, ExtractionConfig
from .preprocessor import ImagePreprocessor, PreprocessConfig
from .skeletonizer import Skeletonizer, SkeletonConfig
from .vectorizer import Vectorizer, VectorizeConfig
from .ocr_engine import OCREngine, OCRConfig
from .geometric_mapper import GeometricMapper, MapperConfig
from .element_detector import ElementDetector, ElementDetectorConfig
from .classifier import ChartClassifier, ClassifierConfig
from .resnet_classifier import ResNet18Classifier, create_resnet_classifier

__all__ = [
    # Main Stage
    "Stage3Extraction",
    "ExtractionConfig",
    # Submodules
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
    # Deep Learning Classifier
    "ResNet18Classifier",
    "create_resnet_classifier",
]
