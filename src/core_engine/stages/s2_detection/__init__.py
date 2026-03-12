"""
Stage 2: Detection & Localization package.

Provides a config-driven, adapter-based chart detection stage.

Registered adapters (populated on import):
    'yolov8'  - Ultralytics YOLOv8 (default)
    'yolov11' - Ultralytics YOLO11 (same API, different weights)
    'mock'    - Deterministic stub for testing

Public API:
    Stage2Detection  - Main stage class (drop-in replacement for legacy)
    DetectionConfig  - Stage configuration model
"""

# Import adapters first to trigger @register() decorators
from . import adapters  # noqa: F401

from .config import DetectionConfig
from .detection import Stage2Detection

__all__ = [
    "Stage2Detection",
    "DetectionConfig",
]
