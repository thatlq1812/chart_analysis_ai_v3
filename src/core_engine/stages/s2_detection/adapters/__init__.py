"""
Stage 2 detection adapters.

All adapters are imported here to trigger @register() decorators
so that AdapterRegistry is populated when the package is loaded.
"""

from .base import BaseDetectionAdapter, RawDetection
from .yolov8 import YOLOv8Adapter, YOLOv11Adapter
from .mock import MockDetectionAdapter

__all__ = [
    "BaseDetectionAdapter",
    "RawDetection",
    "YOLOv8Adapter",
    "YOLOv11Adapter",
    "MockDetectionAdapter",
]
