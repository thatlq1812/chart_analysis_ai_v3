"""
Benchmark suites package.

Auto-imported by BenchmarkRunner to register all suites in REGISTRY.
"""

from . import vlm_extraction
from . import ocr_quality
from . import classifier
from . import slm_reasoning

__all__ = ["vlm_extraction", "ocr_quality", "classifier", "slm_reasoning"]
