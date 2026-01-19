"""
Geo-SLM Chart Analysis - Core Engine

A hybrid AI system for extracting structured data from chart images.

Example:
    from core_engine import ChartAnalysisPipeline
    
    pipeline = ChartAnalysisPipeline.from_config()
    result = pipeline.run("chart.png")
"""

from .pipeline import ChartAnalysisPipeline
from .exceptions import (
    ChartAnalysisError,
    PipelineError,
    StageInputError,
    StageProcessingError,
    ConfigurationError,
    ModelError,
)

__version__ = "3.0.0"
__all__ = [
    "ChartAnalysisPipeline",
    "ChartAnalysisError",
    "PipelineError",
    "StageInputError",
    "StageProcessingError",
    "ConfigurationError",
    "ModelError",
]
