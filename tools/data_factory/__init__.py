"""
Data Factory - Automated Chart Dataset Collection

This module provides tools for collecting chart images from multiple sources:
- Arxiv: Academic papers with charts
- Google Search: Web images of charts
- Roboflow: Pre-annotated datasets
- Synthetic: Generated charts using matplotlib

Usage:
    python -m tools.data_factory.main run-all --limit 100
    python -m tools.data_factory.main hunt --source arxiv --limit 50
    python -m tools.data_factory.main mine --input data/raw_pdfs
"""

from .config import DataFactoryConfig
from .schemas import ArxivPaper, ChartImage, DataManifest

__all__ = [
    "DataFactoryConfig",
    "ArxivPaper",
    "ChartImage",
    "DataManifest",
]
