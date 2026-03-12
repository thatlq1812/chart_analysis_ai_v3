"""
Stage 1: Ingestion & Sanitation Package

Public API for Stage 1.

Usage:
    from src.core_engine.stages.s1_ingestion import Stage1Ingestion, IngestionConfig
"""

from .config import IngestionConfig
from .ingestion import Stage1Ingestion

__all__ = ["Stage1Ingestion", "IngestionConfig"]
