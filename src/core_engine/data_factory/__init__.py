"""
Data Factory - Training Data Generation for Chart Analysis

This module provides tools for generating training data for SLM:
- QA Generator: Generate research-grade QA pairs using Gemini
- (Future) SLM Trainer: Fine-tune SLM on generated data

Note: This is NOT a pipeline stage. It's a downstream consumer
that uses Stage 1-4 outputs to generate training data.

Usage:
    from core_engine.data_factory import (
        ChartQAGeneratorV2,
        QAGeneratorConfig,
    )
    
    config = QAGeneratorConfig(api_key="...", model_name="gemini-3-flash-preview")
    generator = ChartQAGeneratorV2(config)
    
    samples = [{"image_path": "...", "chart_type": "line"}, ...]
    results = generator.generate_batch(samples)

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-28 | That Le | Initial data factory module |
"""

from .qa_generator import (
    ChartQAGeneratorV2,
    QAGeneratorConfig,
)

__all__ = [
    "ChartQAGeneratorV2",
    "QAGeneratorConfig",
]
