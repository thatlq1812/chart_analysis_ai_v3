"""
AI Task Types

Defines the set of AI tasks the system can route.
Used by AIRouter to select the correct provider and fallback chain.
"""

from enum import Enum


class TaskType(str, Enum):
    """
    AI tasks the pipeline needs to perform.

    Each task type maps to a fallback chain of providers defined in AIRouter.
    Simple, well-constrained tasks prefer local SLM (free, offline).
    Complex spatial/visual tasks prefer cloud providers (higher capability).
    """

    CHART_REASONING = "chart_reasoning"
    """Full chart analysis: OCR correction + value mapping + description."""

    OCR_CORRECTION = "ocr_correction"
    """Fix OCR misreads using chart context (axis labels, legend text)."""

    DESCRIPTION_GEN = "description_gen"
    """Generate an academic-style description of the chart."""

    DATA_VALIDATION = "data_validation"
    """Validate extracted data series against visual evidence."""
