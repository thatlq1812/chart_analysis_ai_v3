"""
pix2struct_extractor -- backward-compatibility re-export.

The chart extraction backends have been consolidated into extractors.py.
Import from there directly:

    from .extractors import (
        DeplotExtractor,
        MatchaExtractor,
        Pix2StructBaselineExtractor,
        SVLMExtractor,
        BaseChartExtractor,
        BackendType,
        create_extractor,
    )

This module is kept so that existing code referencing
from .pix2struct_extractor import Pix2StructExtractor
continues to work without changes.
"""

from .extractors import DeplotExtractor as Pix2StructExtractor  # noqa: F401
from .extractors import (  # noqa: F401
    BackendType,
    BaseChartExtractor,
    DeplotExtractor,
    MatchaExtractor,
    Pix2StructBaselineExtractor,
    SVLMExtractor,
    _build_records,
    _is_numeric,
    _parse_deplot_output,
    create_extractor,
)

__all__ = [
    "Pix2StructExtractor",  # backward compat alias for DeplotExtractor
    "BaseChartExtractor",
    "BackendType",
    "DeplotExtractor",
    "MatchaExtractor",
    "Pix2StructBaselineExtractor",
    "SVLMExtractor",
    "create_extractor",
]
