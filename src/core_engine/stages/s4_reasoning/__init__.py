"""
Stage 4: Semantic Reasoning

This module implements the SLM-based reasoning layer:
1. Geometric Value Mapping - Convert pixel coordinates to data values
2. OCR Error Correction - Fix common OCR mistakes using context
3. Legend-Color Association - Map colors to series names
4. Description Generation - Create academic-style summaries

Supports multiple backends:
- Gemini API (default for prototyping)
- Local SLM (Qwen/Llama for production)

Reference: docs/architecture/STAGE4_REASONING.md
"""

from .s4_reasoning import Stage4Reasoning, ReasoningConfig
from .reasoning_engine import ReasoningEngine, ReasoningResult
from .gemini_engine import GeminiReasoningEngine, GeminiConfig

__all__ = [
    # Main Stage
    "Stage4Reasoning",
    "ReasoningConfig",
    # Engines
    "ReasoningEngine",
    "ReasoningResult",
    "GeminiReasoningEngine",
    "GeminiConfig",
]
