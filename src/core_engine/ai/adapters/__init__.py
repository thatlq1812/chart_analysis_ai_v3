"""
AI Adapters Package

Exports adapter classes and the base interface.
"""

from .base import AIResponse, BaseAIAdapter
from .gemini_adapter import GeminiAdapter
from .local_slm_adapter import LocalSLMAdapter
from .openai_adapter import OpenAIAdapter

__all__ = [
    "AIResponse",
    "BaseAIAdapter",
    "GeminiAdapter",
    "LocalSLMAdapter",
    "OpenAIAdapter",
]
