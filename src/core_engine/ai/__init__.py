"""
AI Routing Layer

Multi-provider AI routing with automatic fallback chains.

Quick start:
    from core_engine.ai import AIRouter, TaskType

    router = AIRouter()
    response = await router.route(
        TaskType.CHART_REASONING,
        system_prompt=CHART_REASONING_SYSTEM,
        user_prompt=user_prompt,
    )
    print(response.content)
"""

from .adapters import AIResponse, BaseAIAdapter, GeminiAdapter, LocalSLMAdapter, OpenAIAdapter
from .exceptions import (
    AIAuthenticationError,
    AIInvalidResponseError,
    AIProviderError,
    AIProviderExhaustedError,
    AIRateLimitError,
    AITimeoutError,
)
from .prompts import (
    CHART_REASONING_SYSTEM,
    DATA_VALIDATION_SYSTEM,
    DESCRIPTION_GEN_SYSTEM,
    OCR_CORRECTION_SYSTEM,
    format_description_user,
    format_ocr_correction_user,
    format_reasoning_user,
)
from .router import AIRouter
from .task_types import TaskType

__all__ = [
    # Core routing
    "AIRouter",
    "TaskType",
    # Adapters
    "AIResponse",
    "BaseAIAdapter",
    "GeminiAdapter",
    "LocalSLMAdapter",
    "OpenAIAdapter",
    # Exceptions
    "AIProviderError",
    "AIProviderExhaustedError",
    "AIAuthenticationError",
    "AIRateLimitError",
    "AITimeoutError",
    "AIInvalidResponseError",
    # Prompts
    "CHART_REASONING_SYSTEM",
    "OCR_CORRECTION_SYSTEM",
    "DESCRIPTION_GEN_SYSTEM",
    "DATA_VALIDATION_SYSTEM",
    "format_reasoning_user",
    "format_ocr_correction_user",
    "format_description_user",
]
