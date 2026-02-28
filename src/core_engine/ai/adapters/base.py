"""
Base AI Adapter Interface

Defines the contract that all AI provider adapters must implement.
Pipeline code interacts ONLY with BaseAIAdapter and AIResponse --
never with provider-specific SDKs directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AIResponse:
    """
    Standardized response from any AI provider.

    All adapter implementations MUST return this type.
    Never return raw provider SDK response objects to pipeline code.

    Attributes:
        content: The primary text output from the model
        model_used: Exact model identifier used (e.g. "gemini-2.0-flash")
        provider: Provider ID (e.g. "gemini", "openai", "local_slm")
        confidence: Self-reported or estimated confidence in [0.0, 1.0]
        usage: Token counts / cost info (keys depend on provider)
        raw_response: Unparsed provider response for debugging
        success: Whether the request completed without error
        error_message: Error details if success=False
    """

    content: str
    model_used: str
    provider: str
    confidence: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None
    success: bool = True
    error_message: Optional[str] = None

    @classmethod
    def error(cls, provider: str, model_used: str, message: str) -> "AIResponse":
        """
        Convenience constructor for error responses.

        Args:
            provider: Provider ID
            model_used: Model that was attempted
            message: Error description

        Returns:
            AIResponse with success=False
        """
        return cls(
            content="",
            model_used=model_used,
            provider=provider,
            success=False,
            error_message=message,
        )


class BaseAIAdapter(ABC):
    """
    Abstract base class for all AI provider adapters.

    Concrete implementations: GeminiAdapter, OpenAIAdapter, LocalSLMAdapter.

    Rules:
    - Provider SDK imports are ONLY inside the concrete adapter file
    - All methods MUST return AIResponse (never raw SDK types)
    - All provider-specific exceptions MUST be caught and wrapped in
      core_engine.ai.exceptions types
    - health_check() MUST be fast (< 2s) -- used by router before routing
    """

    provider_id: str  # Override in subclasses: "gemini", "openai", "local_slm"

    @abstractmethod
    async def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """
        Send a reasoning prompt and return a structured response.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User-turn prompt with data payload
            model_id: Override default model for this call
            image_path: Optional path to image for vision-capable models
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            AIResponse with content as the model's text output
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if this provider is reachable and configured.

        Must be fast (< 2s). Called by router before routing requests.

        Returns:
            True if provider is available and API key is valid
        """
        ...

    def get_default_model(self) -> str:
        """
        Return the default model ID for this provider.

        Override in concrete adapters to return the appropriate default.
        """
        raise NotImplementedError(
            f"Adapter '{self.provider_id}' must implement get_default_model()"
        )
