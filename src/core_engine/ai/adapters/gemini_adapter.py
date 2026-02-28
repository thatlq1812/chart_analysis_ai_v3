"""
Gemini AI Adapter

Wraps the Google Generative AI SDK behind the BaseAIAdapter interface.
This is the ONLY file that imports google.generativeai.

Authentication: reads GOOGLE_API_KEY from environment or config/secrets/.env

Usage:
    adapter = GeminiAdapter()
    healthy = await adapter.health_check()
    response = await adapter.reason(system_prompt, user_prompt)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .base import AIResponse, BaseAIAdapter
from ..exceptions import (
    AIAuthenticationError,
    AIInvalidResponseError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseAIAdapter):
    """
    Google Gemini provider adapter.

    Supports text-only and vision (multimodal) models.
    Wraps google-generativeai SDK -- all SDK imports are inside this file.

    Attributes:
        provider_id: "gemini"
        default_model: Default Gemini model to use
    """

    provider_id = "gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Initialize the Gemini adapter.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var,
                     then config/secrets/.env
            default_model: Default Gemini model identifier
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum output tokens
            top_p: Nucleus sampling probability
            timeout_seconds: Request timeout
        """
        self._default_model = default_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._timeout = timeout_seconds

        self._api_key = api_key or os.getenv("GOOGLE_API_KEY") or self._load_key_from_file()
        self._genai: Any = None  # google.generativeai module
        self._model: Any = None  # genai.GenerativeModel instance

        if self._api_key:
            self._initialize_client()
        else:
            logger.warning(
                "GeminiAdapter | no API key found | "
                "set GOOGLE_API_KEY env var or config/secrets/.env"
            )

    # -------------------------------------------------------------------------
    # BaseAIAdapter interface
    # -------------------------------------------------------------------------

    async def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AIResponse:
        """
        Send a reasoning request to Gemini.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User-turn prompt with data
            model_id: Override default model for this call
            image_path: Optional image path for vision models
            **kwargs: Extra generation parameters (temperature, max_tokens)

        Returns:
            AIResponse with content as model output text

        Raises:
            AIAuthenticationError: if API key invalid
            AIRateLimitError: if quota exceeded
            AITimeoutError: if request exceeds timeout
            AIInvalidResponseError: if response cannot be parsed
        """
        if self._model is None:
            return AIResponse.error(
                self.provider_id,
                model_id or self._default_model,
                "Gemini client not initialized (missing API key or SDK)",
            )

        effective_model = model_id or self._default_model

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            if image_path:
                raw = await asyncio.to_thread(
                    self._call_with_image, full_prompt, image_path
                )
            else:
                raw = await asyncio.to_thread(self._call_text_only, full_prompt)

            logger.info(
                f"GeminiAdapter.reason | model={effective_model} | "
                f"vision={image_path is not None} | chars={len(raw)}"
            )
            return AIResponse(
                content=raw,
                model_used=effective_model,
                provider=self.provider_id,
                confidence=0.9,  # Gemini does not return explicit confidence
                success=True,
            )

        except Exception as exc:
            return self._handle_exception(exc, effective_model)

    async def health_check(self) -> bool:
        """
        Verify Gemini API is reachable and key is valid.

        Returns:
            True if a minimal test request succeeds
        """
        if self._model is None:
            return False

        try:
            await asyncio.to_thread(self._call_text_only, "Respond with 'ok'.")
            logger.debug("GeminiAdapter.health_check | status=healthy")
            return True
        except Exception as exc:
            logger.warning(f"GeminiAdapter.health_check | failed | error={exc}")
            return False

    def get_default_model(self) -> str:
        """Return default Gemini model identifier."""
        return self._default_model

    # -------------------------------------------------------------------------
    # Private: SDK initialization
    # -------------------------------------------------------------------------

    def _initialize_client(self) -> None:
        """Initialize google-generativeai client."""
        try:
            import google.generativeai as genai  # type: ignore[import]

            genai.configure(api_key=self._api_key)

            generation_config = genai.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                top_p=self._top_p,
            )

            self._model = genai.GenerativeModel(
                model_name=self._default_model,
                generation_config=generation_config,
            )
            self._genai = genai

            logger.info(
                f"GeminiAdapter | initialized | model={self._default_model}"
            )

        except ImportError:
            logger.error(
                "GeminiAdapter | google-generativeai not installed | "
                "run: pip install google-generativeai"
            )
        except Exception as exc:
            logger.error(f"GeminiAdapter | initialization failed | error={exc}")

    def _load_key_from_file(self) -> Optional[str]:
        """Try to load API key from config/secrets/.env."""
        secrets_path = (
            Path(__file__).resolve().parents[5] / "config" / "secrets" / ".env"
        )
        if not secrets_path.exists():
            return None
        try:
            with open(secrets_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception as exc:
            logger.warning(f"GeminiAdapter | failed to read secrets file | error={exc}")
        return None

    # -------------------------------------------------------------------------
    # Private: API call helpers (sync -- wrapped in asyncio.to_thread)
    # -------------------------------------------------------------------------

    def _call_text_only(self, prompt: str) -> str:
        """Synchronous text-only Gemini call."""
        response = self._model.generate_content(prompt)
        return response.text

    def _call_with_image(self, prompt: str, image_path: str) -> str:
        """Synchronous vision Gemini call."""
        from PIL import Image  # type: ignore[import]

        image = Image.open(image_path)
        response = self._model.generate_content([prompt, image])
        return response.text

    # -------------------------------------------------------------------------
    # Private: exception mapping
    # -------------------------------------------------------------------------

    def _handle_exception(
        self, exc: Exception, model_used: str
    ) -> AIResponse:
        """
        Map provider-specific exceptions to AI layer exceptions.

        Logs the error and returns a failed AIResponse so the router
        can decide whether to retry or fall back.
        """
        exc_type = type(exc).__name__
        msg = str(exc)

        if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower():
            logger.warning(
                f"GeminiAdapter | rate limit | model={model_used} | error={msg}"
            )
            raise AIRateLimitError(self.provider_id, msg)

        if (
            "API_KEY_INVALID" in msg
            or "PERMISSION_DENIED" in msg
            or "403" in msg
        ):
            logger.error(
                f"GeminiAdapter | auth error | model={model_used} | error={msg}"
            )
            raise AIAuthenticationError(self.provider_id, msg)

        if "DEADLINE_EXCEEDED" in msg or "timeout" in msg.lower():
            logger.warning(
                f"GeminiAdapter | timeout | model={model_used} | error={msg}"
            )
            raise AITimeoutError(self.provider_id, msg, self._timeout)

        logger.error(
            f"GeminiAdapter | unexpected error | "
            f"type={exc_type} | model={model_used} | error={msg}"
        )
        raise AIProviderError(self.provider_id, f"{exc_type}: {msg}") from exc
