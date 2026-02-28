"""
OpenAI Adapter

Wraps the OpenAI SDK behind the BaseAIAdapter interface.
This is the ONLY file that imports openai.

Status: Optional fallback provider. Disabled by default.
        Set enabled=True in config/models.yaml to activate.

Authentication: reads OPENAI_API_KEY from environment.

Usage:
    adapter = OpenAIAdapter()
    healthy = await adapter.health_check()
    response = await adapter.reason(system_prompt, user_prompt)
"""

import asyncio
import logging
import os
from typing import Any, Optional

from .base import AIResponse, BaseAIAdapter
from ..exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIAdapter(BaseAIAdapter):
    """
    OpenAI provider adapter.

    Supports text-only and vision (gpt-4o) models.
    Image support requires base64 encoding.

    Attributes:
        provider_id: "openai"
    """

    provider_id = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            default_model: Default model identifier
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            timeout_seconds: Request timeout
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._default_model = default_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._client: Any = None

        if self._api_key:
            self._initialize_client()
        else:
            logger.warning(
                "OpenAIAdapter | no API key found | set OPENAI_API_KEY env var"
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
        Send a request to the OpenAI Chat Completions API.

        Args:
            system_prompt: System role message
            user_prompt: User role message with data payload
            model_id: Override default model
            image_path: Optional image for vision models (gpt-4o)
            **kwargs: Extra generation parameters

        Returns:
            AIResponse with the assistant's reply
        """
        if self._client is None:
            return AIResponse.error(
                self.provider_id,
                model_id or self._default_model,
                "OpenAI client not initialized (missing API key or SDK)",
            )

        effective_model = model_id or self._default_model

        try:
            raw = await asyncio.to_thread(
                self._call_api,
                system_prompt,
                user_prompt,
                effective_model,
                image_path,
                **kwargs,
            )
            logger.info(
                f"OpenAIAdapter.reason | model={effective_model} | chars={len(raw)}"
            )
            return AIResponse(
                content=raw,
                model_used=effective_model,
                provider=self.provider_id,
                confidence=0.88,
                success=True,
            )
        except (AIRateLimitError, AIAuthenticationError, AITimeoutError, AIProviderError):
            raise
        except Exception as exc:
            return self._handle_exception(exc, effective_model)

    async def health_check(self) -> bool:
        """Verify OpenAI API is reachable."""
        if self._client is None:
            return False
        try:
            await asyncio.to_thread(
                self._call_api,
                "You are helpful.",
                "Respond with 'ok'.",
                DEFAULT_MODEL,
                None,
            )
            return True
        except Exception as exc:
            logger.warning(f"OpenAIAdapter.health_check | failed | error={exc}")
            return False

    def get_default_model(self) -> str:
        return self._default_model

    # -------------------------------------------------------------------------
    # Private: SDK initialization
    # -------------------------------------------------------------------------

    def _initialize_client(self) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
            logger.info(f"OpenAIAdapter | initialized | model={self._default_model}")
        except ImportError:
            logger.error(
                "OpenAIAdapter | openai package not installed | "
                "run: pip install openai"
            )
        except Exception as exc:
            logger.error(f"OpenAIAdapter | initialization failed | error={exc}")

    # -------------------------------------------------------------------------
    # Private: API call
    # -------------------------------------------------------------------------

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        image_path: Optional[str],
        **kwargs: Any,
    ) -> str:
        """Synchronous OpenAI chat completion call."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        if image_path:
            import base64

            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        return response.choices[0].message.content or ""

    # -------------------------------------------------------------------------
    # Private: exception mapping
    # -------------------------------------------------------------------------

    def _handle_exception(self, exc: Exception, model_used: str) -> AIResponse:
        """Map OpenAI SDK exceptions to AI layer exceptions."""
        msg = str(exc)
        exc_type = type(exc).__name__

        if "RateLimitError" in exc_type or "429" in msg:
            raise AIRateLimitError(self.provider_id, msg)
        if "AuthenticationError" in exc_type or "401" in msg:
            raise AIAuthenticationError(self.provider_id, msg)
        if "APITimeoutError" in exc_type or "timeout" in msg.lower():
            raise AITimeoutError(self.provider_id, msg, self._timeout)

        logger.error(
            f"OpenAIAdapter | unexpected error | "
            f"type={exc_type} | model={model_used} | error={msg}"
        )
        raise AIProviderError(self.provider_id, f"{exc_type}: {msg}") from exc
