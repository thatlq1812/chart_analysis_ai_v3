"""
Tests for core_engine.ai.exceptions
"""

import pytest

from core_engine.ai.exceptions import (
    AIAuthenticationError,
    AIInvalidResponseError,
    AIProviderError,
    AIProviderExhaustedError,
    AIRateLimitError,
    AITimeoutError,
)


class TestAIExceptionHierarchy:
    def test_rate_limit_is_provider_error(self) -> None:
        err = AIRateLimitError("gemini", "Too many requests")
        assert isinstance(err, AIProviderError)

    def test_auth_is_provider_error(self) -> None:
        err = AIAuthenticationError("openai", "Invalid API key")
        assert isinstance(err, AIProviderError)

    def test_timeout_is_provider_error(self) -> None:
        err = AITimeoutError("local_slm", "Request timed out")
        assert isinstance(err, AIProviderError)

    def test_invalid_response_is_provider_error(self) -> None:
        err = AIInvalidResponseError("gemini", "Malformed JSON")
        assert isinstance(err, AIProviderError)

    def test_exhausted_is_exception(self) -> None:
        err = AIProviderExhaustedError(
            task_type="chart_reasoning",
            errors={"gemini": "rate limited", "openai": "auth error"},
        )
        assert isinstance(err, Exception)

    def test_provider_error_stores_provider_id(self) -> None:
        err = AIRateLimitError("gemini", "rate limited")
        assert err.provider == "gemini"

    def test_provider_error_stores_message(self) -> None:
        err = AIAuthenticationError("openai", "bad key")
        assert "bad key" in str(err)

    def test_exhausted_error_message_lists_providers(self) -> None:
        err = AIProviderExhaustedError(
            task_type="chart_reasoning",
            errors={"gemini": "err1", "openai": "err2", "local_slm": "err3"},
        )
        msg = str(err)
        assert "gemini" in msg or "openai" in msg or "local_slm" in msg

    def test_raise_and_catch_as_provider_error(self) -> None:
        with pytest.raises(AIProviderError):
            raise AIRateLimitError("gemini", "Too many requests")
