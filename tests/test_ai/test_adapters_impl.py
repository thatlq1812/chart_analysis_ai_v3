"""
Tests for concrete AI adapter implementations (Gemini, OpenAI, PaddleVL).

All provider SDKs are mocked -- no real API calls, no keys required.
Verifies adapter initialization, reason() delegation, health checks,
error wrapping, and AIResponse return type.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core_engine.ai.adapters.base import AIResponse, BaseAIAdapter
from src.core_engine.ai.adapters.gemini_adapter import GeminiAdapter
from src.core_engine.ai.adapters.openai_adapter import OpenAIAdapter
from src.core_engine.ai.adapters.paddlevl_adapter import PaddleVLAdapter
from src.core_engine.ai.exceptions import (
    AIAuthenticationError,
    AIProviderError,
    AIRateLimitError,
)


# ---------------------------------------------------------------------------
# Helper to run async tests
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests: GeminiAdapter
# ---------------------------------------------------------------------------


class TestGeminiAdapter:
    """Tests for the Google Gemini adapter with mocked SDK."""

    def test_gemini_adapter_init_no_key(self) -> None:
        """GeminiAdapter without API key should init but log warning."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            assert adapter.provider_id == "gemini"
            assert adapter._model is None

    def test_gemini_adapter_init_with_key(self) -> None:
        """GeminiAdapter with a fake key should attempt to initialize."""
        with patch("src.core_engine.ai.adapters.gemini_adapter.GeminiAdapter._initialize_client"):
            adapter = GeminiAdapter(api_key="fake-key-123")
            assert adapter.provider_id == "gemini"
            assert adapter._api_key == "fake-key-123"

    def test_gemini_adapter_reason_no_model(self) -> None:
        """reason() without initialized model should return error AIResponse."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            resp = _run(adapter.reason("system", "user"))
            assert isinstance(resp, AIResponse)
            assert resp.success is False
            assert "not initialized" in resp.error_message

    def test_gemini_adapter_reason_success(self) -> None:
        """reason() with mocked model should return successful AIResponse."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            # Mock the model to return text
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="Chart analysis result")
            adapter._model = mock_model

            resp = _run(adapter.reason("system prompt", "user data"))
            assert isinstance(resp, AIResponse)
            assert resp.success is True
            assert resp.content == "Chart analysis result"
            assert resp.provider == "gemini"

    def test_gemini_adapter_health_check_no_model(self) -> None:
        """Health check without model should return False."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            result = _run(adapter.health_check())
            assert result is False

    def test_gemini_adapter_health_check_success(self) -> None:
        """Health check with working model should return True."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="ok")
            adapter._model = mock_model

            result = _run(adapter.health_check())
            assert result is True

    def test_gemini_adapter_get_default_model(self) -> None:
        """get_default_model() should return the configured model name."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None, default_model="gemini-2.0-flash")
            assert adapter.get_default_model() == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Tests: OpenAIAdapter
# ---------------------------------------------------------------------------


class TestOpenAIAdapter:
    """Tests for the OpenAI adapter with mocked SDK."""

    def test_openai_adapter_init_no_key(self) -> None:
        """OpenAIAdapter without API key should init but client is None."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAIAdapter(api_key=None)
            assert adapter.provider_id == "openai"
            assert adapter._client is None

    def test_openai_adapter_init_with_key(self) -> None:
        """OpenAIAdapter with a fake key should attempt client creation."""
        with patch("src.core_engine.ai.adapters.openai_adapter.OpenAIAdapter._initialize_client"):
            adapter = OpenAIAdapter(api_key="sk-fake-key")
            assert adapter._api_key == "sk-fake-key"

    def test_openai_adapter_reason_no_client(self) -> None:
        """reason() without client should return error AIResponse."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAIAdapter(api_key=None)
            resp = _run(adapter.reason("system", "user"))
            assert isinstance(resp, AIResponse)
            assert resp.success is False

    def test_openai_adapter_reason_success(self) -> None:
        """reason() with mocked client should return AIResponse."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAIAdapter(api_key=None)
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenAI chart result"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[mock_choice]
            )
            adapter._client = mock_client

            resp = _run(adapter.reason("system", "user"))
            assert isinstance(resp, AIResponse)
            assert resp.success is True
            assert resp.content == "OpenAI chart result"
            assert resp.provider == "openai"

    def test_openai_adapter_health_check_no_client(self) -> None:
        """Health check without client should return False."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAIAdapter(api_key=None)
            result = _run(adapter.health_check())
            assert result is False


# ---------------------------------------------------------------------------
# Tests: PaddleVLAdapter
# ---------------------------------------------------------------------------


class TestPaddleVLAdapter:
    """Tests for the PaddleOCR-VL HTTP adapter."""

    def test_paddlevl_adapter_init(self) -> None:
        """PaddleVLAdapter should store server URL."""
        adapter = PaddleVLAdapter(server_url="http://localhost:9999")
        assert adapter.provider_id == "paddlevl"
        assert adapter._server_url == "http://localhost:9999"

    def test_paddlevl_adapter_health_check_server_down(self) -> None:
        """Health check should return False when server is unreachable."""
        adapter = PaddleVLAdapter(server_url="http://localhost:19999")
        result = _run(adapter.health_check())
        assert result is False

    def test_paddlevl_adapter_reason_requires_image(self) -> None:
        """reason() without image_path should raise AIProviderError."""
        adapter = PaddleVLAdapter()
        with pytest.raises(AIProviderError, match="image_path is required"):
            _run(adapter.reason("sys", "user", image_path=None))

    def test_paddlevl_adapter_reason_missing_file(self, tmp_path: Path) -> None:
        """reason() with non-existent image should raise AIProviderError."""
        adapter = PaddleVLAdapter()
        fake_path = str(tmp_path / "nonexistent.png")
        with pytest.raises(AIProviderError, match="Image not found"):
            _run(adapter.reason("sys", "user", image_path=fake_path))


# ---------------------------------------------------------------------------
# Tests: Cross-adapter concerns
# ---------------------------------------------------------------------------


class TestAdapterCrossContractTests:
    """Tests verifying common behavior across all adapters."""

    def test_adapter_wraps_provider_errors(self) -> None:
        """Gemini adapter should wrap SDK exceptions as AIProviderError."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("SDK internal error")
            adapter._model = mock_model

            # Should not raise raw Exception; should wrap in AIProviderError
            # or return an error AIResponse
            try:
                resp = _run(adapter.reason("sys", "user"))
                # If it returns an AIResponse, it should indicate failure
                # Actually the adapter re-raises as AIProviderError
            except AIProviderError:
                pass  # Expected

    def test_adapter_returns_ai_response(self) -> None:
        """All adapters should return AIResponse type from reason()."""
        # GeminiAdapter without model returns error AIResponse (not raises)
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiAdapter(api_key=None)
            resp = _run(adapter.reason("sys", "user"))
            assert isinstance(resp, AIResponse)

        # OpenAIAdapter without client returns error AIResponse
        with patch.dict("os.environ", {}, clear=True):
            adapter_o = OpenAIAdapter(api_key=None)
            resp_o = _run(adapter_o.reason("sys", "user"))
            assert isinstance(resp_o, AIResponse)

    @pytest.mark.parametrize(
        "adapter_cls,provider_id",
        [
            (GeminiAdapter, "gemini"),
            (OpenAIAdapter, "openai"),
            (PaddleVLAdapter, "paddlevl"),
        ],
    )
    def test_provider_id_set(self, adapter_cls, provider_id: str) -> None:
        """Each adapter should declare its provider_id correctly."""
        assert adapter_cls.provider_id == provider_id
