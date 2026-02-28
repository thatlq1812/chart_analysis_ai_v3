"""
Tests for core_engine.ai.adapters.base (AIResponse and BaseAIAdapter contract)
"""

import pytest

from core_engine.ai.adapters.base import AIResponse, BaseAIAdapter
from core_engine.ai.task_types import TaskType


class TestAIResponse:
    def test_success_response(self) -> None:
        resp = AIResponse(
            content="Some analysis",
            model_used="gemini-2.0-flash",
            provider="gemini",
            confidence=0.9,
        )
        assert resp.success is True
        assert resp.error_message is None
        assert resp.content == "Some analysis"
        assert resp.confidence == 0.9

    def test_error_classmethod(self) -> None:
        resp = AIResponse.error("gemini", "gemini-2.0-flash", "Rate limit hit")
        assert resp.success is False
        assert resp.content == ""
        assert resp.error_message == "Rate limit hit"
        assert resp.provider == "gemini"
        assert resp.model_used == "gemini-2.0-flash"

    def test_default_confidence_zero(self) -> None:
        resp = AIResponse(content="x", model_used="m", provider="p")
        assert resp.confidence == 0.0

    def test_default_usage_empty_dict(self) -> None:
        resp = AIResponse(content="x", model_used="m", provider="p")
        assert isinstance(resp.usage, dict)
        assert len(resp.usage) == 0

    def test_raw_response_none_by_default(self) -> None:
        resp = AIResponse(content="x", model_used="m", provider="p")
        assert resp.raw_response is None

    def test_usage_dict_not_shared_between_instances(self) -> None:
        r1 = AIResponse(content="a", model_used="m", provider="p")
        r2 = AIResponse(content="b", model_used="m", provider="p")
        r1.usage["tokens"] = 100
        assert "tokens" not in r2.usage


class TestBaseAIAdapterContract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseAIAdapter()  # type: ignore[abstract]

    def test_concrete_must_implement_reason(self) -> None:
        class IncompleteAdapter(BaseAIAdapter):
            provider_id = "test"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_concrete_with_both_methods_works(self) -> None:
        class MinimalAdapter(BaseAIAdapter):
            provider_id = "test"

            async def reason(
                self, system_prompt, user_prompt, model_id=None, image_path=None
            ) -> AIResponse:
                return AIResponse(
                    content="ok",
                    model_used=model_id or "test-model",
                    provider=self.provider_id,
                    success=True,
                )

            async def health_check(self) -> bool:
                return True

        adapter = MinimalAdapter()
        assert adapter.provider_id == "test"
