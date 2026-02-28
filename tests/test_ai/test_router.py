"""
Tests for core_engine.ai.router (AIRouter fallback chain logic)

All tests use mock adapters -- no real API calls are made.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_engine.ai.adapters.base import AIResponse, BaseAIAdapter
from core_engine.ai.exceptions import (
    AIProviderError,
    AIProviderExhaustedError,
    AIRateLimitError,
)
from core_engine.ai.router import AIRouter
from core_engine.ai.task_types import TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adapter(
    provider_id: str,
    *,
    healthy: bool = True,
    response_content: str = "ok",
    confidence: float = 0.9,
    raises: Optional[Exception] = None,
) -> BaseAIAdapter:
    """Create a concrete adapter with configurable behavior for testing."""

    # Capture params in closure for the concrete class below
    _provider_id = provider_id
    _healthy = healthy
    _response_content = response_content
    _confidence = confidence
    _raises = raises

    class MockAdapter(BaseAIAdapter):
        provider_id = _provider_id  # type: ignore[assignment]

        async def reason(
            self, system_prompt, user_prompt, model_id=None, image_path=None, **kw
        ) -> AIResponse:
            if _raises:
                raise _raises
            return AIResponse(
                content=_response_content,
                model_used=model_id or f"{_provider_id}-default",
                provider=_provider_id,
                confidence=_confidence,
                success=True,
            )

        async def health_check(self) -> bool:
            return _healthy

    return MockAdapter()


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAIRouterInit:
    def test_custom_adapters_registered(self) -> None:
        a = make_adapter("gemini")
        router = AIRouter(adapters={"gemini": a})
        assert "gemini" in router._adapters

    def test_confidence_threshold_stored(self) -> None:
        router = AIRouter(adapters={}, confidence_threshold=0.5)
        assert router.confidence_threshold == 0.5

    def test_default_chains_set_for_all_task_types(self) -> None:
        router = AIRouter(adapters={})
        for task in TaskType:
            assert task in router._chains


class TestAIRouterRouting:
    def test_uses_first_healthy_provider(self) -> None:
        gemini = make_adapter("gemini", response_content="gemini_result")
        openai = make_adapter("openai", response_content="openai_result")

        chains = {TaskType.CHART_REASONING: ["gemini", "openai"]}
        router = AIRouter(
            adapters={"gemini": gemini, "openai": openai},
            fallback_chains=chains,
            confidence_threshold=0.5,
        )

        resp = run_async(
            router.route(
                TaskType.CHART_REASONING,
                system_prompt="sys",
                user_prompt="user",
            )
        )
        assert resp.content == "gemini_result"
        assert resp.provider == "gemini"

    def test_falls_back_when_first_unhealthy(self) -> None:
        gemini = make_adapter("gemini", healthy=False)
        openai = make_adapter("openai", response_content="fallback")

        chains = {TaskType.CHART_REASONING: ["gemini", "openai"]}
        router = AIRouter(
            adapters={"gemini": gemini, "openai": openai},
            fallback_chains=chains,
            confidence_threshold=0.5,
        )

        resp = run_async(
            router.route(TaskType.CHART_REASONING, "sys", "user")
        )
        assert resp.provider == "openai"
        assert resp.content == "fallback"

    def test_falls_back_when_first_raises(self) -> None:
        failing = make_adapter(
            "gemini",
            healthy=True,
            raises=AIRateLimitError("gemini", "rate limited"),
        )
        working = make_adapter("openai", response_content="recovered", confidence=0.95)

        chains = {TaskType.CHART_REASONING: ["gemini", "openai"]}
        router = AIRouter(
            adapters={"gemini": failing, "openai": working},
            fallback_chains=chains,
            confidence_threshold=0.5,
        )

        resp = run_async(
            router.route(TaskType.CHART_REASONING, "sys", "user")
        )
        assert resp.provider == "openai"
        assert resp.content == "recovered"

    def test_low_confidence_triggers_fallback(self) -> None:
        low_conf = make_adapter("gemini", confidence=0.3, response_content="weak")
        high_conf = make_adapter("openai", confidence=0.95, response_content="strong")

        chains = {TaskType.CHART_REASONING: ["gemini", "openai"]}
        router = AIRouter(
            adapters={"gemini": low_conf, "openai": high_conf},
            fallback_chains=chains,
            confidence_threshold=0.7,
        )

        resp = run_async(
            router.route(TaskType.CHART_REASONING, "sys", "user")
        )
        assert resp.content == "strong"
        assert resp.provider == "openai"

    def test_all_providers_fail_raises_exhausted(self) -> None:
        failing1 = make_adapter(
            "gemini", raises=AIRateLimitError("gemini", "err")
        )
        failing2 = make_adapter(
            "openai", raises=AIRateLimitError("openai", "err")
        )

        chains = {TaskType.CHART_REASONING: ["gemini", "openai"]}
        router = AIRouter(
            adapters={"gemini": failing1, "openai": failing2},
            fallback_chains=chains,
            confidence_threshold=0.5,
        )

        with pytest.raises(AIProviderExhaustedError):
            run_async(router.route(TaskType.CHART_REASONING, "sys", "user"))

    def test_missing_provider_in_adapters_skipped(self) -> None:
        working = make_adapter("openai", response_content="works", confidence=0.9)

        chains = {TaskType.CHART_REASONING: ["gemini", "openai"]}
        # "gemini" not registered in adapters
        router = AIRouter(
            adapters={"openai": working},
            fallback_chains=chains,
            confidence_threshold=0.5,
        )

        resp = run_async(
            router.route(TaskType.CHART_REASONING, "sys", "user")
        )
        assert resp.provider == "openai"


class TestAIRouterRouteSync:
    def test_route_sync_returns_response(self) -> None:
        adapter = make_adapter("gemini", response_content="sync_result")
        chains = {TaskType.OCR_CORRECTION: ["gemini"]}
        router = AIRouter(
            adapters={"gemini": adapter},
            fallback_chains=chains,
            confidence_threshold=0.5,
        )
        resp = router.route_sync(
            TaskType.OCR_CORRECTION, "sys", "user"
        )
        assert resp.content == "sync_result"
        assert resp.success is True


class TestAIRouterDefaultChains:
    """Verify that default fallback chains contain sensible entries."""

    def setup_method(self) -> None:
        self.router = AIRouter(adapters={})

    def test_chart_reasoning_chain_not_empty(self) -> None:
        chain = self.router._chains[TaskType.CHART_REASONING]
        assert len(chain) >= 1

    def test_ocr_correction_chain_not_empty(self) -> None:
        chain = self.router._chains[TaskType.OCR_CORRECTION]
        assert len(chain) >= 1

    def test_description_gen_chain_not_empty(self) -> None:
        chain = self.router._chains[TaskType.DESCRIPTION_GEN]
        assert len(chain) >= 1

    def test_data_validation_chain_not_empty(self) -> None:
        chain = self.router._chains[TaskType.DATA_VALIDATION]
        assert len(chain) >= 1
