"""
Tests for Stage 4 reasoning engines (AIRouterEngine, GeminiReasoningEngine).

All tests mock AI adapters and routers to avoid real API calls.
Verifies engine initialization, delegation, and fallback behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.core_engine.ai.adapters.base import AIResponse
from src.core_engine.ai.exceptions import AIProviderExhaustedError
from src.core_engine.schemas.common import BoundingBox, Point
from src.core_engine.schemas.enums import ChartType
from src.core_engine.schemas.stage_outputs import (
    ChartElement,
    DataPoint,
    DataSeries,
    OCRText,
    RawMetadata,
    RefinedChartData,
    SessionInfo,
    Stage3Output,
    Stage4Output,
)
from src.core_engine.stages.s4_reasoning import (
    AIRouterEngine,
    GeminiReasoningEngine,
    GeminiConfig,
    ReasoningConfig,
    Stage4Reasoning,
)
from src.core_engine.stages.s4_reasoning.reasoning_engine import ReasoningResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ai_response() -> AIResponse:
    """Successful AIResponse with valid JSON chart reasoning output."""
    return AIResponse(
        content=(
            '{"title": "Revenue", "x_axis_label": "Year", "y_axis_label": "USD",'
            ' "series": [{"name": "Sales", "points": [{"label": "2023", "value": 100}]}],'
            ' "description": "A bar chart of revenue."}'
        ),
        model_used="mock-model",
        provider="mock",
        confidence=0.9,
        success=True,
    )


@pytest.fixture
def mock_router(mock_ai_response: AIResponse) -> MagicMock:
    """Mock AIRouter that returns success response via route_sync."""
    router = MagicMock()
    router.route_sync.return_value = mock_ai_response
    router._adapters = {"mock": MagicMock()}
    return router


@pytest.fixture
def sample_raw_metadata() -> RawMetadata:
    """Minimal RawMetadata for testing reasoning."""
    return RawMetadata(
        chart_id="chart_test_001",
        chart_type=ChartType.BAR,
        texts=[
            OCRText(
                text="Revenue",
                bbox=BoundingBox(
                    x_min=10, y_min=5, x_max=100, y_max=20, confidence=0.9
                ),
                confidence=0.9,
                role="title",
            ),
        ],
        elements=[
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(
                    x_min=20, y_min=50, x_max=40, y_max=150, confidence=0.95
                ),
                center=Point(x=30, y=100),
            ),
        ],
    )


@pytest.fixture
def sample_stage3_output(sample_raw_metadata: RawMetadata) -> Stage3Output:
    return Stage3Output(
        session=SessionInfo(
            session_id="reasoning_test",
            source_file=Path("test.png"),
            config_hash="test1234",
        ),
        metadata=[sample_raw_metadata],
    )


# ---------------------------------------------------------------------------
# Tests: AIRouterEngine
# ---------------------------------------------------------------------------


class TestAIRouterEngine:
    """Tests for the AIRouter-backed reasoning engine."""

    def test_ai_router_engine_init(self, mock_router: MagicMock) -> None:
        """AIRouterEngine should initialize with a mock router."""
        engine = AIRouterEngine(router=mock_router)
        assert engine.is_available() is True

    def test_ai_router_engine_delegates_to_router(
        self,
        mock_router: MagicMock,
        sample_raw_metadata: RawMetadata,
    ) -> None:
        """reason() should call router.route_sync()."""
        engine = AIRouterEngine(router=mock_router)
        result = engine.reason(sample_raw_metadata)
        assert mock_router.route_sync.called
        assert result.success is True
        assert result.title == "Revenue"

    def test_ai_router_engine_returns_reasoning_result(
        self,
        mock_router: MagicMock,
        sample_raw_metadata: RawMetadata,
    ) -> None:
        """reason() should return a ReasoningResult dataclass."""
        engine = AIRouterEngine(router=mock_router)
        result = engine.reason(sample_raw_metadata)
        assert isinstance(result, ReasoningResult)
        assert len(result.series) == 1

    def test_ai_router_engine_handles_provider_exhaustion(
        self,
        sample_raw_metadata: RawMetadata,
    ) -> None:
        """When all providers fail, engine should return failed ReasoningResult."""
        router = MagicMock()
        router._adapters = {"mock": MagicMock()}
        router.route_sync.side_effect = AIProviderExhaustedError(
            "chart_reasoning", {"mock": "timeout"}
        )

        engine = AIRouterEngine(router=router)
        result = engine.reason(sample_raw_metadata)
        assert result.success is False
        assert "exhausted" in result.error_message.lower()

    def test_ai_router_engine_handles_bad_json(
        self,
        sample_raw_metadata: RawMetadata,
    ) -> None:
        """Malformed JSON response should produce failed ReasoningResult."""
        router = MagicMock()
        router._adapters = {"mock": MagicMock()}
        router.route_sync.return_value = AIResponse(
            content="not valid json {{{",
            model_used="mock",
            provider="mock",
            confidence=0.9,
            success=True,
        )

        engine = AIRouterEngine(router=router)
        result = engine.reason(sample_raw_metadata)
        assert result.success is False
        assert "json" in (result.error_message or "").lower()

    def test_ai_router_engine_correct_ocr_empty(
        self,
        mock_router: MagicMock,
    ) -> None:
        """correct_ocr with empty texts should return original."""
        engine = AIRouterEngine(router=mock_router)
        texts, corrections = engine.correct_ocr([], ChartType.BAR)
        assert texts == []
        assert corrections == []


# ---------------------------------------------------------------------------
# Tests: GeminiReasoningEngine
# ---------------------------------------------------------------------------


class TestGeminiReasoningEngine:
    """Tests for the legacy Gemini reasoning engine."""

    def test_gemini_engine_fallback(self) -> None:
        """GeminiReasoningEngine should handle missing API key gracefully."""
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiConfig(model_name="gemini-2.0-flash")
            engine = GeminiReasoningEngine(config)
            # Engine should initialize but be unavailable for inference
            assert isinstance(engine, GeminiReasoningEngine)


# ---------------------------------------------------------------------------
# Tests: Stage4Reasoning orchestrator
# ---------------------------------------------------------------------------


class TestStage4Reasoning:
    """Tests for the Stage 4 orchestrator."""

    def test_stage4_reasoning_init(self) -> None:
        """Stage4Reasoning should accept a ReasoningConfig."""
        config = ReasoningConfig(engine="rule_based")
        stage = Stage4Reasoning(config)
        assert stage.config.engine == "rule_based"

    def test_stage4_processes_metadata(
        self,
        mock_router: MagicMock,
        sample_stage3_output: Stage3Output,
    ) -> None:
        """Process RawMetadata -> RefinedChartData with mocked engine."""
        config = ReasoningConfig(engine="router")
        stage = Stage4Reasoning(config)
        # Replace engine with our mock-backed one
        stage.engine = AIRouterEngine(router=mock_router)

        result = stage.process(sample_stage3_output)
        assert isinstance(result, Stage4Output)
        assert len(result.charts) == 1
        assert result.charts[0].chart_id == "chart_test_001"

    def test_stage4_handles_empty_input(self) -> None:
        """Empty metadata list should produce empty Stage4Output."""
        config = ReasoningConfig(engine="rule_based")
        stage = Stage4Reasoning(config)

        s3 = Stage3Output(
            session=SessionInfo(
                session_id="empty_test",
                source_file=Path("test.png"),
                config_hash="abcdef12",
            ),
            metadata=[],
        )
        result = stage.process(s3)
        assert isinstance(result, Stage4Output)
        assert len(result.charts) == 0

    def test_stage4_fallback_on_engine_error(
        self,
        sample_stage3_output: Stage3Output,
    ) -> None:
        """When engine fails and fallback is enabled, should produce fallback result."""
        config = ReasoningConfig(engine="router", use_fallback_on_error=True)
        stage = Stage4Reasoning(config)
        # Make engine raise on every chart
        mock_engine = MagicMock()
        mock_engine.reason.side_effect = RuntimeError("Provider down")
        stage.engine = mock_engine

        result = stage.process(sample_stage3_output)
        # Should not raise, fallback should produce a result
        assert isinstance(result, Stage4Output)
        assert len(result.charts) == 1

    @pytest.mark.parametrize("engine_type", ["router", "gemini", "rule_based"])
    def test_stage4_engine_types(self, engine_type: str) -> None:
        """Stage4 should accept different engine type strings."""
        config = ReasoningConfig(engine=engine_type)
        stage = Stage4Reasoning(config)
        assert stage.config.engine == engine_type
