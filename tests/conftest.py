"""
Pytest Configuration

Shared fixtures and configuration for tests.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_session_info():
    """Create sample SessionInfo for testing."""
    from src.core_engine.schemas import SessionInfo
    
    return SessionInfo(
        session_id="test_session_001",
        source_file=Path("test_chart.png"),
        config_hash="abc12345",
    )


@pytest.fixture
def sample_bbox():
    """Create sample BoundingBox for testing."""
    from src.core_engine.schemas import BoundingBox
    
    return BoundingBox(
        x_min=10,
        y_min=20,
        x_max=100,
        y_max=150,
        confidence=0.95,
    )


@pytest.fixture
def sample_color():
    """Create sample Color for testing."""
    from src.core_engine.schemas import Color
    
    return Color(r=255, g=128, b=64)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    dirs = ["raw", "processed", "cache"]
    for d in dirs:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def sample_config():
    """Create sample pipeline configuration."""
    return {
        "pipeline": {
            "name": "test-pipeline",
            "version": "1.0.0",
            "stages": {
                "ingestion": {"enabled": True},
                "detection": {"enabled": True},
                "extraction": {"enabled": True},
                "reasoning": {"enabled": True},
                "reporting": {"enabled": True},
            },
        },
        "session": {
            "id_prefix": "test",
            "timestamp_format": "%Y%m%d_%H%M%S",
        },
    }


# ============================================================================
# Cross-stage fixtures for integration testing
# ============================================================================


@pytest.fixture
def mock_ai_router():
    """Mock AIRouter that always returns a success AIResponse."""
    from src.core_engine.ai.adapters.base import AIResponse

    router = MagicMock()
    router.route_sync.return_value = AIResponse(
        content='{"title": "Test Chart", "series": []}',
        model_used="mock",
        provider="mock",
        confidence=0.9,
    )
    router._adapters = {"mock": MagicMock()}
    return router


@pytest.fixture
def sample_stage1_output(tmp_path):
    """Create a minimal Stage1Output with a real test image on disk."""
    from PIL import Image
    from src.core_engine.schemas import SessionInfo
    from src.core_engine.schemas.stage_outputs import CleanImage, Stage1Output

    img_path = tmp_path / "test_chart.png"
    img = Image.new("RGB", (200, 200), "white")
    img.save(img_path)

    session = SessionInfo(
        session_id="fixture_session_001",
        source_file=img_path,
        config_hash="fixture_hash",
    )
    return Stage1Output(
        session=session,
        images=[
            CleanImage(
                image_path=img_path,
                original_path=img_path,
                page_number=1,
                width=200,
                height=200,
            )
        ],
    )


@pytest.fixture
def sample_stage2_output(tmp_path, sample_stage1_output):
    """Create Stage2Output with 2 detected charts."""
    from src.core_engine.schemas import BoundingBox
    from src.core_engine.schemas.stage_outputs import DetectedChart, Stage2Output

    charts = []
    for i in range(2):
        chart_path = tmp_path / f"chart_{i:03d}.png"
        from PIL import Image
        Image.new("RGB", (100, 100), "gray").save(chart_path)
        charts.append(
            DetectedChart(
                chart_id=f"chart_{i:03d}",
                source_image=sample_stage1_output.images[0].image_path,
                cropped_path=chart_path,
                bbox=BoundingBox(
                    x_min=10 + i * 50,
                    y_min=10,
                    x_max=60 + i * 50,
                    y_max=100,
                    confidence=0.9 - i * 0.1,
                ),
            )
        )

    return Stage2Output(
        session=sample_stage1_output.session,
        charts=charts,
        total_detected=2,
    )


@pytest.fixture
def sample_stage3_output(sample_stage1_output):
    """Create Stage3Output with raw metadata for 1 chart."""
    from src.core_engine.schemas.enums import ChartType
    from src.core_engine.schemas.stage_outputs import RawMetadata, Stage3Output

    return Stage3Output(
        session=sample_stage1_output.session,
        metadata=[
            RawMetadata(
                chart_id="chart_000",
                chart_type=ChartType.BAR,
            ),
        ],
    )


@pytest.fixture
def sample_pipeline_config():
    """Load pipeline config from config/pipeline.yaml or return defaults."""
    from omegaconf import OmegaConf

    config_path = Path("config") / "pipeline.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)

    return OmegaConf.create(
        {
            "pipeline": {
                "name": "test-pipeline",
                "version": "1.0.0",
                "stages": {
                    "ingestion": {"enabled": True},
                    "detection": {"enabled": True},
                    "extraction": {"enabled": True},
                    "reasoning": {"enabled": True},
                    "reporting": {"enabled": True},
                },
            },
        }
    )
