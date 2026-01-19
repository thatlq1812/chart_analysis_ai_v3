"""
Pytest Configuration

Shared fixtures and configuration for tests.
"""

import pytest
from pathlib import Path
from datetime import datetime

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
