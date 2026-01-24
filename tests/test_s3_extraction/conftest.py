"""
Pytest fixtures for Stage 3 Extraction tests.

Shared fixtures for testing all extraction submodules.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from datetime import datetime
import cv2

from core_engine.schemas.common import SessionInfo, BoundingBox
from core_engine.schemas.stage_outputs import Stage2Output, DetectedChart


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def session_info() -> SessionInfo:
    """Create standard session info for tests."""
    return SessionInfo(
        session_id="test_session",
        created_at=datetime.now(),
        source_file=Path("test_doc.pdf"),
        total_pages=1,
        config_hash="test_hash_12345",
    )


@pytest.fixture
def white_image() -> np.ndarray:
    """Create blank white image."""
    return np.ones((400, 400, 3), dtype=np.uint8) * 255


@pytest.fixture
def black_image() -> np.ndarray:
    """Create blank black image."""
    return np.zeros((400, 400, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Create grayscale test image."""
    return np.ones((400, 400), dtype=np.uint8) * 128


@pytest.fixture
def bar_chart_image() -> np.ndarray:
    """Create standard bar chart test image."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)
    cv2.line(img, (50, 350), (380, 350), (0, 0, 0), 2)
    
    # Draw bars
    cv2.rectangle(img, (80, 150), (120, 350), (255, 0, 0), -1)
    cv2.rectangle(img, (150, 200), (190, 350), (0, 255, 0), -1)
    cv2.rectangle(img, (220, 100), (260, 350), (0, 0, 255), -1)
    cv2.rectangle(img, (290, 250), (330, 350), (255, 128, 0), -1)
    
    return img


@pytest.fixture
def line_chart_image() -> np.ndarray:
    """Create standard line chart test image."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)
    cv2.line(img, (50, 350), (380, 350), (0, 0, 0), 2)
    
    # Draw line
    points = [(80, 280), (150, 200), (220, 250), (290, 150), (350, 180)]
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (0, 0, 255), 2)
    
    # Draw markers
    for p in points:
        cv2.circle(img, p, 5, (0, 0, 255), -1)
    
    return img


@pytest.fixture
def scatter_chart_image() -> np.ndarray:
    """Create standard scatter chart test image."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)
    cv2.line(img, (50, 350), (380, 350), (0, 0, 0), 2)
    
    # Draw scattered points
    np.random.seed(42)
    for _ in range(20):
        x = np.random.randint(80, 350)
        y = np.random.randint(80, 320)
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        cv2.circle(img, (x, y), 6, color, -1)
    
    return img


@pytest.fixture
def pie_chart_image() -> np.ndarray:
    """Create standard pie chart test image."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    center = (200, 200)
    radius = 120
    
    cv2.ellipse(img, center, (radius, radius), 0, 0, 120, (255, 0, 0), -1)
    cv2.ellipse(img, center, (radius, radius), 0, 120, 220, (0, 255, 0), -1)
    cv2.ellipse(img, center, (radius, radius), 0, 220, 360, (0, 0, 255), -1)
    
    return img


@pytest.fixture
def save_temp_image(tmp_path: Path):
    """Factory fixture to save images to temp path."""
    def _save(image: np.ndarray, name: str = "test.png") -> Path:
        path = tmp_path / name
        # Convert RGB to BGR for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), image)
        return path
    
    return _save


@pytest.fixture
def stage2_output_factory(session_info: SessionInfo, save_temp_image):
    """Factory to create Stage2Output from images."""
    def _create(images: list, chart_ids: list = None) -> Stage2Output:
        if chart_ids is None:
            chart_ids = [f"chart_{i:03d}" for i in range(len(images))]
        
        charts = []
        for i, (img, chart_id) in enumerate(zip(images, chart_ids)):
            path = save_temp_image(img, f"{chart_id}.png")
            charts.append(
                DetectedChart(
                    chart_id=chart_id,
                    source_image=path,
                    cropped_path=path,
                    bbox=BoundingBox(
                        x_min=0,
                        y_min=0,
                        x_max=img.shape[1],
                        y_max=img.shape[0],
                        confidence=0.95,
                    ),
                    page_number=i + 1,
                )
            )
        
        return Stage2Output(
            session=session_info,
            charts=charts,
            total_detected=len(charts),
            skipped_low_confidence=0,
        )
    
    return _create


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_paddleocr: mark test as requiring PaddleOCR"
    )
    config.addinivalue_line(
        "markers", "requires_skimage: mark test as requiring scikit-image"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
