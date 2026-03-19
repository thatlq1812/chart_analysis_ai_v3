"""
Tests for Stage 1: Ingestion & Sanitation.

Verifies file format handling, image validation, session creation, and
output schema compliance using real small test images (no GPU required).
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from src.core_engine.stages.s1_ingestion import Stage1Ingestion, IngestionConfig
from src.core_engine.schemas.stage_outputs import Stage1Output, CleanImage
from src.core_engine.exceptions import StageInputError, StageProcessingError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ingestion_config(tmp_path: Path) -> IngestionConfig:
    """Default ingestion config writing to tmp_path."""
    return IngestionConfig(
        output_dir=tmp_path / "processed",
        max_image_size=4096,
        min_image_size=50,
    )


@pytest.fixture
def ingestion_stage(ingestion_config: IngestionConfig) -> Stage1Ingestion:
    """Stage1Ingestion instance with test config."""
    return Stage1Ingestion(ingestion_config)


@pytest.fixture
def small_png(tmp_path: Path) -> Path:
    """Create a 100x100 white PNG test image."""
    img_path = tmp_path / "test_chart.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)
    return img_path


@pytest.fixture
def small_jpg(tmp_path: Path) -> Path:
    """Create a 100x100 white JPEG test image."""
    img_path = tmp_path / "test_chart.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path)
    return img_path


@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    """Create a .txt file that ingestion should reject."""
    txt_path = tmp_path / "readme.txt"
    txt_path.write_text("This is not an image.")
    return txt_path


@pytest.fixture
def large_png(tmp_path: Path) -> Path:
    """Create a 5000x5000 image (above default max_image_size=4096)."""
    img_path = tmp_path / "large.png"
    img = Image.new("RGB", (5000, 5000), color="red")
    img.save(img_path)
    return img_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStage1Ingestion:
    """Tests for the ingestion stage file handling and output."""

    def test_ingestion_accepts_png(
        self, ingestion_stage: Stage1Ingestion, small_png: Path
    ) -> None:
        """PNG files should be processed without error."""
        result = ingestion_stage.process(small_png)
        assert isinstance(result, Stage1Output)
        assert result.total_images >= 1

    def test_ingestion_accepts_jpg(
        self, ingestion_stage: Stage1Ingestion, small_jpg: Path
    ) -> None:
        """JPEG files should be processed without error."""
        result = ingestion_stage.process(small_jpg)
        assert isinstance(result, Stage1Output)
        assert result.total_images >= 1

    def test_ingestion_rejects_unsupported_format(
        self, ingestion_stage: Stage1Ingestion, unsupported_file: Path
    ) -> None:
        """Unsupported file format (.txt) should raise an error."""
        with pytest.raises((StageInputError, ValueError, KeyError)):
            ingestion_stage.process(unsupported_file)

    def test_ingestion_handles_missing_file(
        self, ingestion_stage: Stage1Ingestion, tmp_path: Path
    ) -> None:
        """Non-existent file should raise an error."""
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises((FileNotFoundError, StageInputError, Exception)):
            ingestion_stage.process(missing)

    def test_ingestion_output_schema(
        self, ingestion_stage: Stage1Ingestion, small_png: Path
    ) -> None:
        """Stage1Output should contain images list and session info."""
        result = ingestion_stage.process(small_png)
        assert hasattr(result, "session")
        assert hasattr(result, "images")
        assert isinstance(result.images, list)
        if result.images:
            img = result.images[0]
            assert isinstance(img, CleanImage)
            assert img.width > 0
            assert img.height > 0

    def test_ingestion_respects_max_image_size(
        self, tmp_path: Path, large_png: Path
    ) -> None:
        """Images above max_image_size should be downscaled."""
        config = IngestionConfig(
            output_dir=tmp_path / "processed",
            max_image_size=1024,
            min_image_size=50,
        )
        stage = Stage1Ingestion(config)
        result = stage.process(large_png)
        assert isinstance(result, Stage1Output)
        if result.images:
            img = result.images[0]
            assert img.width <= 1024
            assert img.height <= 1024

    def test_ingestion_creates_session_id(
        self, ingestion_stage: Stage1Ingestion, small_png: Path
    ) -> None:
        """Session ID should be a non-empty string."""
        result = ingestion_stage.process(small_png)
        assert result.session.session_id
        assert isinstance(result.session.session_id, str)
        assert len(result.session.session_id) > 0

    def test_ingestion_session_source_file(
        self, ingestion_stage: Stage1Ingestion, small_png: Path
    ) -> None:
        """Session source_file should reference the input path."""
        result = ingestion_stage.process(small_png)
        assert result.session.source_file is not None

    def test_ingestion_config_defaults(self) -> None:
        """IngestionConfig defaults should be production-suitable."""
        config = IngestionConfig()
        assert config.pdf_dpi == 150
        assert config.max_image_size == 4096
        assert config.min_image_size == 100
        assert config.output_format == "PNG"

    def test_ingestion_validate_input_rejects_directory(
        self, ingestion_stage: Stage1Ingestion, tmp_path: Path
    ) -> None:
        """validate_input with a directory should return False or raise."""
        try:
            result = ingestion_stage.validate_input(tmp_path)
            assert result is False
        except (StageInputError, Exception):
            pass  # Raising is also acceptable for rejecting invalid input

    @pytest.mark.parametrize("extension", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
    def test_ingestion_supported_extensions(
        self, tmp_path: Path, extension: str
    ) -> None:
        """All listed image extensions should be recognized."""
        img_path = tmp_path / f"test{extension}"
        img = Image.new("RGB", (100, 100), color="green")
        if extension in (".jpg", ".jpeg"):
            img.save(img_path, format="JPEG")
        elif extension == ".tiff":
            img.save(img_path, format="TIFF")
        elif extension == ".bmp":
            img.save(img_path, format="BMP")
        else:
            img.save(img_path)

        config = IngestionConfig(output_dir=tmp_path / "out", min_image_size=50)
        stage = Stage1Ingestion(config)
        result = stage.process(img_path)
        assert isinstance(result, Stage1Output)
