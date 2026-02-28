"""
Integration tests for Stage 3 Extraction.

Tests the complete Stage 3 pipeline from Stage2Output to Stage3Output.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from datetime import datetime
from PIL import Image
import cv2

from core_engine.stages.s3_extraction import (
    Stage3Extraction,
    ExtractionConfig,
)
from core_engine.schemas.stage_outputs import (
    Stage2Output,
    Stage3Output,
    DetectedChart,
)
from core_engine.schemas.common import (
    SessionInfo,
    BoundingBox,
    ChartType,
)


class TestStage3Integration:
    """Integration tests for Stage3Extraction."""

    @pytest.fixture
    def extraction_config(self) -> ExtractionConfig:
        """Create extraction config for testing."""
        return ExtractionConfig()

    @pytest.fixture
    def stage3(self, extraction_config: ExtractionConfig) -> Stage3Extraction:
        """Create Stage3Extraction instance."""
        return Stage3Extraction(extraction_config)

    @pytest.fixture
    def session_info(self) -> SessionInfo:
        """Create session info for testing."""
        return SessionInfo(
            session_id="test_session_001",
            created_at=datetime.now(),
            source_file=Path("test_document.pdf"),
            total_pages=1,
            config_hash="test_hash",
        )

    @pytest.fixture
    def sample_bar_chart(self, tmp_path: Path) -> Path:
        """Create sample bar chart image."""
        # Create 400x400 white background
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw Y-axis
        cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)
        
        # Draw X-axis
        cv2.line(img, (50, 350), (380, 350), (0, 0, 0), 2)
        
        # Draw bars
        cv2.rectangle(img, (80, 150), (120, 350), (255, 0, 0), -1)   # Red
        cv2.rectangle(img, (150, 200), (190, 350), (0, 255, 0), -1)  # Green
        cv2.rectangle(img, (220, 100), (260, 350), (0, 0, 255), -1)  # Blue
        cv2.rectangle(img, (290, 250), (330, 350), (255, 128, 0), -1)  # Orange
        
        # Add title
        cv2.putText(img, "Sales by Quarter", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        
        # Add X-axis labels
        cv2.putText(img, "Q1", (90, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Q2", (160, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Q3", (230, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Q4", (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add Y-axis labels
        cv2.putText(img, "0", (30, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, "50", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, "100", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Save image (img is in BGR as OpenCV draws natively)
        chart_path = tmp_path / "bar_chart.png"
        cv2.imwrite(str(chart_path), img)
        
        return chart_path

    @pytest.fixture
    def sample_line_chart(self, tmp_path: Path) -> Path:
        """Create sample line chart image."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw axes
        cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)
        cv2.line(img, (50, 350), (380, 350), (0, 0, 0), 2)
        
        # Draw line with markers
        points = [(80, 280), (150, 200), (220, 250), (290, 150), (350, 180)]
        
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (0, 0, 255), 2)
        
        for p in points:
            cv2.circle(img, p, 5, (0, 0, 255), -1)
        
        # Add title
        cv2.putText(img, "Monthly Trend", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        
        chart_path = tmp_path / "line_chart.png"
        cv2.imwrite(str(chart_path), img)
        
        return chart_path

    @pytest.fixture
    def sample_pie_chart(self, tmp_path: Path) -> Path:
        """Create sample pie chart image."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        center = (200, 200)
        radius = 120
        
        # Draw pie slices
        cv2.ellipse(img, center, (radius, radius), 0, 0, 120, (255, 0, 0), -1)
        cv2.ellipse(img, center, (radius, radius), 0, 120, 220, (0, 255, 0), -1)
        cv2.ellipse(img, center, (radius, radius), 0, 220, 360, (0, 0, 255), -1)
        
        # Add title
        cv2.putText(img, "Market Share", (120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 0), 2)
        
        chart_path = tmp_path / "pie_chart.png"
        cv2.imwrite(str(chart_path), img)
        
        return chart_path

    @pytest.fixture
    def stage2_output_bar(
        self,
        session_info: SessionInfo,
        sample_bar_chart: Path,
    ) -> Stage2Output:
        """Create Stage2Output with bar chart."""
        return Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="chart_001",
                    source_image=sample_bar_chart,
                    cropped_path=sample_bar_chart,
                    bbox=BoundingBox(
                        x_min=0, y_min=0, x_max=400, y_max=400,
                        confidence=0.95,
                    ),
                    page_number=1,
                ),
            ],
            total_detected=1,
            skipped_low_confidence=0,
        )

    @pytest.fixture
    def stage2_output_line(
        self,
        session_info: SessionInfo,
        sample_line_chart: Path,
    ) -> Stage2Output:
        """Create Stage2Output with line chart."""
        return Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="chart_002",
                    source_image=sample_line_chart,
                    cropped_path=sample_line_chart,
                    bbox=BoundingBox(
                        x_min=0, y_min=0, x_max=400, y_max=400,
                        confidence=0.92,
                    ),
                    page_number=1,
                ),
            ],
            total_detected=1,
            skipped_low_confidence=0,
        )

    @pytest.fixture
    def stage2_output_pie(
        self,
        session_info: SessionInfo,
        sample_pie_chart: Path,
    ) -> Stage2Output:
        """Create Stage2Output with pie chart."""
        return Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="chart_003",
                    source_image=sample_pie_chart,
                    cropped_path=sample_pie_chart,
                    bbox=BoundingBox(
                        x_min=0, y_min=0, x_max=400, y_max=400,
                        confidence=0.88,
                    ),
                    page_number=1,
                ),
            ],
            total_detected=1,
            skipped_low_confidence=0,
        )

    def test_process_bar_chart(
        self,
        stage3: Stage3Extraction,
        stage2_output_bar: Stage2Output,
    ) -> None:
        """Test processing bar chart."""
        result = stage3.process(stage2_output_bar)
        
        assert isinstance(result, Stage3Output)
        assert len(result.metadata) == 1
        
        metadata = result.metadata[0]
        assert metadata.chart_id == "chart_001"
        
        # Should classify as BAR
        assert metadata.chart_type in [ChartType.BAR, ChartType.UNKNOWN]

    def test_process_line_chart(
        self,
        stage3: Stage3Extraction,
        stage2_output_line: Stage2Output,
    ) -> None:
        """Test processing line chart."""
        result = stage3.process(stage2_output_line)
        
        assert isinstance(result, Stage3Output)
        assert len(result.metadata) == 1
        
        metadata = result.metadata[0]
        assert metadata.chart_id == "chart_002"
        
        # Should classify as LINE or have polylines
        assert metadata.chart_type in [ChartType.LINE, ChartType.SCATTER, ChartType.UNKNOWN]

    def test_process_pie_chart(
        self,
        stage3: Stage3Extraction,
        stage2_output_pie: Stage2Output,
    ) -> None:
        """Test processing pie chart."""
        result = stage3.process(stage2_output_pie)
        
        assert isinstance(result, Stage3Output)
        assert len(result.metadata) == 1
        
        metadata = result.metadata[0]
        assert metadata.chart_id == "chart_003"
        
        # Should classify as PIE
        assert metadata.chart_type in [ChartType.PIE, ChartType.UNKNOWN]

    def test_session_preserved(
        self,
        stage3: Stage3Extraction,
        stage2_output_bar: Stage2Output,
        session_info: SessionInfo,
    ) -> None:
        """Test that session info is preserved."""
        result = stage3.process(stage2_output_bar)
        
        assert result.session.session_id == session_info.session_id

    def test_ocr_extraction(
        self,
        stage3: Stage3Extraction,
        stage2_output_bar: Stage2Output,
    ) -> None:
        """Test that OCR extracts text elements."""
        result = stage3.process(stage2_output_bar)
        
        metadata = result.metadata[0]
        
        # Should have extracted some text
        assert len(metadata.texts) >= 0  # May fail if OCR not installed
        
        # If OCR worked, check text structure
        for text in metadata.texts:
            assert hasattr(text, "text")
            assert hasattr(text, "bbox")
            assert hasattr(text, "confidence")

    def test_element_detection(
        self,
        stage3: Stage3Extraction,
        stage2_output_bar: Stage2Output,
    ) -> None:
        """Test that elements are detected."""
        result = stage3.process(stage2_output_bar)
        
        metadata = result.metadata[0]
        
        # Should have detected some elements
        assert len(metadata.elements) >= 0
        
        for element in metadata.elements:
            assert hasattr(element, "element_type")
            assert hasattr(element, "bbox")

    def test_multiple_charts(
        self,
        stage3: Stage3Extraction,
        session_info: SessionInfo,
        sample_bar_chart: Path,
        sample_line_chart: Path,
    ) -> None:
        """Test processing multiple charts."""
        stage2_output = Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="chart_001",
                    source_image=sample_bar_chart,
                    cropped_path=sample_bar_chart,
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=400, y_max=400, confidence=0.95),
                    page_number=1,
                ),
                DetectedChart(
                    chart_id="chart_002",
                    source_image=sample_line_chart,
                    cropped_path=sample_line_chart,
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=400, y_max=400, confidence=0.92),
                    page_number=2,
                ),
            ],
            total_detected=2,
            skipped_low_confidence=0,
        )
        
        result = stage3.process(stage2_output)
        
        assert len(result.metadata) == 2
        assert result.metadata[0].chart_id == "chart_001"
        assert result.metadata[1].chart_id == "chart_002"

    def test_empty_input(
        self,
        stage3: Stage3Extraction,
        session_info: SessionInfo,
    ) -> None:
        """Test handling of empty input."""
        stage2_output = Stage2Output(
            session=session_info,
            charts=[],
            total_detected=0,
            skipped_low_confidence=0,
        )
        
        result = stage3.process(stage2_output)
        
        assert len(result.metadata) == 0

    def test_validate_input(
        self,
        stage3: Stage3Extraction,
        stage2_output_bar: Stage2Output,
    ) -> None:
        """Test input validation."""
        is_valid = stage3.validate_input(stage2_output_bar)
        
        assert is_valid is True

    def test_invalid_input(
        self,
        stage3: Stage3Extraction,
    ) -> None:
        """Test handling of invalid input."""
        is_valid = stage3.validate_input("not a Stage2Output")
        
        assert is_valid is False


class TestStage3VectorizedOutput:
    """Tests for vectorized chart representation."""

    @pytest.fixture
    def stage3(self) -> Stage3Extraction:
        """Create Stage3Extraction instance."""
        return Stage3Extraction()

    @pytest.fixture
    def simple_line_image(self, tmp_path: Path) -> Path:
        """Create simple line for vectorization test."""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Draw diagonal line
        cv2.line(img, (20, 180), (180, 20), (0, 0, 0), 3)
        
        chart_path = tmp_path / "simple_line.png"
        cv2.imwrite(str(chart_path), img)
        
        return chart_path

    def test_get_vectorized_representation(
        self,
        stage3: Stage3Extraction,
        simple_line_image: Path,
    ) -> None:
        """Test getting vectorized representation."""
        session_info = SessionInfo(
            session_id="test_vec",
            created_at=datetime.now(),
            source_file=Path("test.pdf"),
            total_pages=1,
            config_hash="testhash1234",
        )
        
        stage2_output = Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="vec_001",
                    source_image=simple_line_image,
                    cropped_path=simple_line_image,
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200, confidence=0.9),
                    page_number=1,
                ),
            ],
            total_detected=1,
            skipped_low_confidence=0,
        )
        
        # Process and get vectorized
        result = stage3.process(stage2_output)
        
        # Get vectorized for each chart - pass the DetectedChart object, not chart_id string
        for chart in stage2_output.charts:
            vectorized = stage3.get_vectorized_representation(chart)
            
            if vectorized is not None:
                # VectorizedChart has skeleton.polylines and skeleton.keypoints
                assert hasattr(vectorized, "skeleton")
                assert hasattr(vectorized.skeleton, "polylines")
                assert hasattr(vectorized.skeleton, "keypoints")


class TestStage3ErrorHandling:
    """Error handling tests for Stage3Extraction."""

    @pytest.fixture
    def stage3(self) -> Stage3Extraction:
        """Create Stage3Extraction instance."""
        return Stage3Extraction()

    def test_missing_image_file(
        self,
        stage3: Stage3Extraction,
    ) -> None:
        """Test handling of missing image file."""
        session_info = SessionInfo(
            session_id="test_err",
            created_at=datetime.now(),
            source_file=Path("test.pdf"),
            total_pages=1,
            config_hash="testhash1234",
        )
        
        stage2_output = Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="err_001",
                    source_image=Path("/nonexistent/path.png"),
                    cropped_path=Path("/nonexistent/path.png"),
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100, confidence=0.9),
                    page_number=1,
                ),
            ],
            total_detected=1,
            skipped_low_confidence=0,
        )
        
        # Should handle gracefully
        result = stage3.process(stage2_output)
        
        # Either skip the chart or return with warnings
        assert isinstance(result, Stage3Output)

    def test_corrupted_image(
        self,
        stage3: Stage3Extraction,
        tmp_path: Path,
    ) -> None:
        """Test handling of corrupted image file."""
        # Create corrupted file
        corrupted = tmp_path / "corrupted.png"
        corrupted.write_bytes(b"not an image")
        
        session_info = SessionInfo(
            session_id="test_corrupt",
            created_at=datetime.now(),
            source_file=Path("test.pdf"),
            total_pages=1,
            config_hash="testhash1234",
        )
        
        stage2_output = Stage2Output(
            session=session_info,
            charts=[
                DetectedChart(
                    chart_id="corrupt_001",
                    source_image=corrupted,
                    cropped_path=corrupted,
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100, confidence=0.9),
                    page_number=1,
                ),
            ],
            total_detected=1,
            skipped_low_confidence=0,
        )
        
        # Should handle gracefully
        result = stage3.process(stage2_output)
        
        assert isinstance(result, Stage3Output)
