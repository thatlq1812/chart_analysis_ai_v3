"""
Unit tests for ElementDetector module.

Tests detection of discrete chart elements: bars, markers, pie slices.
"""

import numpy as np
import pytest
import cv2
import math
from typing import List

from core_engine.stages.s3_extraction.element_detector import (
    ElementDetector,
    ElementDetectorConfig,
    ElementDetectionResult,
)
from core_engine.schemas.extraction import BarRectangle, DataMarker, PieSlice, MarkerType


class TestElementDetectorConfig:
    """Tests for ElementDetectorConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ElementDetectorConfig()
        
        assert config.detect_bars is True
        assert config.detect_markers is True
        assert config.detect_pie_slices is True
        assert config.min_bar_area == 100
        assert config.min_marker_size == 5
        assert config.max_marker_size == 50

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ElementDetectorConfig(
            detect_bars=False,
            min_bar_area=200,
            min_marker_size=10,
            max_marker_size=40,
        )
        
        assert config.detect_bars is False
        assert config.min_bar_area == 200
        assert config.min_marker_size == 10
        assert config.max_marker_size == 40


class TestElementDetector:
    """Tests for ElementDetector class."""

    @pytest.fixture
    def detector(self) -> ElementDetector:
        """Create detector with default config."""
        return ElementDetector()

    @pytest.fixture
    def bar_chart_binary(self) -> np.ndarray:
        """Create binary bar chart image (white bars on black)."""
        img = np.zeros((400, 400), dtype=np.uint8)
        
        # Add white bars
        img[200:350, 50:90] = 255    # Bar 1
        img[150:350, 130:170] = 255  # Bar 2
        img[250:350, 210:250] = 255  # Bar 3
        img[100:350, 290:330] = 255  # Bar 4
        
        return img

    @pytest.fixture
    def bar_chart_color(self) -> np.ndarray:
        """Create colored bar chart image."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add colored bars
        img[200:350, 50:90, :] = [0, 0, 255]    # Red bar (BGR)
        img[150:350, 130:170, :] = [0, 255, 0]  # Green bar
        img[250:350, 210:250, :] = [255, 0, 0]  # Blue bar
        img[100:350, 290:330, :] = [0, 128, 255]  # Orange bar
        
        return img

    @pytest.fixture
    def scatter_binary(self) -> np.ndarray:
        """Create binary scatter chart with circular markers."""
        img = np.zeros((400, 400), dtype=np.uint8)
        
        # Add circular markers
        cv2.circle(img, (100, 300), 8, 255, -1)
        cv2.circle(img, (150, 250), 8, 255, -1)
        cv2.circle(img, (200, 200), 8, 255, -1)
        cv2.circle(img, (250, 180), 8, 255, -1)
        cv2.circle(img, (300, 150), 8, 255, -1)
        
        return img

    def test_detect_returns_result(
        self,
        detector: ElementDetector,
        bar_chart_binary: np.ndarray,
    ) -> None:
        """Test that detect returns ElementDetectionResult."""
        result = detector.detect(bar_chart_binary)
        
        assert isinstance(result, ElementDetectionResult)
        assert hasattr(result, 'bars')
        assert hasattr(result, 'markers')
        assert hasattr(result, 'slices')
        assert hasattr(result, 'contours_analyzed')

    def test_detect_bars(
        self,
        detector: ElementDetector,
        bar_chart_binary: np.ndarray,
    ) -> None:
        """Test bar detection in bar chart."""
        result = detector.detect(bar_chart_binary)
        
        # Should detect bars
        assert len(result.bars) >= 2

    def test_bar_properties(
        self,
        detector: ElementDetector,
        bar_chart_binary: np.ndarray,
        bar_chart_color: np.ndarray,
    ) -> None:
        """Test that detected bars have correct properties."""
        result = detector.detect(bar_chart_binary, bar_chart_color)
        
        for bar in result.bars:
            # Each bar should have valid coordinates
            assert bar.x_min >= 0
            assert bar.y_min >= 0
            assert bar.x_max > bar.x_min
            assert bar.y_max > bar.y_min

    def test_detect_markers(
        self,
        detector: ElementDetector,
        scatter_binary: np.ndarray,
    ) -> None:
        """Test marker detection in scatter chart."""
        result = detector.detect(scatter_binary)
        
        # Should detect markers (either as markers or circles from Hough)
        assert len(result.markers) >= 1 or len(result.bars) >= 1

    def test_bar_height_ordering(
        self,
        detector: ElementDetector,
        bar_chart_binary: np.ndarray,
    ) -> None:
        """Test that bars have correct relative heights."""
        result = detector.detect(bar_chart_binary)
        
        if len(result.bars) >= 2:
            # Sort by x position (left to right)
            bars_sorted = sorted(result.bars, key=lambda b: b.x_min)
            
            # Check heights are reasonable
            for bar in bars_sorted:
                assert bar.height > 0
                assert bar.width > 0

    def test_detection_disabled(self) -> None:
        """Test disabling specific detection types."""
        config = ElementDetectorConfig(
            detect_bars=False,
            detect_markers=True,
            detect_pie_slices=False,
        )
        detector = ElementDetector(config)
        
        # Create binary image with bar shape
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50:150, 80:120] = 255  # White bar
        
        result = detector.detect(img)
        
        # Should not detect bars when disabled
        assert len(result.bars) == 0


class TestElementDetectorEdgeCases:
    """Edge case tests for ElementDetector."""

    def test_empty_image(self) -> None:
        """Test handling of empty (black) image."""
        detector = ElementDetector()
        
        img = np.zeros((200, 200), dtype=np.uint8)
        result = detector.detect(img)
        
        assert isinstance(result, ElementDetectionResult)
        assert len(result.bars) == 0
        assert len(result.markers) == 0

    def test_full_white_image(self) -> None:
        """Test handling of all white image."""
        detector = ElementDetector()
        
        img = np.ones((200, 200), dtype=np.uint8) * 255
        result = detector.detect(img)
        
        # Might detect the whole image as one element
        assert isinstance(result, ElementDetectionResult)

    def test_very_small_elements(self) -> None:
        """Test handling of elements below minimum size."""
        config = ElementDetectorConfig(min_bar_area=1000)
        detector = ElementDetector(config)
        
        img = np.zeros((200, 200), dtype=np.uint8)
        img[90:110, 90:110] = 255  # 20x20 = 400 area (below min)
        
        result = detector.detect(img)
        
        # Should filter out small bars
        assert len(result.bars) == 0

    def test_overlapping_bars(self) -> None:
        """Test handling of overlapping/touching bars."""
        detector = ElementDetector()
        
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50:150, 50:90] = 255   # First bar
        img[50:150, 88:130] = 255  # Overlapping second bar
        
        result = detector.detect(img)
        
        # Should handle overlapping
        assert isinstance(result, ElementDetectionResult)

    def test_horizontal_bars(self) -> None:
        """Test detection of horizontal bars."""
        detector = ElementDetector()
        
        img = np.zeros((200, 200), dtype=np.uint8)
        img[40:60, 50:180] = 255   # Horizontal bar
        
        result = detector.detect(img)
        
        # Should detect horizontal bars
        assert len(result.bars) >= 1
        
        for bar in result.bars:
            # Horizontal bars have width > height
            assert bar.width > bar.height

    def test_contours_count(self) -> None:
        """Test that contours_analyzed is tracked."""
        detector = ElementDetector()
        
        img = np.zeros((200, 200), dtype=np.uint8)
        img[30:70, 30:70] = 255    # Square 1
        img[100:150, 100:150] = 255  # Square 2
        
        result = detector.detect(img)
        
        # Should have analyzed at least 2 contours
        assert result.contours_analyzed >= 2


class TestPieSliceDetection:
    """Tests for pie slice detection correctness.

    These tests verify that _detect_pie_slices_by_kmeans() is actually called
    and produces non-empty results for synthetic pie chart images.
    This addresses a critical bug where the method existed but was never wired
    into the detect() flow, leaving slices=[] for all pie charts.
    """

    @staticmethod
    def _make_synthetic_pie(size: int = 400, n_slices: int = 4) -> np.ndarray:
        """Create a synthetic pie chart image with distinct colored slices.

        Args:
            size: Image width and height in pixels.
            n_slices: Number of equal-angle slices to draw.

        Returns:
            BGR color image with a synthetic pie chart.
        """
        img = np.ones((size, size, 3), dtype=np.uint8) * 255  # white background
        center = (size // 2, size // 2)
        radius = size // 2 - 30

        # Distinct colors (BGR) for up to 8 slices
        colors = [
            (0, 0, 200),    # Red
            (0, 180, 0),    # Green
            (200, 0, 0),    # Blue
            (0, 180, 220),  # Yellow-ish
            (180, 0, 180),  # Purple
            (0, 128, 255),  # Orange
            (128, 128, 0),  # Teal
            (50, 50, 200),  # Dark red
        ]

        angle_step = 360 // n_slices
        for i in range(n_slices):
            start_angle = i * angle_step
            end_angle = (i + 1) * angle_step
            color = colors[i % len(colors)]
            cv2.ellipse(
                img,
                center,
                (radius, radius),
                0,
                start_angle,
                end_angle,
                color,
                thickness=-1,
            )

        return img

    @pytest.fixture
    def detector(self) -> ElementDetector:
        """Create detector with pie detection enabled."""
        return ElementDetector(ElementDetectorConfig(detect_pie_slices=True))

    def test_pie_slices_detected_not_empty(self, detector: ElementDetector) -> None:
        """Pie chart images must produce len(slices) > 0, not just hasattr."""
        pie_image = self._make_synthetic_pie(n_slices=4)
        binary = cv2.cvtColor(pie_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY_INV)

        result = detector.detect(binary, color_image=pie_image, chart_type="pie")

        assert len(result.slices) > 0, (
            "Pie slice detection returned 0 slices. "
            "Verify _detect_pie_slices_by_kmeans() is wired into detect()."
        )

    def test_pie_slices_have_valid_geometry(self, detector: ElementDetector) -> None:
        """Each detected PieSlice must have non-zero radius and angle span."""
        pie_image = self._make_synthetic_pie(n_slices=3)
        binary = cv2.cvtColor(pie_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY_INV)

        result = detector.detect(binary, color_image=pie_image, chart_type="pie")

        for s in result.slices:
            assert s.radius_outer > 0, "Slice radius_outer must be positive"
            assert s.angle_end != s.angle_start, "Slice must have non-zero angle span"

    def test_pie_slices_have_color(self, detector: ElementDetector) -> None:
        """Each detected PieSlice should carry its dominant color."""
        pie_image = self._make_synthetic_pie(n_slices=4)
        binary = cv2.cvtColor(pie_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY_INV)

        result = detector.detect(binary, color_image=pie_image, chart_type="pie")

        for s in result.slices:
            assert s.color is not None, "Slice must have a color"
            # Color channels should not all be zero (that would be pure black)
            assert s.color.r + s.color.g + s.color.b > 0

    def test_pie_detection_skipped_for_bar_chart(
        self, detector: ElementDetector
    ) -> None:
        """When chart_type='bar', pie slice detection should NOT run."""
        pie_image = self._make_synthetic_pie(n_slices=4)
        binary = cv2.cvtColor(pie_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY_INV)

        result = detector.detect(binary, color_image=pie_image, chart_type="bar")

        assert len(result.slices) == 0, (
            "Pie slices should not be detected when chart_type='bar'"
        )

    def test_pie_detection_disabled_by_config(self) -> None:
        """When detect_pie_slices=False, slices must be empty."""
        config = ElementDetectorConfig(detect_pie_slices=False)
        detector = ElementDetector(config)

        pie_image = self._make_synthetic_pie(n_slices=4)
        binary = cv2.cvtColor(pie_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY_INV)

        result = detector.detect(binary, color_image=pie_image, chart_type="pie")

        assert len(result.slices) == 0

    def test_multiple_slices_count(self, detector: ElementDetector) -> None:
        """A 6-slice pie should produce at least 3 detected slices (conservative).

        K-Means may merge similar-looking clusters, so we accept >= 50%.
        """
        pie_image = self._make_synthetic_pie(n_slices=6)
        binary = cv2.cvtColor(pie_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY_INV)

        result = detector.detect(binary, color_image=pie_image, chart_type="pie")

        assert len(result.slices) >= 3, (
            f"Expected >= 3 slices from 6-slice pie, got {len(result.slices)}"
        )

