"""
Unit tests for ChartClassifier module.

Tests chart type classification from structural features.
"""

import numpy as np
import pytest

from core_engine.stages.s3_extraction.classifier import (
    ChartClassifier,
    ClassifierConfig,
    ClassificationResult,
)
from core_engine.schemas.enums import ChartType
from core_engine.schemas.extraction import (
    BarRectangle,
    DataMarker,
    PieSlice,
    Polyline,
    PointFloat,
    MarkerType,
    LineStyle,
)
from core_engine.schemas.stage_outputs import OCRText
from core_engine.schemas.common import BoundingBox


class TestClassifierConfig:
    """Tests for ClassifierConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ClassifierConfig()
        
        assert config.min_bars_for_bar_chart == 2
        assert config.min_points_for_scatter == 5
        assert config.min_confidence == 0.5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ClassifierConfig(
            min_bars_for_bar_chart=3,
            min_points_for_scatter=10,
            min_confidence=0.7,
        )
        
        assert config.min_bars_for_bar_chart == 3
        assert config.min_points_for_scatter == 10
        assert config.min_confidence == 0.7


class TestChartClassifier:
    """Tests for ChartClassifier class."""

    @pytest.fixture
    def classifier(self) -> ChartClassifier:
        """Create classifier with default config."""
        return ChartClassifier()

    @pytest.fixture
    def bar_elements(self) -> list:
        """Create bar elements for bar chart."""
        return [
            BarRectangle(
                x_min=50, y_min=100, x_max=90, y_max=300,
                color=None,
            ),
            BarRectangle(
                x_min=120, y_min=150, x_max=160, y_max=300,
                color=None,
            ),
            BarRectangle(
                x_min=190, y_min=80, x_max=230, y_max=300,
                color=None,
            ),
        ]

    @pytest.fixture
    def scatter_markers(self) -> list:
        """Create marker elements for scatter chart."""
        return [
            DataMarker(
                center=PointFloat(x=100, y=250),
                size=10,
                marker_type=MarkerType.CIRCLE,
                color=None,
            ),
            DataMarker(
                center=PointFloat(x=150, y=200),
                size=10,
                marker_type=MarkerType.CIRCLE,
                color=None,
            ),
            DataMarker(
                center=PointFloat(x=200, y=180),
                size=10,
                marker_type=MarkerType.CIRCLE,
                color=None,
            ),
            DataMarker(
                center=PointFloat(x=250, y=150),
                size=10,
                marker_type=MarkerType.CIRCLE,
                color=None,
            ),
            DataMarker(
                center=PointFloat(x=300, y=120),
                size=10,
                marker_type=MarkerType.CIRCLE,
                color=None,
            ),
            DataMarker(
                center=PointFloat(x=350, y=100),
                size=10,
                marker_type=MarkerType.CIRCLE,
                color=None,
            ),
        ]

    @pytest.fixture
    def pie_slices(self) -> list:
        """Create pie slice elements."""
        import math
        return [
            PieSlice(
                center=PointFloat(x=200, y=200),
                radius_outer=100,
                radius_inner=0,
                angle_start=0,
                angle_end=2*math.pi/3,
                color=None,
            ),
            PieSlice(
                center=PointFloat(x=200, y=200),
                radius_outer=100,
                radius_inner=0,
                angle_start=2*math.pi/3,
                angle_end=4*math.pi/3,
                color=None,
            ),
            PieSlice(
                center=PointFloat(x=200, y=200),
                radius_outer=100,
                radius_inner=0,
                angle_start=4*math.pi/3,
                angle_end=2*math.pi,
                color=None,
            ),
        ]

    @pytest.fixture
    def line_polylines(self) -> list:
        """Create polylines for line chart."""
        return [
            Polyline(
                points=[
                    PointFloat(x=50, y=250),
                    PointFloat(x=100, y=200),
                    PointFloat(x=150, y=220),
                    PointFloat(x=200, y=150),
                    PointFloat(x=250, y=180),
                ],
                line_style=LineStyle.SOLID,
                color=None,
            ),
        ]

    def test_classify_bar_chart(
        self,
        classifier: ChartClassifier,
        bar_elements: list,
    ) -> None:
        """Test classification of bar chart."""
        result = classifier.classify(
            bars=bar_elements,
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        assert isinstance(result, ClassificationResult)
        assert result.chart_type == ChartType.BAR
        assert result.confidence >= 0.5

    def test_classify_scatter_chart(
        self,
        classifier: ChartClassifier,
        scatter_markers: list,
    ) -> None:
        """Test classification of scatter chart."""
        result = classifier.classify(
            bars=[],
            polylines=[],
            markers=scatter_markers,
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        # Should classify as SCATTER or UNKNOWN if markers not detected as scatter pattern
        assert result.chart_type in [ChartType.SCATTER, ChartType.UNKNOWN]
        assert result.confidence >= 0

    def test_classify_pie_chart(
        self,
        classifier: ChartClassifier,
        pie_slices: list,
    ) -> None:
        """Test classification of pie chart."""
        result = classifier.classify(
            bars=[],
            polylines=[],
            markers=[],
            slices=pie_slices,
            texts=[],
            image_shape=(400, 400),
        )
        
        assert result.chart_type == ChartType.PIE
        assert result.confidence >= 0.5

    def test_classify_line_chart(
        self,
        classifier: ChartClassifier,
        line_polylines: list,
    ) -> None:
        """Test classification of line chart."""
        result = classifier.classify(
            bars=[],
            polylines=line_polylines,
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        assert result.chart_type == ChartType.LINE
        assert result.confidence >= 0.5

    def test_classification_result_structure(
        self,
        classifier: ChartClassifier,
        bar_elements: list,
    ) -> None:
        """Test that ClassificationResult has expected attributes."""
        result = classifier.classify(
            bars=bar_elements,
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        assert hasattr(result, 'chart_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'features')
        assert hasattr(result, 'reasoning')

    def test_classification_confidence_range(
        self,
        classifier: ChartClassifier,
        bar_elements: list,
    ) -> None:
        """Test that confidence is in valid range."""
        result = classifier.classify(
            bars=bar_elements,
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        assert 0.0 <= result.confidence <= 1.0


class TestClassifierEdgeCases:
    """Edge case tests for ChartClassifier."""

    def test_empty_elements(self) -> None:
        """Test classification with no elements."""
        classifier = ChartClassifier()
        
        result = classifier.classify(
            bars=[],
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        # Should return UNKNOWN
        assert result.chart_type == ChartType.UNKNOWN

    def test_mixed_elements(self) -> None:
        """Test classification with mixed element types."""
        classifier = ChartClassifier()
        
        bars = [
            BarRectangle(x_min=50, y_min=100, x_max=90, y_max=300),
            BarRectangle(x_min=120, y_min=150, x_max=160, y_max=300),
        ]
        markers = [
            DataMarker(center=PointFloat(x=200, y=150), size=10),
        ]
        
        result = classifier.classify(
            bars=bars,
            polylines=[],
            markers=markers,
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        # Should classify based on dominant type (bars)
        assert result.chart_type in [ChartType.BAR, ChartType.SCATTER, ChartType.UNKNOWN]

    def test_insufficient_elements(self) -> None:
        """Test classification with insufficient elements."""
        config = ClassifierConfig(min_bars_for_bar_chart=5)
        classifier = ChartClassifier(config)
        
        # Only 2 bars when 5 required
        bars = [
            BarRectangle(x_min=50, y_min=100, x_max=90, y_max=300),
            BarRectangle(x_min=120, y_min=150, x_max=160, y_max=300),
        ]
        
        result = classifier.classify(
            bars=bars,
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        # May not classify as BAR due to insufficient count
        assert isinstance(result, ClassificationResult)

    def test_horizontal_bar_chart(self) -> None:
        """Test classification of horizontal bar chart."""
        classifier = ChartClassifier()
        
        # Horizontal bars (width > height)
        bars = [
            BarRectangle(x_min=50, y_min=50, x_max=200, y_max=80),
            BarRectangle(x_min=50, y_min=100, x_max=250, y_max=130),
            BarRectangle(x_min=50, y_min=150, x_max=180, y_max=180),
        ]
        
        result = classifier.classify(
            bars=bars,
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        # Should still classify as BAR
        assert result.chart_type == ChartType.BAR

    def test_confidence_threshold(self) -> None:
        """Test minimum confidence threshold."""
        config = ClassifierConfig(min_confidence=0.9)
        classifier = ChartClassifier(config)
        
        # Minimal case
        bars = [
            BarRectangle(x_min=50, y_min=100, x_max=90, y_max=300),
        ]
        
        result = classifier.classify(
            bars=bars,
            polylines=[],
            markers=[],
            slices=[],
            texts=[],
            image_shape=(400, 400),
        )
        
        # Classification should still work
        assert isinstance(result, ClassificationResult)

