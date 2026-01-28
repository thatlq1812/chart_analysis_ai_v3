"""
Unit tests for Vectorizer module.

Tests RDP (Ramer-Douglas-Peucker) algorithm and vector simplification.
"""

import numpy as np
import pytest
from typing import List, Tuple

from core_engine.stages.s3_extraction.vectorizer import (
    Vectorizer,
    VectorizeConfig,
    VectorizeResult,
)
from core_engine.schemas.extraction import PointFloat, Polyline


class TestVectorizeConfig:
    """Tests for VectorizeConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VectorizeConfig()
        
        assert config.epsilon == 2.0
        assert config.adaptive_epsilon is True
        assert config.subpixel_refinement is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VectorizeConfig(
            epsilon=5.0,
            adaptive_epsilon=False,
            subpixel_refinement=False,
        )
        
        assert config.epsilon == 5.0
        assert config.adaptive_epsilon is False
        assert config.subpixel_refinement is False


class TestVectorizer:
    """Tests for Vectorizer class."""

    @pytest.fixture
    def vectorizer(self) -> Vectorizer:
        """Create vectorizer with default config."""
        return Vectorizer()

    @pytest.fixture
    def straight_line_path(self) -> List[List[Tuple[int, int]]]:
        """Create path representing straight horizontal line."""
        return [[(x, 50) for x in range(20, 80)]]

    @pytest.fixture
    def diagonal_line_path(self) -> List[List[Tuple[int, int]]]:
        """Create path representing diagonal line."""
        return [[(20 + i, 20 + i) for i in range(60)]]

    @pytest.fixture
    def zigzag_path(self) -> List[List[Tuple[int, int]]]:
        """Create path with zigzag pattern."""
        path = []
        # First horizontal segment
        for i in range(20):
            path.append((10 + i, 30))
        # Down segment
        for i in range(20):
            path.append((30, 30 + i))
        # Second horizontal segment
        for i in range(20):
            path.append((30 + i, 50))
        return [path]

    @pytest.fixture
    def noisy_line_path(self) -> List[List[Tuple[int, int]]]:
        """Create path with small noise deviations."""
        path = []
        for x in range(20, 80):
            y_offset = int(np.sin(x * 0.2) * 2)
            path.append((x, 50 + y_offset))
        return [path]

    def test_process_returns_result(
        self,
        vectorizer: Vectorizer,
        straight_line_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test that process returns VectorizeResult."""
        result = vectorizer.process(straight_line_path)
        
        assert isinstance(result, VectorizeResult)
        assert hasattr(result, 'polylines')
        assert hasattr(result, 'vertices')
        assert hasattr(result, 'compression_ratio')

    def test_polylines_format(
        self,
        vectorizer: Vectorizer,
        straight_line_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test that polylines have correct format."""
        result = vectorizer.process(straight_line_path)
        
        assert len(result.polylines) > 0
        for polyline in result.polylines:
            assert isinstance(polyline, Polyline)
            assert len(polyline.points) >= 2
            for point in polyline.points:
                assert isinstance(point, PointFloat)

    def test_straight_line_simplification(
        self,
        vectorizer: Vectorizer,
        straight_line_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test that straight line is simplified to few points."""
        result = vectorizer.process(straight_line_path)
        
        assert len(result.polylines) >= 1
        
        # Straight line should simplify to approximately 2 points
        total_points = sum(len(p.points) for p in result.polylines)
        assert total_points <= 5  # Allow small tolerance

    def test_diagonal_line_simplification(
        self,
        vectorizer: Vectorizer,
        diagonal_line_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test diagonal line simplification."""
        result = vectorizer.process(diagonal_line_path)
        
        assert len(result.polylines) >= 1
        
        # Diagonal line should also simplify to few points
        for polyline in result.polylines:
            assert len(polyline.points) >= 2

    def test_zigzag_preserves_corners(
        self,
        vectorizer: Vectorizer,
        zigzag_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test that zigzag pattern preserves corner points."""
        result = vectorizer.process(zigzag_path)
        
        # Should detect multiple segments or preserve corners
        total_points = sum(len(p.points) for p in result.polylines)
        
        # Zigzag should have more points than straight line
        assert total_points >= 3

    def test_noisy_line_smoothing(
        self,
        noisy_line_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test that noisy line is smoothed."""
        # Higher epsilon for more smoothing
        config = VectorizeConfig(epsilon=5.0)
        vectorizer = Vectorizer(config)
        
        result = vectorizer.process(noisy_line_path)
        
        # Should simplify despite noise
        assert len(result.polylines) >= 1

    def test_rdp_algorithm_basic(self, vectorizer: Vectorizer) -> None:
        """Test RDP algorithm on simple point list."""
        # Points forming a straight line with one outlier
        points = [
            (0, 0),
            (10, 0),
            (20, 1),  # Slight deviation
            (30, 0),
            (40, 0),
        ]
        
        simplified = vectorizer._rdp_simplify(points, epsilon=2.0)
        
        # Should simplify to approximately 2 points (start and end)
        assert len(simplified) >= 2
        assert len(simplified) < len(points)

    def test_rdp_preserves_significant_points(self, vectorizer: Vectorizer) -> None:
        """Test that RDP preserves significant deviation points."""
        # Points forming L-shape
        points = [
            (0, 0),
            (10, 0),
            (20, 0),
            (20, 10),
            (20, 20),
        ]
        
        simplified = vectorizer._rdp_simplify(points, epsilon=1.0)
        
        # Should preserve the corner at (20, 0)
        assert len(simplified) >= 3

    def test_compression_ratio(
        self,
        vectorizer: Vectorizer,
        straight_line_path: List[List[Tuple[int, int]]],
    ) -> None:
        """Test compression ratio calculation."""
        result = vectorizer.process(straight_line_path)
        
        # Compression ratio should be between 0 and 1 for simplification
        assert result.compression_ratio >= 0
        assert result.total_points_before >= result.total_points_after


class TestVectorizerEdgeCases:
    """Edge case tests for Vectorizer."""

    def test_empty_paths(self) -> None:
        """Test handling of empty paths list."""
        vectorizer = Vectorizer()
        
        result = vectorizer.process([])
        
        assert isinstance(result, VectorizeResult)
        assert len(result.polylines) == 0

    def test_single_point_path(self) -> None:
        """Test handling of path with single point."""
        vectorizer = Vectorizer()
        
        result = vectorizer.process([[(25, 25)]])
        
        # Single point path should be skipped (needs >= 2 points)
        assert isinstance(result, VectorizeResult)

    def test_two_point_path(self) -> None:
        """Test handling of minimal valid path."""
        vectorizer = Vectorizer()
        
        result = vectorizer.process([[(10, 10), (50, 50)]])
        
        assert len(result.polylines) == 1
        assert len(result.polylines[0].points) == 2

    def test_multiple_disconnected_paths(self) -> None:
        """Test handling of multiple disconnected paths."""
        vectorizer = Vectorizer()
        
        paths = [
            [(x, 20) for x in range(10, 40)],  # First line
            [(x, 80) for x in range(60, 90)],  # Second line
        ]
        
        result = vectorizer.process(paths)
        
        # Should return multiple polylines
        assert len(result.polylines) == 2

    def test_large_epsilon(self) -> None:
        """Test with very large epsilon (aggressive simplification)."""
        config = VectorizeConfig(
            epsilon=100.0,
            adaptive_epsilon=False,
            use_curvature_adaptive=False,  # Disable to test pure large epsilon
            use_hierarchical=False,  # Disable to test pure RDP
        )
        vectorizer = Vectorizer(config)
        
        # Create complex path
        path = []
        for i in range(80):
            y = int(50 + 20 * np.sin(i * 0.1))
            path.append((10 + i, y))
        
        result = vectorizer.process([path])
        
        # Should simplify significantly
        total_points = sum(len(p.points) for p in result.polylines)
        assert total_points <= 10

    def test_small_epsilon(self) -> None:
        """Test with small epsilon (minimal simplification)."""
        config = VectorizeConfig(epsilon=0.1, adaptive_epsilon=False)
        vectorizer = Vectorizer(config)
        
        path = [(x, 50) for x in range(10, 30)]
        result = vectorizer.process([path])
        
        # Should keep more points with small epsilon
        assert len(result.polylines) >= 1

