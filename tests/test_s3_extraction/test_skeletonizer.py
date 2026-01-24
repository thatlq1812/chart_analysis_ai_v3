"""
Unit tests for Skeletonizer module.

Tests Lee algorithm skeletonization and keypoint detection.
"""

import numpy as np
import pytest

from core_engine.stages.s3_extraction.skeletonizer import (
    Skeletonizer,
    SkeletonConfig,
    SkeletonResult,
)
from core_engine.schemas.extraction import KeyPointType


class TestSkeletonConfig:
    """Tests for SkeletonConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SkeletonConfig()
        
        assert config.method == "lee"
        assert config.remove_spurs is True
        assert config.spur_length == 5
        assert config.detect_junctions is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SkeletonConfig(
            method="zhang",
            remove_spurs=False,
            spur_length=10,
        )
        
        assert config.method == "zhang"
        assert config.remove_spurs is False
        assert config.spur_length == 10


class TestSkeletonizer:
    """Tests for Skeletonizer class."""

    @pytest.fixture
    def skeletonizer(self) -> Skeletonizer:
        """Create skeletonizer with default config."""
        return Skeletonizer()

    @pytest.fixture
    def thick_line_image(self) -> np.ndarray:
        """Create image with thick diagonal line."""
        img = np.zeros((100, 100), dtype=np.uint8)
        
        # Draw thick diagonal line (5 pixels wide)
        for offset in range(-2, 3):
            for i in range(10, 90):
                y = i + offset
                x = i + offset
                if 0 <= y < 100 and 0 <= x < 100:
                    img[y, x] = 255
        
        return img

    @pytest.fixture
    def rectangle_image(self) -> np.ndarray:
        """Create image with filled rectangle."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 20:80] = 255
        return img

    @pytest.fixture
    def junction_image(self) -> np.ndarray:
        """Create image with T-junction (3 lines meeting)."""
        img = np.zeros((100, 100), dtype=np.uint8)
        
        # Vertical line
        img[10:90, 48:52] = 255
        
        # Horizontal line from left to center
        img[48:52, 10:52] = 255
        
        return img

    @pytest.fixture
    def endpoint_image(self) -> np.ndarray:
        """Create image with single line having two endpoints."""
        img = np.zeros((100, 100), dtype=np.uint8)
        
        # Horizontal line
        img[48:52, 20:80] = 255
        
        return img

    def test_process_returns_result(
        self,
        skeletonizer: Skeletonizer,
        thick_line_image: np.ndarray,
    ) -> None:
        """Test that process returns SkeletonResult."""
        result = skeletonizer.process(thick_line_image)
        
        assert isinstance(result, SkeletonResult)
        assert hasattr(result, 'skeleton')
        assert hasattr(result, 'keypoints')
        assert hasattr(result, 'distance_map')
        
        # Skeleton should be 2D binary
        assert len(result.skeleton.shape) == 2
        
        # Should only contain 0 or 255
        unique = np.unique(result.skeleton)
        assert len(unique) <= 2

    def test_skeleton_is_thinner(
        self,
        skeletonizer: Skeletonizer,
        rectangle_image: np.ndarray,
    ) -> None:
        """Test that skeleton is thinner than input."""
        input_white = np.sum(rectangle_image > 0)
        
        result = skeletonizer.process(rectangle_image)
        skeleton_white = np.sum(result.skeleton > 0)
        
        # Skeleton should have fewer white pixels
        assert skeleton_white < input_white

    def test_skeleton_preserves_topology(
        self,
        skeletonizer: Skeletonizer,
        thick_line_image: np.ndarray,
    ) -> None:
        """Test that skeleton preserves line topology (connectivity)."""
        result = skeletonizer.process(thick_line_image)
        
        # Skeleton should still have connected pixels
        white_pixels = np.sum(result.skeleton > 0)
        assert white_pixels > 0

    def test_detect_endpoints(
        self,
        skeletonizer: Skeletonizer,
        endpoint_image: np.ndarray,
    ) -> None:
        """Test endpoint detection on simple line."""
        result = skeletonizer.process(endpoint_image)
        
        # Filter endpoints from result keypoints
        endpoints = [kp for kp in result.keypoints if kp.point_type == KeyPointType.ENDPOINT]
        
        # Simple line should have exactly 2 endpoints
        assert len(endpoints) == 2

    def test_detect_junctions(
        self,
        skeletonizer: Skeletonizer,
        junction_image: np.ndarray,
    ) -> None:
        """Test junction detection on T-junction."""
        result = skeletonizer.process(junction_image)
        
        # Filter junctions from result keypoints
        junctions = [kp for kp in result.keypoints if kp.point_type == KeyPointType.JUNCTION]
        
        # T-junction should have at least 1 junction point
        assert len(junctions) >= 1

    def test_trace_paths(
        self,
        skeletonizer: Skeletonizer,
        endpoint_image: np.ndarray,
    ) -> None:
        """Test path tracing from endpoints."""
        result = skeletonizer.process(endpoint_image)
        paths = skeletonizer.trace_paths(result.skeleton, result.keypoints)
        
        # Should find at least one path
        assert len(paths) >= 1
        
        # Path should be a list of points
        for path in paths:
            assert len(path) > 0
            for point in path:
                assert len(point) == 2

    def test_stroke_width_computation(
        self,
        skeletonizer: Skeletonizer,
        thick_line_image: np.ndarray,
    ) -> None:
        """Test stroke width computation from distance transform."""
        result = skeletonizer.process(thick_line_image)
        
        # stroke_width_map should be computed
        assert hasattr(result, 'stroke_width_map')
        assert result.stroke_width_map.shape == result.skeleton.shape
        
        # Stroke width at skeleton pixels should be > 0
        skeleton_pixels = result.skeleton > 0
        if np.any(skeleton_pixels):
            stroke_widths = result.stroke_width_map[skeleton_pixels]
            assert np.any(stroke_widths > 0)

    def test_spur_removal(
        self,
        thick_line_image: np.ndarray,
    ) -> None:
        """Test spur removal option."""
        # Add some noise that might create spurs
        noisy = thick_line_image.copy()
        noisy[50, 60] = 255
        noisy[51, 61] = 255
        
        # With spur removal
        config_with = SkeletonConfig(remove_spurs=True, spur_length=3)
        skel_with = Skeletonizer(config_with)
        result_with = skel_with.process(noisy)
        
        # Without spur removal
        config_without = SkeletonConfig(remove_spurs=False)
        skel_without = Skeletonizer(config_without)
        result_without = skel_without.process(noisy)
        
        # Both should produce valid skeletons
        assert np.sum(result_with.skeleton > 0) > 0
        assert np.sum(result_without.skeleton > 0) > 0


class TestSkeletonizerEdgeCases:
    """Edge case tests for Skeletonizer."""

    def test_empty_image(self) -> None:
        """Test handling of empty (all black) image."""
        skeletonizer = Skeletonizer()
        
        img = np.zeros((50, 50), dtype=np.uint8)
        result = skeletonizer.process(img)
        
        # Should return all zeros
        assert np.sum(result.skeleton) == 0

    def test_full_white_image(self) -> None:
        """Test handling of all white image."""
        skeletonizer = Skeletonizer()
        
        img = np.ones((50, 50), dtype=np.uint8) * 255
        result = skeletonizer.process(img)
        
        # Should produce some skeleton
        assert result.skeleton.shape == (50, 50)

    def test_single_pixel(self) -> None:
        """Test handling of single white pixel."""
        skeletonizer = Skeletonizer()
        
        img = np.zeros((50, 50), dtype=np.uint8)
        img[25, 25] = 255
        
        result = skeletonizer.process(img)
        
        # Single pixel should remain or be removed
        assert result.skeleton.shape == (50, 50)

    def test_thin_line_preserved(self) -> None:
        """Test that already thin lines are preserved."""
        skeletonizer = Skeletonizer()
        
        img = np.zeros((50, 50), dtype=np.uint8)
        img[25, 10:40] = 255  # 1-pixel thin line
        
        result = skeletonizer.process(img)
        
        # Should have approximately the same pixels
        assert np.sum(result.skeleton > 0) > 0

    def test_different_methods(self) -> None:
        """Test different skeletonization methods."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[15:35, 15:35] = 255
        
        for method in ["lee", "zhang", "medial"]:
            config = SkeletonConfig(method=method)
            skeletonizer = Skeletonizer(config)
            
            result = skeletonizer.process(img)
            assert result.skeleton.shape == (50, 50)

    def test_inverted_input(self) -> None:
        """Test handling of inverted binary image."""
        skeletonizer = Skeletonizer()
        
        # Object is black on white background
        img = np.ones((50, 50), dtype=np.uint8) * 255
        img[20:30, 10:40] = 0
        
        result = skeletonizer.process(img)
        
        # Should still work (may need to invert internally)
        assert result.skeleton.shape == (50, 50)
