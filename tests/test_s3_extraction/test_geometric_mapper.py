"""
Unit tests for GeometricMapper module.

Tests axis calibration and pixel-to-value mapping.
"""

import numpy as np
import pytest
from typing import List, Tuple

from core_engine.stages.s3_extraction.geometric_mapper import (
    GeometricMapper,
    MapperConfig,
    CalibrationResult,
)
from core_engine.schemas.extraction import PointFloat, ScaleMapping


class TestMapperConfig:
    """Tests for MapperConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MapperConfig()
        
        assert config.min_calibration_points == 2
        assert config.auto_detect_scale is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MapperConfig(
            min_calibration_points=3,
            auto_detect_scale=False,
        )
        
        assert config.min_calibration_points == 3
        assert config.auto_detect_scale is False


class TestGeometricMapper:
    """Tests for GeometricMapper class."""

    @pytest.fixture
    def mapper(self) -> GeometricMapper:
        """Create mapper with default config."""
        return GeometricMapper()

    @pytest.fixture
    def y_axis_tick_values(self) -> List[Tuple[float, float]]:
        """Create tick values for Y-axis: (pixel_y, value)."""
        # Y increases downward in image, value increases upward
        return [
            (390.0, 0.0),    # Bottom: pixel 390 = value 0
            (240.0, 50.0),   # Middle: pixel 240 = value 50
            (90.0, 100.0),   # Top: pixel 90 = value 100
        ]

    @pytest.fixture
    def x_axis_tick_values(self) -> List[Tuple[float, float]]:
        """Create tick values for X-axis: (pixel_x, value)."""
        return [
            (100.0, 1.0),
            (200.0, 2.0),
            (300.0, 3.0),
        ]

    def test_calibrate_y_axis_linear(
        self,
        mapper: GeometricMapper,
        y_axis_tick_values: List[Tuple[float, float]],
    ) -> None:
        """Test Y-axis calibration with linear scale."""
        result = mapper.calibrate_y_axis(y_axis_tick_values)
        
        assert result is not None
        assert isinstance(result, CalibrationResult)
        assert result.scale is not None
        assert result.r_squared >= 0.95  # Good linear fit

    def test_pixel_to_value_y(
        self,
        mapper: GeometricMapper,
        y_axis_tick_values: List[Tuple[float, float]],
    ) -> None:
        """Test pixel to value conversion for Y-axis."""
        mapper.calibrate_y_axis(y_axis_tick_values)
        
        # Test known calibration points
        val_bottom = mapper.pixel_to_value_y(390)
        val_top = mapper.pixel_to_value_y(90)
        
        # Allow some tolerance
        assert val_bottom is not None
        assert val_top is not None
        assert abs(val_bottom - 0) < 10
        assert abs(val_top - 100) < 10

    def test_calibrate_x_axis_linear(
        self,
        mapper: GeometricMapper,
        x_axis_tick_values: List[Tuple[float, float]],
    ) -> None:
        """Test X-axis calibration with linear scale."""
        result = mapper.calibrate_x_axis(x_axis_tick_values)
        
        assert result is not None
        assert isinstance(result, CalibrationResult)
        assert result.scale is not None
        assert result.r_squared >= 0.95

    def test_pixel_to_value_x(
        self,
        mapper: GeometricMapper,
        x_axis_tick_values: List[Tuple[float, float]],
    ) -> None:
        """Test pixel to value conversion for X-axis."""
        mapper.calibrate_x_axis(x_axis_tick_values)
        
        val = mapper.pixel_to_value_x(200)
        assert val is not None
        assert abs(val - 2.0) < 0.5

    def test_fit_linear_regression(self, mapper: GeometricMapper) -> None:
        """Test linear regression fitting."""
        pixels = np.array([100.0, 200.0, 300.0, 400.0])
        values = np.array([40.0, 30.0, 20.0, 10.0])  # Inverted Y
        
        result = mapper._fit_least_squares(pixels, values)
        
        assert result is not None
        # Should have negative slope (inverted Y)
        assert result.scale.slope < 0
        # Good fit
        assert result.r_squared > 0.99

    def test_interpolation(
        self,
        mapper: GeometricMapper,
        y_axis_tick_values: List[Tuple[float, float]],
    ) -> None:
        """Test interpolation between calibration points."""
        mapper.calibrate_y_axis(y_axis_tick_values)
        
        # Middle point between 0 and 100
        mid_pixel = 240  # Approximately middle
        mid_value = mapper.pixel_to_value_y(mid_pixel)
        
        assert mid_value is not None
        # Should be around 50
        assert 40 < mid_value < 60

    def test_point_to_values(
        self,
        mapper: GeometricMapper,
        y_axis_tick_values: List[Tuple[float, float]],
        x_axis_tick_values: List[Tuple[float, float]],
    ) -> None:
        """Test converting a point to data values."""
        mapper.calibrate_x_axis(x_axis_tick_values)
        mapper.calibrate_y_axis(y_axis_tick_values)
        
        point = PointFloat(x=200.0, y=240.0)
        x_val, y_val = mapper.point_to_values(point)
        
        assert x_val is not None
        assert y_val is not None
        assert abs(x_val - 2.0) < 0.5
        assert abs(y_val - 50.0) < 10


class TestGeometricMapperEdgeCases:
    """Edge case tests for GeometricMapper."""

    def test_insufficient_calibration_points(self) -> None:
        """Test handling of insufficient calibration points."""
        config = MapperConfig(min_calibration_points=3)
        mapper = GeometricMapper(config)
        
        # Only 2 points when 3 required
        tick_values = [
            (380.0, 0.0),
            (80.0, 100.0),
        ]
        
        result = mapper.calibrate_y_axis(tick_values)
        
        # Should return None
        assert result is None

    def test_empty_tick_values(self) -> None:
        """Test handling of empty tick list."""
        mapper = GeometricMapper()
        
        result = mapper.calibrate_y_axis([])
        
        assert result is None

    def test_pixel_to_value_uncalibrated(self) -> None:
        """Test pixel_to_value when axis not calibrated."""
        mapper = GeometricMapper()
        
        # No calibration done
        value = mapper.pixel_to_value_y(100)
        
        assert value is None

    def test_logarithmic_scale_detection(self) -> None:
        """Test detection of logarithmic scale."""
        config = MapperConfig(auto_detect_scale=True)
        mapper = GeometricMapper(config)
        
        # Logarithmic values: 1, 10, 100, 1000 at equal pixel spacing
        tick_values = [
            (380.0, 1.0),
            (280.0, 10.0),
            (180.0, 100.0),
            (80.0, 1000.0),
        ]
        
        result = mapper.calibrate_y_axis(tick_values)
        
        assert result is not None
        # May detect log scale if fits better
        assert result.scale is not None

    def test_set_plot_boundaries(self) -> None:
        """Test setting plot boundaries."""
        mapper = GeometricMapper()
        
        mapper.set_plot_boundaries(50, 350, 50, 350)
        
        assert mapper.plot_x_min == 50
        assert mapper.plot_x_max == 350
        assert mapper.plot_y_min == 50
        assert mapper.plot_y_max == 350

    def test_normalize_point(self) -> None:
        """Test point normalization within plot area."""
        mapper = GeometricMapper()
        mapper.set_plot_boundaries(0, 100, 0, 100)
        
        point = PointFloat(x=50.0, y=50.0)
        normalized = mapper.normalize_point(point)
        
        assert abs(normalized.x - 0.5) < 0.01
        assert abs(normalized.y - 0.5) < 0.01

    def test_estimate_value_from_bar_height(self) -> None:
        """Test bar value estimation."""
        mapper = GeometricMapper()
        
        # Calibrate Y axis
        tick_values = [
            (400.0, 0.0),
            (200.0, 50.0),
            (0.0, 100.0),
        ]
        mapper.calibrate_y_axis(tick_values)
        
        # Bar from bottom to middle
        value = mapper.estimate_value_from_bar_height(
            bar_top_y=200.0,
            bar_bottom_y=400.0,
            baseline_value=0.0,
        )
        
        assert value is not None
        assert abs(value - 50.0) < 5

    def test_r_squared_quality(self) -> None:
        """Test R-squared value reflects fit quality."""
        mapper = GeometricMapper()
        
        # Perfect linear relationship
        perfect_ticks = [
            (100.0, 30.0),
            (200.0, 20.0),
            (300.0, 10.0),
        ]
        
        result = mapper.calibrate_y_axis(perfect_ticks)
        
        assert result is not None
        assert result.r_squared > 0.999

    def test_calibration_with_noise(self) -> None:
        """Test calibration with slightly noisy data."""
        mapper = GeometricMapper()
        
        # Slightly noisy data (not perfectly linear)
        noisy_ticks = [
            (100.0, 31.0),   # Expected ~30
            (200.0, 19.5),   # Expected ~20
            (300.0, 10.5),   # Expected ~10
        ]
        
        result = mapper.calibrate_y_axis(noisy_ticks)
        
        assert result is not None
        # Should still get good fit
        assert result.r_squared > 0.95

