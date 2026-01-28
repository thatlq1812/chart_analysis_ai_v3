"""
Unit tests for Stage 4: Value Mapper

Tests pixel-to-value conversion and calibration.
"""

import pytest
import numpy as np

from src.core_engine.schemas.common import BoundingBox, Color, Point
from src.core_engine.schemas.enums import ChartType
from src.core_engine.schemas.stage_outputs import (
    AxisInfo,
    ChartElement,
    DataSeries,
    OCRText,
    RawMetadata,
)
from src.core_engine.stages.s4_reasoning import (
    GeometricValueMapper,
    ValueMapperConfig,
    AxisMapping,
    MappingResult,
    ScaleType,
)


class TestAxisMapping:
    """Test AxisMapping pixel-to-value conversion."""
    
    def test_linear_mapping_non_inverted(self):
        """Test linear mapping without Y inversion."""
        mapping = AxisMapping(
            slope=1.0,
            intercept=0.0,
            pixel_min=0,
            pixel_max=100,
            value_min=0,
            value_max=100,
        )
        
        # Middle point
        value = mapping.pixel_to_value(50, inverted=False)
        assert abs(value - 50) < 0.1
        
        # Endpoints
        assert abs(mapping.pixel_to_value(0, inverted=False) - 0) < 0.1
        assert abs(mapping.pixel_to_value(100, inverted=False) - 100) < 0.1
    
    def test_linear_mapping_inverted(self):
        """Test linear mapping with Y inversion (image coordinates)."""
        mapping = AxisMapping(
            slope=1.0,
            intercept=0.0,
            pixel_min=0,
            pixel_max=200,
            value_min=0,
            value_max=100,
        )
        
        # In inverted mode, pixel 0 = max value, pixel 200 = min value
        value_at_top = mapping.pixel_to_value(0, inverted=True)
        value_at_bottom = mapping.pixel_to_value(200, inverted=True)
        
        assert abs(value_at_top - 100) < 0.1  # Top of image = max value
        assert abs(value_at_bottom - 0) < 0.1  # Bottom = min value
    
    def test_value_to_pixel_inverse(self):
        """Test value-to-pixel is inverse of pixel-to-value."""
        mapping = AxisMapping(
            slope=1.0,
            intercept=0.0,
            pixel_min=50,
            pixel_max=350,
            value_min=0,
            value_max=500,
        )
        
        # Round trip: pixel -> value -> pixel
        original_pixel = 150
        value = mapping.pixel_to_value(original_pixel, inverted=True)
        recovered_pixel = mapping.value_to_pixel(value, inverted=True)
        
        assert abs(recovered_pixel - original_pixel) < 1.0
    
    def test_non_zero_value_range(self):
        """Test mapping with non-zero value range."""
        mapping = AxisMapping(
            slope=1.0,
            intercept=0.0,
            pixel_min=100,
            pixel_max=400,
            value_min=2020,
            value_max=2025,
        )
        
        # Middle point
        value = mapping.pixel_to_value(250, inverted=True)
        assert 2022 <= value <= 2023


class TestGeometricValueMapper:
    """Test GeometricValueMapper class."""
    
    @pytest.fixture
    def sample_axis_info(self):
        """Create sample AxisInfo for testing."""
        return AxisInfo(
            x_axis_detected=True,
            y_axis_detected=True,
            x_min=0,
            x_max=100,
            y_min=0,
            y_max=500,
            x_scale_factor=3.0,  # 3 pixels per unit
            y_scale_factor=0.5,  # 0.5 pixels per unit
            x_calibration_confidence=0.9,
            y_calibration_confidence=0.85,
        )
    
    @pytest.fixture
    def sample_ocr_texts(self):
        """Create sample OCR texts for tick labels."""
        return [
            # Y-axis ticks (left side)
            OCRText(
                text="500",
                bbox=BoundingBox(x_min=10, y_min=50, x_max=40, y_max=70, confidence=0.9),
                confidence=0.95,
                role="ylabel",
            ),
            OCRText(
                text="250",
                bbox=BoundingBox(x_min=10, y_min=150, x_max=40, y_max=170, confidence=0.9),
                confidence=0.90,
                role="value",
            ),
            OCRText(
                text="0",
                bbox=BoundingBox(x_min=10, y_min=250, x_max=40, y_max=270, confidence=0.9),
                confidence=0.92,
                role="value",
            ),
            # X-axis ticks (bottom)
            OCRText(
                text="Q1",
                bbox=BoundingBox(x_min=100, y_min=280, x_max=130, y_max=300, confidence=0.9),
                confidence=0.88,
                role="xlabel",
            ),
            OCRText(
                text="Q4",
                bbox=BoundingBox(x_min=300, y_min=280, x_max=330, y_max=300, confidence=0.9),
                confidence=0.85,
                role="xlabel",
            ),
        ]
    
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = GeometricValueMapper()
        assert not mapper.is_calibrated
        assert mapper.x_mapping is None
        assert mapper.y_mapping is None
    
    def test_calibrate_from_axis_info(self, sample_axis_info):
        """Test calibration from AxisInfo."""
        mapper = GeometricValueMapper()
        success = mapper.calibrate_from_axis_info(
            sample_axis_info,
            image_width=500,
            image_height=400,
        )
        
        assert success
        assert mapper.is_calibrated
        assert mapper.y_mapping is not None
        assert mapper.y_mapping.value_min == 0
        assert mapper.y_mapping.value_max == 500
    
    def test_calibrate_from_tick_labels(self, sample_ocr_texts):
        """Test calibration from OCR tick labels."""
        mapper = GeometricValueMapper()
        success = mapper.calibrate_from_tick_labels(
            sample_ocr_texts,
            image_width=400,
            image_height=300,
        )
        
        assert success
        assert mapper.y_mapping is not None
    
    def test_pixel_to_value_y(self, sample_axis_info):
        """Test Y-axis pixel to value conversion."""
        mapper = GeometricValueMapper()
        mapper.calibrate_from_axis_info(sample_axis_info, 500, 400)
        
        result = mapper.pixel_to_value_y(150)  # Middle of image
        
        assert isinstance(result, MappingResult)
        assert result.confidence > 0
        # Value should be somewhere in middle of range
        assert 100 < result.mapped_value < 400
    
    def test_pixel_to_value_x(self, sample_axis_info):
        """Test X-axis pixel to value conversion."""
        mapper = GeometricValueMapper()
        mapper.calibrate_from_axis_info(sample_axis_info, 500, 400)
        
        result = mapper.pixel_to_value_x(225)  # ~middle
        
        assert isinstance(result, MappingResult)
    
    def test_map_point(self, sample_axis_info):
        """Test mapping a Point object."""
        mapper = GeometricValueMapper()
        mapper.calibrate_from_axis_info(sample_axis_info, 500, 400)
        
        point = Point(x=200, y=100)
        x_result, y_result = mapper.map_point(point)
        
        assert isinstance(x_result, MappingResult)
        assert isinstance(y_result, MappingResult)
    
    def test_map_elements_to_series(self, sample_axis_info):
        """Test mapping chart elements to series."""
        mapper = GeometricValueMapper()
        mapper.calibrate_from_axis_info(sample_axis_info, 500, 400)
        
        # Create test elements
        elements = [
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(x_min=100, y_min=100, x_max=150, y_max=200, confidence=0.9),
                center=Point(x=125, y=150),
                color=Color(r=66, g=133, b=244),
            ),
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(x_min=200, y_min=80, x_max=250, y_max=200, confidence=0.9),
                center=Point(x=225, y=140),
                color=Color(r=66, g=133, b=244),
            ),
        ]
        
        series = mapper.map_elements_to_series(elements, ChartType.BAR)
        
        assert len(series) == 1  # Same color = one series
        assert len(series[0].points) == 2
    
    def test_calibration_summary(self, sample_axis_info):
        """Test getting calibration summary."""
        mapper = GeometricValueMapper()
        mapper.calibrate_from_axis_info(sample_axis_info, 500, 400)
        
        summary = mapper.get_calibration_summary()
        
        assert summary["is_calibrated"]
        assert summary["y_axis"]["calibrated"]
        assert "range" in summary["y_axis"]
    
    def test_uncalibrated_mapping(self):
        """Test mapping returns raw values when uncalibrated."""
        mapper = GeometricValueMapper()
        
        result = mapper.pixel_to_value_y(150)
        
        assert result.mapped_value == 150  # Returns raw pixel
        assert result.confidence == 0.0  # No confidence
    
    def test_extrapolation_handling(self, sample_axis_info):
        """Test handling of extrapolated values."""
        config = ValueMapperConfig(clamp_to_axis_range=True)
        mapper = GeometricValueMapper(config)
        mapper.calibrate_from_axis_info(sample_axis_info, 500, 400)
        
        # Request value way outside range
        result = mapper.pixel_to_value_y(-100)  # Above image
        
        assert result.extrapolated


class TestValueMapperConfig:
    """Test ValueMapperConfig validation."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ValueMapperConfig()
        
        assert config.y_inverted is True
        assert config.round_integers is True
        assert config.decimal_places == 2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ValueMapperConfig(
            y_inverted=False,
            round_integers=False,
            decimal_places=4,
        )
        
        assert config.y_inverted is False
        assert config.round_integers is False
        assert config.decimal_places == 4
