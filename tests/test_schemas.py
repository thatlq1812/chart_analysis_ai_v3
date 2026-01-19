"""
Tests for common schemas.
"""

import pytest
from src.core_engine.schemas import BoundingBox, Color, Point, ChartType


class TestBoundingBox:
    """Tests for BoundingBox schema."""
    
    def test_create_valid_bbox(self):
        """Should create valid bounding box."""
        bbox = BoundingBox(
            x_min=10,
            y_min=20,
            x_max=100,
            y_max=150,
            confidence=0.95,
        )
        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 100
        assert bbox.y_max == 150
        assert bbox.confidence == 0.95
    
    def test_width_property(self, sample_bbox):
        """Should calculate correct width."""
        assert sample_bbox.width == 90  # 100 - 10
    
    def test_height_property(self, sample_bbox):
        """Should calculate correct height."""
        assert sample_bbox.height == 130  # 150 - 20
    
    def test_area_property(self, sample_bbox):
        """Should calculate correct area."""
        assert sample_bbox.area == 90 * 130
    
    def test_center_property(self, sample_bbox):
        """Should calculate correct center."""
        center = sample_bbox.center
        assert center == (55, 85)  # ((10+100)//2, (20+150)//2)
    
    def test_to_xyxy(self, sample_bbox):
        """Should return xyxy tuple."""
        assert sample_bbox.to_xyxy() == (10, 20, 100, 150)
    
    def test_to_xywh(self, sample_bbox):
        """Should return xywh tuple."""
        assert sample_bbox.to_xywh() == (10, 20, 90, 130)
    
    def test_invalid_negative_coords(self):
        """Should reject negative coordinates."""
        with pytest.raises(ValueError):
            BoundingBox(x_min=-10, y_min=20, x_max=100, y_max=150)
    
    def test_confidence_bounds(self):
        """Should reject confidence outside [0, 1]."""
        with pytest.raises(ValueError):
            BoundingBox(x_min=10, y_min=20, x_max=100, y_max=150, confidence=1.5)


class TestColor:
    """Tests for Color schema."""
    
    def test_create_valid_color(self):
        """Should create valid color."""
        color = Color(r=255, g=128, b=64)
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64
    
    def test_hex_property(self, sample_color):
        """Should return correct hex string."""
        assert sample_color.hex == "#ff8040"
    
    def test_rgb_tuple(self, sample_color):
        """Should return RGB tuple."""
        assert sample_color.rgb_tuple == (255, 128, 64)
    
    def test_from_hex(self):
        """Should create color from hex string."""
        color = Color.from_hex("#ff8040")
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64
    
    def test_from_hex_without_hash(self):
        """Should handle hex without # prefix."""
        color = Color.from_hex("ff8040")
        assert color.r == 255
    
    def test_distance_to(self):
        """Should calculate color distance."""
        c1 = Color(r=0, g=0, b=0)
        c2 = Color(r=255, g=255, b=255)
        distance = c1.distance_to(c2)
        assert distance == pytest.approx(441.67, rel=0.01)
    
    def test_invalid_channel_value(self):
        """Should reject values outside [0, 255]."""
        with pytest.raises(ValueError):
            Color(r=300, g=128, b=64)


class TestPoint:
    """Tests for Point schema."""
    
    def test_create_valid_point(self):
        """Should create valid point."""
        point = Point(x=100, y=200)
        assert point.x == 100
        assert point.y == 200
    
    def test_distance_to(self):
        """Should calculate distance to another point."""
        p1 = Point(x=0, y=0)
        p2 = Point(x=3, y=4)
        assert p1.distance_to(p2) == 5.0


class TestChartType:
    """Tests for ChartType enum."""
    
    def test_all_types_exist(self):
        """Should have all expected chart types."""
        types = [t.value for t in ChartType]
        assert "bar" in types
        assert "line" in types
        assert "pie" in types
        assert "scatter" in types
        assert "area" in types
        assert "unknown" in types
    
    def test_string_value(self):
        """Should work as string."""
        assert ChartType.BAR == "bar"
        assert ChartType.LINE.value == "line"
