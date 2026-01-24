"""
Extraction Schemas

Pydantic models for Stage 3 (Extraction) geometric data structures.
Implements the Geo-SLM vector-based representation for chart data.

Reference: docs/instruction_p2_research.md
"""

from typing import List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .common import Point, Color


# =============================================================================
# VECTOR GEOMETRY TYPES
# =============================================================================

class LineStyle(str, Enum):
    """Line stroke style detected from morphological analysis."""
    
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DASH_DOT = "dash_dot"
    UNKNOWN = "unknown"


class MarkerType(str, Enum):
    """Data point marker shape."""
    
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    CROSS = "cross"
    STAR = "star"
    NONE = "none"


class KeyPointType(str, Enum):
    """Type of keypoint in skeleton graph."""
    
    ENDPOINT = "endpoint"       # Terminal point of a line
    JUNCTION = "junction"       # Where multiple lines meet
    CORNER = "corner"           # Sharp angle change
    INFLECTION = "inflection"   # Curvature direction change
    VERTEX = "vertex"           # RDP-preserved vertex (data point)


# =============================================================================
# GEOMETRIC PRIMITIVES
# =============================================================================

class PointFloat(BaseModel):
    """
    2D point with sub-pixel precision.
    
    Supports float coordinates for accurate geometric calculations.
    """
    
    model_config = ConfigDict(frozen=True)
    
    x: float = Field(..., description="X coordinate (sub-pixel precision)")
    y: float = Field(..., description="Y coordinate (sub-pixel precision)")
    
    def to_int_point(self) -> Point:
        """Convert to integer Point."""
        return Point(x=round(self.x), y=round(self.y))
    
    def distance_to(self, other: "PointFloat") -> float:
        """Euclidean distance to another point."""
        import math
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class LineSegment(BaseModel):
    """
    A line segment in pixel space.
    
    Fundamental unit of vectorized chart data.
    """
    
    model_config = ConfigDict(frozen=True)
    
    start: PointFloat = Field(..., description="Start point")
    end: PointFloat = Field(..., description="End point")
    stroke_width: float = Field(default=1.0, gt=0, description="Estimated stroke width")
    
    @computed_field
    @property
    def length(self) -> float:
        """Segment length in pixels."""
        return self.start.distance_to(self.end)
    
    @computed_field
    @property
    def angle(self) -> float:
        """Angle in radians from horizontal."""
        import math
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.atan2(dy, dx)
    
    @computed_field
    @property
    def midpoint(self) -> PointFloat:
        """Midpoint of segment."""
        return PointFloat(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )


class KeyPoint(BaseModel):
    """
    A significant point detected in skeleton graph.
    
    Keypoints are vertices preserved by RDP algorithm,
    representing actual data points or structural features.
    """
    
    model_config = ConfigDict(frozen=True)
    
    point: PointFloat = Field(..., description="Coordinates")
    point_type: KeyPointType = Field(..., description="Type of keypoint")
    confidence: float = Field(default=1.0, ge=0, le=1)
    is_vertex: bool = Field(
        default=False,
        description="True if RDP-preserved vertex (actual data point)"
    )


# =============================================================================
# POLYLINE AND CURVE STRUCTURES
# =============================================================================

class Polyline(BaseModel):
    """
    A connected sequence of line segments.
    
    Represents a single data series or structural element.
    Uses piecewise linear representation (no smooth interpolation).
    """
    
    points: List[PointFloat] = Field(
        ...,
        min_length=2,
        description="Ordered vertices of polyline"
    )
    line_style: LineStyle = Field(default=LineStyle.SOLID)
    stroke_width: float = Field(default=1.0, gt=0)
    color: Optional[Color] = None
    
    # Morphological profile for series grouping
    segment_lengths: List[float] = Field(
        default_factory=list,
        description="Lengths of individual segments (for dash pattern)"
    )
    gap_lengths: List[float] = Field(
        default_factory=list,
        description="Lengths of gaps between segments (for dash pattern)"
    )
    
    @computed_field
    @property
    def total_length(self) -> float:
        """Total polyline length."""
        length = 0.0
        for i in range(len(self.points) - 1):
            length += self.points[i].distance_to(self.points[i + 1])
        return length
    
    @computed_field
    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return len(self.points)
    
    @computed_field
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Bounding box as (x_min, y_min, x_max, y_max)."""
        x_coords = [p.x for p in self.points]
        y_coords = [p.y for p in self.points]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_segments(self) -> List[LineSegment]:
        """Convert to list of LineSegments."""
        segments = []
        for i in range(len(self.points) - 1):
            segments.append(LineSegment(
                start=self.points[i],
                end=self.points[i + 1],
                stroke_width=self.stroke_width,
            ))
        return segments


# =============================================================================
# CHART ELEMENT REPRESENTATIONS
# =============================================================================

class BarRectangle(BaseModel):
    """
    A bar/rectangle element detected in bar charts.
    
    Extracted using contour approximation, not skeletonization.
    """
    
    model_config = ConfigDict(frozen=True)
    
    x_min: float = Field(..., description="Left edge")
    y_min: float = Field(..., description="Top edge")
    x_max: float = Field(..., description="Right edge")
    y_max: float = Field(..., description="Bottom edge")
    color: Optional[Color] = None
    
    @computed_field
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @computed_field
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @computed_field
    @property
    def center(self) -> PointFloat:
        return PointFloat(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2,
        )
    
    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height


class PieSlice(BaseModel):
    """
    A pie/donut slice element.
    
    Defined by center, radii, and angle range.
    """
    
    model_config = ConfigDict(frozen=True)
    
    center: PointFloat = Field(..., description="Pie center")
    radius_outer: float = Field(..., gt=0, description="Outer radius")
    radius_inner: float = Field(default=0, ge=0, description="Inner radius (0 for pie, >0 for donut)")
    angle_start: float = Field(..., description="Start angle in radians")
    angle_end: float = Field(..., description="End angle in radians")
    color: Optional[Color] = None
    
    @computed_field
    @property
    def angle_span(self) -> float:
        """Angular span in radians."""
        return abs(self.angle_end - self.angle_start)
    
    @computed_field
    @property
    def percentage(self) -> float:
        """Percentage of full circle."""
        import math
        return (self.angle_span / (2 * math.pi)) * 100


class DataMarker(BaseModel):
    """
    A data point marker detected in scatter/line charts.
    
    Detected using Hough transform and contour analysis.
    """
    
    model_config = ConfigDict(frozen=True)
    
    center: PointFloat = Field(..., description="Marker center")
    marker_type: MarkerType = Field(default=MarkerType.CIRCLE)
    size: float = Field(..., gt=0, description="Marker size (diameter/width)")
    color: Optional[Color] = None


# =============================================================================
# AXIS AND SCALE STRUCTURES
# =============================================================================

class AxisLine(BaseModel):
    """
    Detected axis line with tick marks.
    """
    
    start: PointFloat = Field(..., description="Axis start point")
    end: PointFloat = Field(..., description="Axis end point")
    is_horizontal: bool = Field(..., description="True if X-axis, False if Y-axis")
    tick_positions: List[PointFloat] = Field(
        default_factory=list,
        description="Detected tick mark positions"
    )


class ScaleMapping(BaseModel):
    """
    Mapping between pixel coordinates and data values.
    
    Supports linear and logarithmic scales.
    Built from OCR axis labels and geometric positions.
    """
    
    # Linear model: data_value = slope * pixel + intercept
    slope: float = Field(..., description="Pixels to value ratio")
    intercept: float = Field(..., description="Value at pixel 0")
    
    # Scale type
    is_logarithmic: bool = Field(default=False)
    log_base: float = Field(default=10.0, gt=1)
    
    # Calibration info
    num_calibration_points: int = Field(default=0, ge=0)
    fit_error: float = Field(default=0.0, ge=0, description="Least squares fit error")
    
    def pixel_to_value(self, pixel: float) -> float:
        """Convert pixel coordinate to data value."""
        import math
        if self.is_logarithmic:
            log_val = self.slope * pixel + self.intercept
            return math.pow(self.log_base, log_val)
        return self.slope * pixel + self.intercept
    
    def value_to_pixel(self, value: float) -> float:
        """Convert data value to pixel coordinate."""
        import math
        if self.is_logarithmic:
            log_val = math.log(value, self.log_base)
            return (log_val - self.intercept) / self.slope
        return (value - self.intercept) / self.slope


# =============================================================================
# SKELETON GRAPH REPRESENTATION
# =============================================================================

class SkeletonEdge(BaseModel):
    """
    An edge in the skeleton graph.
    """
    
    start_idx: int = Field(..., ge=0, description="Index of start keypoint")
    end_idx: int = Field(..., ge=0, description="Index of end keypoint")
    path_pixels: List[PointFloat] = Field(
        default_factory=list,
        description="Pixel path along edge"
    )
    stroke_width: float = Field(default=1.0, gt=0)


class SkeletonGraph(BaseModel):
    """
    Complete skeleton graph representation of chart.
    
    Topology-preserving structure for geometric analysis.
    """
    
    keypoints: List[KeyPoint] = Field(default_factory=list)
    edges: List[SkeletonEdge] = Field(default_factory=list)
    
    # Simplified polylines after RDP
    polylines: List[Polyline] = Field(default_factory=list)
    
    # Detected discrete elements
    bars: List[BarRectangle] = Field(default_factory=list)
    slices: List[PieSlice] = Field(default_factory=list)
    markers: List[DataMarker] = Field(default_factory=list)
    
    @computed_field
    @property
    def node_count(self) -> int:
        return len(self.keypoints)
    
    @computed_field
    @property
    def edge_count(self) -> int:
        return len(self.edges)


# =============================================================================
# EXTRACTION RESULT
# =============================================================================

class VectorizedChart(BaseModel):
    """
    Complete vectorized representation of a chart.
    
    Contains both geometric structure and extracted metadata.
    """
    
    chart_id: str = Field(..., description="Chart identifier from Stage 2")
    
    # Image metadata
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    is_negative_processed: bool = Field(default=True)
    
    # Skeleton graph
    skeleton: SkeletonGraph = Field(default_factory=SkeletonGraph)
    
    # Detected axes
    x_axis: Optional[AxisLine] = None
    y_axis: Optional[AxisLine] = None
    
    # Scale mappings
    x_scale: Optional[ScaleMapping] = None
    y_scale: Optional[ScaleMapping] = None
    
    # Processing metadata
    preprocessing_applied: List[str] = Field(
        default_factory=list,
        description="List of preprocessing operations applied"
    )
    vectorization_epsilon: float = Field(
        default=1.0,
        gt=0,
        description="RDP epsilon used for simplification"
    )
