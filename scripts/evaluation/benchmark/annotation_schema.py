"""
Benchmark Annotation Schema

Pydantic models defining the ground truth annotation format for the
Stage 3 ceiling experiment benchmark. Each annotated chart captures:

- Correct chart type and difficulty level
- All visible text with correct roles
- Element count and types
- Axis ranges (for Cartesian charts)
- Actual data values (gold standard)

Usage:
    from scripts.evaluation.benchmark.annotation_schema import ChartAnnotation
    annotation = ChartAnnotation.model_validate(json_data)
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    """Chart visual complexity for stratified analysis."""

    SIMPLE = "simple"           # Clean, single-series, no overlap
    MODERATE = "moderate"       # Multi-series, rotated labels, some clutter
    COMPLEX = "complex"         # Stacked, log-scale, dense data, overlapping


class AxisType(str, Enum):
    """Type of axis scale."""

    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    NONE = "none"               # No axis (pie, donut)


class TextRoleGT(str, Enum):
    """Ground truth text roles (superset of pipeline TextRole)."""

    TITLE = "title"
    SUBTITLE = "subtitle"
    X_AXIS_LABEL = "x_axis_label"
    Y_AXIS_LABEL = "y_axis_label"
    X_TICK = "x_tick"
    Y_TICK = "y_tick"
    LEGEND = "legend"
    DATA_LABEL = "data_label"
    ANNOTATION = "annotation"
    SOURCE = "source"           # "Source: ..." footnotes
    OTHER = "other"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class TextAnnotation(BaseModel):
    """Single text element visible in the chart."""

    text: str = Field(..., description="Exact text content as shown in chart")
    role: TextRoleGT = Field(..., description="Semantic role of this text")
    notes: Optional[str] = Field(
        None,
        description="Annotator notes (e.g. 'partially occluded', 'rotated 90 deg')",
    )


class AxisAnnotation(BaseModel):
    """Ground truth axis information for Cartesian charts."""

    x_axis_type: AxisType = Field(
        default=AxisType.NONE,
        description="Scale type of X axis",
    )
    y_axis_type: AxisType = Field(
        default=AxisType.NONE,
        description="Scale type of Y axis",
    )
    x_min: Optional[float] = Field(None, description="Minimum X value (numeric axes)")
    x_max: Optional[float] = Field(None, description="Maximum X value (numeric axes)")
    y_min: Optional[float] = Field(None, description="Minimum Y value (numeric axes)")
    y_max: Optional[float] = Field(None, description="Maximum Y value (numeric axes)")
    x_categories: Optional[List[str]] = Field(
        None,
        description="Category labels on X axis (for categorical axis)",
    )
    y_categories: Optional[List[str]] = Field(
        None,
        description="Category labels on Y axis (for categorical axis)",
    )
    x_label: Optional[str] = Field(None, description="X axis label text")
    y_label: Optional[str] = Field(None, description="Y axis label text")


class DataPoint(BaseModel):
    """Single data point in a series."""

    x: Optional[Union[float, str]] = Field(None, description="X value or category label")
    y: Optional[float] = Field(None, description="Y value")
    label: Optional[str] = Field(None, description="Data label (for pie slices, etc.)")
    value: Optional[float] = Field(None, description="Value (for pie slices, bar values)")
    percentage: Optional[float] = Field(
        None, ge=0, le=100,
        description="Percentage (for pie/donut slices)",
    )


class DataSeries(BaseModel):
    """A single data series in the chart."""

    name: Optional[str] = Field(None, description="Series name from legend")
    points: List[DataPoint] = Field(
        default_factory=list,
        description="Ordered data points in this series",
    )


class ElementAnnotation(BaseModel):
    """Ground truth element counts and types."""

    primary_element_type: str = Field(
        ...,
        description="Primary element type: bar, point, line, slice, area, box",
    )
    element_count: int = Field(
        ..., ge=0,
        description="Total number of primary visual elements",
    )
    series_count: int = Field(
        default=1, ge=1,
        description="Number of distinct data series",
    )
    has_grid_lines: bool = Field(default=False)
    has_legend: bool = Field(default=False)
    has_data_labels: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Complexity traits
# ---------------------------------------------------------------------------

class ComplexityTraits(BaseModel):
    """Specific visual complexity traits for fine-grained analysis."""

    is_stacked: bool = Field(default=False, description="Stacked bars/areas")
    is_grouped: bool = Field(default=False, description="Grouped/clustered bars")
    is_multi_series: bool = Field(default=False, description="Multiple data series")
    has_log_scale: bool = Field(default=False, description="Logarithmic axis")
    has_rotated_labels: bool = Field(default=False, description="Rotated tick labels")
    has_negative_values: bool = Field(default=False, description="Values below zero")
    has_overlapping_elements: bool = Field(
        default=False, description="Elements overlap visually",
    )
    is_3d: bool = Field(default=False, description="3D perspective rendering")
    is_donut: bool = Field(default=False, description="Donut variant of pie")
    is_exploded: bool = Field(default=False, description="Exploded pie/donut")
    has_error_bars: bool = Field(default=False, description="Error bars present")
    has_trend_line: bool = Field(default=False, description="Trend/regression line")
    has_secondary_axis: bool = Field(default=False, description="Dual Y axes")
    has_dense_data: bool = Field(
        default=False, description=">20 data points, visual clutter",
    )
    notes: Optional[str] = Field(
        None, description="Free-form notes on visual difficulty",
    )


# ---------------------------------------------------------------------------
# Top-level annotation
# ---------------------------------------------------------------------------

class ChartAnnotation(BaseModel):
    """
    Complete ground truth annotation for a single chart image.

    This is the gold standard used by the evaluation harness to measure
    Stage 3 extraction accuracy for the ceiling experiment.
    """

    # -- Identity --
    chart_id: str = Field(..., description="Unique chart identifier (filename stem)")
    image_path: str = Field(
        ..., description="Relative path from project root to chart image",
    )
    chart_type: str = Field(
        ...,
        description="Ground truth chart type (bar, line, pie, scatter, area, "
        "histogram, heatmap, box, stacked_bar, grouped_bar, donut)",
    )

    # -- Difficulty --
    difficulty: Difficulty = Field(..., description="Visual complexity level")
    complexity_traits: ComplexityTraits = Field(
        default_factory=ComplexityTraits,
    )

    # -- Text --
    title: Optional[str] = Field(None, description="Chart title text (exact)")
    texts: List[TextAnnotation] = Field(
        default_factory=list,
        description="All visible text elements with roles",
    )

    # -- Elements --
    elements: ElementAnnotation = Field(
        ..., description="Ground truth element counts and types",
    )

    # -- Axes --
    axis: Optional[AxisAnnotation] = Field(
        None,
        description="Axis information (None for pie/donut charts)",
    )

    # -- Data values --
    data_series: List[DataSeries] = Field(
        default_factory=list,
        description="All data series with values (gold standard)",
    )

    # -- Metadata --
    annotator: str = Field(
        default="human",
        description="Who created this annotation (human / ai-assisted)",
    )
    annotation_notes: Optional[str] = Field(
        None, description="General notes about this chart or annotation quality",
    )
    source_paper: Optional[str] = Field(
        None, description="ArXiv ID or source paper identifier",
    )

    @model_validator(mode="after")
    def validate_axis_for_cartesian(self) -> "ChartAnnotation":
        """Cartesian chart types should have axis info."""
        cartesian_types = {
            "bar", "line", "scatter", "area", "histogram",
            "box", "stacked_bar", "grouped_bar",
        }
        if self.chart_type in cartesian_types and self.axis is None:
            # Warn but do not fail - annotator might not have axis data
            pass
        return self

    def to_evaluation_dict(self) -> Dict[str, Any]:
        """
        Export a flat dict optimized for comparison with Stage 3 output.

        Returns:
            Dict with keys matching evaluation harness expectations:
            - chart_type: str
            - text_count: int
            - text_roles: dict[role, count]
            - element_count: int
            - element_type: str
            - series_count: int
            - x_min, x_max, y_min, y_max: float or None
            - data_point_count: int
        """
        text_roles: Dict[str, int] = {}
        for t in self.texts:
            text_roles[t.role.value] = text_roles.get(t.role.value, 0) + 1

        total_data_points = sum(len(s.points) for s in self.data_series)

        return {
            "chart_id": self.chart_id,
            "chart_type": self.chart_type,
            "difficulty": self.difficulty.value,
            "text_count": len(self.texts),
            "text_roles": text_roles,
            "title": self.title,
            "element_count": self.elements.element_count,
            "element_type": self.elements.primary_element_type,
            "series_count": self.elements.series_count,
            "has_legend": self.elements.has_legend,
            "has_data_labels": self.elements.has_data_labels,
            "x_min": self.axis.x_min if self.axis else None,
            "x_max": self.axis.x_max if self.axis else None,
            "y_min": self.axis.y_min if self.axis else None,
            "y_max": self.axis.y_max if self.axis else None,
            "x_axis_type": self.axis.x_axis_type.value if self.axis else None,
            "y_axis_type": self.axis.y_axis_type.value if self.axis else None,
            "data_point_count": total_data_points,
        }


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------

def export_json_schema(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Export the annotation schema as JSON Schema for external tools."""
    schema = ChartAnnotation.model_json_schema()
    if output_path:
        import json
        from pathlib import Path

        Path(output_path).write_text(
            json.dumps(schema, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return schema


if __name__ == "__main__":
    export_json_schema("data/benchmark/annotation_schema.json")
