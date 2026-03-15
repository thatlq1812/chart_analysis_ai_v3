"""
Geometric Value Mapper

Converts pixel coordinates from Stage 3 extraction to actual data values
using axis calibration information.

Key features:
- Linear scale mapping (y = ax + b → value = (pixel - b) / a)
- Logarithmic scale support
- Confidence scoring based on calibration quality
- Handles inverted Y-axis (image coordinate system)
- Batch mapping for efficiency

Reference: docs/architecture/STAGE4_REASONING.md
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from ...schemas.common import Color, Point
from ...schemas.enums import ChartType
from ...schemas.stage_outputs import (
    AxisInfo,
    ChartElement,
    DataPoint,
    DataSeries,
    OCRText,
    RawMetadata,
)

logger = logging.getLogger(__name__)


class ScaleType(str, Enum):
    """Axis scale type."""
    LINEAR = "linear"
    LOG = "log"
    LOG10 = "log10"
    LOG2 = "log2"
    UNKNOWN = "unknown"


class ValueMapperConfig(BaseModel):
    """Configuration for value mapping."""
    
    # Scale detection
    auto_detect_log_scale: bool = Field(
        default=True,
        description="Automatically detect logarithmic scale"
    )
    default_scale_type: ScaleType = Field(
        default=ScaleType.LINEAR,
        description="Default scale type when auto-detection fails"
    )
    
    # Coordinate system
    y_inverted: bool = Field(
        default=True,
        description="Y-axis inverted (image coordinates: top=0)"
    )
    
    # Calibration thresholds
    min_confidence_for_mapping: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Minimum calibration confidence to apply mapping"
    )
    
    # Value validation
    clamp_to_axis_range: bool = Field(
        default=True,
        description="Clamp mapped values to detected axis range"
    )
    extrapolation_limit: float = Field(
        default=0.1,
        ge=0,
        description="Maximum extrapolation beyond axis range (fraction)"
    )
    
    # Precision
    round_integers: bool = Field(
        default=True,
        description="Round values to integers if axis labels are integers"
    )
    decimal_places: int = Field(
        default=2,
        ge=0,
        le=6,
        description="Decimal places for non-integer values"
    )


@dataclass
class MappingResult:
    """Result of a single value mapping."""
    
    pixel_value: float  # Original pixel coordinate
    mapped_value: float  # Converted data value
    confidence: float  # Mapping confidence [0-1]
    extrapolated: bool = False  # Whether value was extrapolated
    clamped: bool = False  # Whether value was clamped
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pixel": self.pixel_value,
            "value": self.mapped_value,
            "confidence": self.confidence,
            "extrapolated": self.extrapolated,
            "clamped": self.clamped,
        }


@dataclass
class AxisMapping:
    """Axis-specific mapping parameters."""
    
    # Linear: value = slope * pixel + intercept
    # For inverted Y: value = slope * (max_pixel - pixel) + intercept
    slope: float  # pixels per unit value
    intercept: float  # value at pixel=0 (or pixel=max for inverted)
    
    # Detected range
    pixel_min: float
    pixel_max: float
    value_min: float
    value_max: float
    
    # Scale info
    scale_type: ScaleType = ScaleType.LINEAR
    log_base: float = 10.0
    
    # Quality metrics
    r_squared: float = 0.0
    confidence: float = 0.0
    is_integer_scale: bool = False
    
    def pixel_to_value(self, pixel: float, inverted: bool = True) -> float:
        """Convert pixel coordinate to value."""
        if self.scale_type == ScaleType.LINEAR:
            if inverted:
                # Inverted Y: higher pixel = lower value
                normalized = (self.pixel_max - pixel) / (self.pixel_max - self.pixel_min)
            else:
                normalized = (pixel - self.pixel_min) / (self.pixel_max - self.pixel_min)
            
            value = self.value_min + normalized * (self.value_max - self.value_min)
            return value
        
        elif self.scale_type in (ScaleType.LOG, ScaleType.LOG10, ScaleType.LOG2):
            # Log scale
            if inverted:
                normalized = (self.pixel_max - pixel) / (self.pixel_max - self.pixel_min)
            else:
                normalized = (pixel - self.pixel_min) / (self.pixel_max - self.pixel_min)
            
            # Log interpolation
            log_min = math.log10(max(self.value_min, 1e-10))
            log_max = math.log10(max(self.value_max, 1e-10))
            log_value = log_min + normalized * (log_max - log_min)
            return math.pow(10, log_value)
        
        return 0.0
    
    def value_to_pixel(self, value: float, inverted: bool = True) -> float:
        """Convert value to pixel coordinate (inverse mapping)."""
        if self.scale_type == ScaleType.LINEAR:
            if abs(self.value_max - self.value_min) < 1e-10:
                return (self.pixel_min + self.pixel_max) / 2
            
            normalized = (value - self.value_min) / (self.value_max - self.value_min)
            
            if inverted:
                pixel = self.pixel_max - normalized * (self.pixel_max - self.pixel_min)
            else:
                pixel = self.pixel_min + normalized * (self.pixel_max - self.pixel_min)
            
            return pixel
        
        # Log scale
        log_min = math.log10(max(self.value_min, 1e-10))
        log_max = math.log10(max(self.value_max, 1e-10))
        log_value = math.log10(max(value, 1e-10))
        
        if abs(log_max - log_min) < 1e-10:
            return (self.pixel_min + self.pixel_max) / 2
        
        normalized = (log_value - log_min) / (log_max - log_min)
        
        if inverted:
            return self.pixel_max - normalized * (self.pixel_max - self.pixel_min)
        else:
            return self.pixel_min + normalized * (self.pixel_max - self.pixel_min)


class GeometricValueMapper:
    """
    Maps pixel coordinates to actual data values.
    
    Uses calibration information from Stage 3 (AxisInfo) to build
    mapping functions for X and Y axes.
    
    Example:
        mapper = GeometricValueMapper()
        mapper.calibrate_from_axis_info(axis_info)
        
        # Map single point
        value = mapper.pixel_to_value_y(250)
        
        # Map all elements in metadata
        series = mapper.map_metadata_to_series(metadata)
    """
    
    def __init__(self, config: Optional[ValueMapperConfig] = None):
        """
        Initialize mapper.
        
        Args:
            config: Mapping configuration
        """
        self.config = config or ValueMapperConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.x_mapping: Optional[AxisMapping] = None
        self.y_mapping: Optional[AxisMapping] = None
        
        self._is_calibrated = False
    
    @property
    def is_calibrated(self) -> bool:
        """Check if mapper has been calibrated."""
        return self._is_calibrated
    
    def calibrate_from_axis_info(
        self,
        axis_info: AxisInfo,
        image_width: int = 0,
        image_height: int = 0,
    ) -> bool:
        """
        Calibrate mapper using AxisInfo from Stage 3.
        
        Args:
            axis_info: Axis information from Stage 3
            image_width: Image width for pixel range estimation
            image_height: Image height for pixel range estimation
        
        Returns:
            True if calibration successful
        """
        self.logger.debug("Calibrating from AxisInfo")
        
        success = False
        
        # X-axis calibration
        if axis_info.x_axis_detected and axis_info.x_min is not None:
            try:
                # Estimate pixel range from image dimensions or use defaults
                x_pixel_min = 50  # Typical left margin
                x_pixel_max = image_width - 50 if image_width > 100 else 400
                
                # Use scale factor if available
                if axis_info.x_scale_factor and axis_info.x_scale_factor > 0:
                    value_range = (axis_info.x_max or 0) - (axis_info.x_min or 0)
                    x_pixel_max = x_pixel_min + value_range * axis_info.x_scale_factor
                
                self.x_mapping = AxisMapping(
                    slope=1.0,
                    intercept=0.0,
                    pixel_min=x_pixel_min,
                    pixel_max=x_pixel_max,
                    value_min=axis_info.x_min,
                    value_max=axis_info.x_max or axis_info.x_min,
                    confidence=axis_info.x_calibration_confidence,
                    r_squared=axis_info.x_calibration_confidence,
                    is_integer_scale=self._check_integer_scale(
                        axis_info.x_min, axis_info.x_max
                    ),
                )
                success = True
                self.logger.debug(
                    f"X-axis calibrated | range=[{axis_info.x_min}, {axis_info.x_max}] | "
                    f"confidence={axis_info.x_calibration_confidence:.2f}"
                )
            except Exception as e:
                self.logger.warning(f"X-axis calibration failed: {e}")
        
        # Y-axis calibration
        if axis_info.y_axis_detected and axis_info.y_min is not None:
            try:
                # Estimate pixel range (Y is typically inverted in images)
                y_pixel_min = 50  # Top margin
                y_pixel_max = image_height - 50 if image_height > 100 else 300
                
                if axis_info.y_scale_factor and axis_info.y_scale_factor > 0:
                    value_range = (axis_info.y_max or 0) - (axis_info.y_min or 0)
                    y_pixel_max = y_pixel_min + value_range * axis_info.y_scale_factor
                
                self.y_mapping = AxisMapping(
                    slope=1.0,
                    intercept=0.0,
                    pixel_min=y_pixel_min,
                    pixel_max=y_pixel_max,
                    value_min=axis_info.y_min,
                    value_max=axis_info.y_max or axis_info.y_min,
                    confidence=axis_info.y_calibration_confidence,
                    r_squared=axis_info.y_calibration_confidence,
                    is_integer_scale=self._check_integer_scale(
                        axis_info.y_min, axis_info.y_max
                    ),
                )
                success = True
                self.logger.debug(
                    f"Y-axis calibrated | range=[{axis_info.y_min}, {axis_info.y_max}] | "
                    f"confidence={axis_info.y_calibration_confidence:.2f}"
                )
            except Exception as e:
                self.logger.warning(f"Y-axis calibration failed: {e}")
        
        self._is_calibrated = success
        return success
    
    def calibrate_from_tick_labels(
        self,
        ocr_texts: List[OCRText],
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        Calibrate from OCR tick labels.
        
        Identifies tick labels based on position (left edge for Y, bottom for X)
        and extracts numeric values for calibration.
        
        Args:
            ocr_texts: OCR text results from Stage 3
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            True if calibration successful
        """
        self.logger.debug("Calibrating from OCR tick labels")
        
        y_ticks: List[Tuple[float, float]] = []  # (pixel_y, value)
        x_ticks: List[Tuple[float, float]] = []  # (pixel_x, value)
        
        for text in ocr_texts:
            # Parse numeric value
            value = self._parse_numeric(text.text)
            if value is None:
                continue
            
            # Classify as X or Y tick based on position
            center_x = text.bbox.center[0]
            center_y = text.bbox.center[1]
            
            # Y-axis ticks: typically on left side
            if center_x < image_width * 0.2:
                y_ticks.append((center_y, value))
            
            # X-axis ticks: typically on bottom
            elif center_y > image_height * 0.8:
                x_ticks.append((center_x, value))
        
        success = False
        
        # Calibrate Y-axis
        if len(y_ticks) >= 2:
            y_ticks.sort(key=lambda t: t[0])  # Sort by pixel position
            
            pixels = [t[0] for t in y_ticks]
            values = [t[1] for t in y_ticks]
            
            self.y_mapping = AxisMapping(
                slope=1.0,
                intercept=0.0,
                pixel_min=min(pixels),
                pixel_max=max(pixels),
                value_min=min(values),
                value_max=max(values),
                confidence=0.7,
                is_integer_scale=self._check_integer_scale(min(values), max(values)),
            )
            success = True
            self.logger.debug(f"Y-axis calibrated from {len(y_ticks)} tick labels")
        
        # Calibrate X-axis
        if len(x_ticks) >= 2:
            x_ticks.sort(key=lambda t: t[0])
            
            pixels = [t[0] for t in x_ticks]
            values = [t[1] for t in x_ticks]
            
            self.x_mapping = AxisMapping(
                slope=1.0,
                intercept=0.0,
                pixel_min=min(pixels),
                pixel_max=max(pixels),
                value_min=min(values),
                value_max=max(values),
                confidence=0.7,
                is_integer_scale=self._check_integer_scale(min(values), max(values)),
            )
            success = True
            self.logger.debug(f"X-axis calibrated from {len(x_ticks)} tick labels")
        
        self._is_calibrated = success
        return success
    
    def pixel_to_value_x(self, pixel_x: float) -> MappingResult:
        """
        Map X pixel coordinate to value.
        
        Args:
            pixel_x: X coordinate in pixels
        
        Returns:
            MappingResult with mapped value
        """
        if not self.x_mapping:
            return MappingResult(
                pixel_value=pixel_x,
                mapped_value=pixel_x,
                confidence=0.0,
            )
        
        return self._map_pixel_to_value(
            pixel_x,
            self.x_mapping,
            inverted=False,  # X typically not inverted
        )
    
    def pixel_to_value_y(self, pixel_y: float) -> MappingResult:
        """
        Map Y pixel coordinate to value.
        
        Args:
            pixel_y: Y coordinate in pixels
        
        Returns:
            MappingResult with mapped value
        """
        if not self.y_mapping:
            return MappingResult(
                pixel_value=pixel_y,
                mapped_value=pixel_y,
                confidence=0.0,
            )
        
        return self._map_pixel_to_value(
            pixel_y,
            self.y_mapping,
            inverted=self.config.y_inverted,
        )
    
    def map_point(self, point: Point) -> Tuple[MappingResult, MappingResult]:
        """
        Map a Point to (x_value, y_value).
        
        Args:
            point: Point with pixel coordinates
        
        Returns:
            Tuple of (x_result, y_result)
        """
        x_result = self.pixel_to_value_x(float(point.x))
        y_result = self.pixel_to_value_y(float(point.y))
        return (x_result, y_result)
    
    def map_elements_to_series(
        self,
        elements: List[ChartElement],
        chart_type: ChartType,
        x_labels: Optional[List[str]] = None,
    ) -> List[DataSeries]:
        """
        Map chart elements to data series.
        
        Groups elements by color and maps pixel positions to values.
        
        Args:
            elements: Chart elements from Stage 3
            chart_type: Chart type for appropriate mapping
            x_labels: Optional X-axis labels for categorical data
        
        Returns:
            List of DataSeries with mapped values
        """
        if not elements:
            return []
        
        # Group by color
        color_groups: Dict[Tuple[int, int, int], List[ChartElement]] = {}
        
        for elem in elements:
            if elem.color:
                key = (elem.color.r, elem.color.g, elem.color.b)
            else:
                key = (128, 128, 128)  # Default gray
            
            if key not in color_groups:
                color_groups[key] = []
            color_groups[key].append(elem)
        
        series_list = []
        
        for idx, (color_key, group_elements) in enumerate(color_groups.items()):
            # Sort elements by X position
            sorted_elements = sorted(group_elements, key=lambda e: e.center.x)
            
            points = []
            for i, elem in enumerate(sorted_elements):
                # Map coordinates
                x_result, y_result = self.map_point(elem.center)
                
                # Determine label
                if x_labels and i < len(x_labels):
                    label = x_labels[i]
                elif chart_type in (ChartType.BAR, ChartType.HISTOGRAM):
                    # For bar charts, use index or mapped X
                    label = f"Bar {i + 1}"
                else:
                    label = f"{x_result.mapped_value:.1f}"
                
                # Use Y value (most charts encode value in Y)
                value = y_result.mapped_value
                
                # For horizontal bar charts, use X value
                if chart_type == ChartType.BAR and self._is_horizontal_bar(sorted_elements):
                    value = x_result.mapped_value
                
                # Round if appropriate
                if self.config.round_integers and self.y_mapping:
                    if self.y_mapping.is_integer_scale:
                        value = round(value)
                    else:
                        value = round(value, self.config.decimal_places)
                
                # Confidence combines calibration and mapping quality
                confidence = min(x_result.confidence, y_result.confidence)
                if x_result.extrapolated or y_result.extrapolated:
                    confidence *= 0.8
                
                points.append(DataPoint(
                    label=label,
                    value=value,
                    confidence=confidence,
                ))
            
            series_list.append(DataSeries(
                name=f"Series {idx + 1}",
                color=Color(r=color_key[0], g=color_key[1], b=color_key[2]),
                points=points,
            ))
        
        return series_list
    
    def map_metadata_to_series(
        self,
        metadata: RawMetadata,
        image_width: int = 0,
        image_height: int = 0,
    ) -> List[DataSeries]:
        """
        Complete mapping from RawMetadata to DataSeries.
        
        Calibrates from axis_info and maps all elements.
        
        Args:
            metadata: Raw metadata from Stage 3
            image_width: Image width (optional, for better calibration)
            image_height: Image height (optional)
        
        Returns:
            List of DataSeries with mapped values
        """
        # VLM table — direct structured extraction, no geometric calibration needed.
        # Stage 3 always runs with a VLM extractor backend (deplot / matcha / pix2struct / svlm).
        # When the model produced a valid table (extraction_confidence > 0), use it directly.
        if (
            metadata.pix2struct_table is not None
            and metadata.pix2struct_table.extraction_confidence > 0
        ):
            series = self._pix2struct_to_series(metadata.pix2struct_table)
            if series:
                self.logger.debug(
                    f"map_metadata_to_series: VLM table ok | "
                    f"chart_id={metadata.chart_id} | series={len(series)}"
                )
                return series

        # VLM returned empty or failed — no geometry fallback available.
        self.logger.warning(
            f"map_metadata_to_series: VLM extraction empty or unavailable | "
            f"chart_id={metadata.chart_id} | returning empty series"
        )
        return []

    def _pix2struct_to_series(
        self,
        table: Any,
    ) -> List[DataSeries]:
        """
        Convert a Pix2StructResult table to DataSeries.

        Convention used by Pix2Struct derendering output:
            headers[0]    = x-axis / category label column
            headers[1..N] = series names (one series per remaining column)

        Example table:
            headers = ["Year", "Model A", "Model B"]
            rows    = [["2020", "0.82", "0.79"],
                       ["2021", "0.85", "0.81"]]
        Produces:
            Series "Model A": [("2020", 0.82), ("2021", 0.85)]
            Series "Model B": [("2020", 0.79), ("2021", 0.81)]

        Args:
            table: Pix2StructResult from Stage 3

        Returns:
            List of DataSeries with confidence=0.95 (direct from model)
        """
        headers: List[str] = table.headers
        rows: List[List[str]] = table.rows

        if not headers or not rows:
            return []

        # Single-column table: treat entire column as one series
        if len(headers) == 1:
            series_names = [headers[0]]
            label_col_idx = None
        else:
            label_col_idx = 0
            series_names = headers[1:]

        series_list: List[DataSeries] = []

        for s_idx, series_name in enumerate(series_names):
            col_idx = s_idx if label_col_idx is None else s_idx + 1
            points: List[DataPoint] = []

            for row_idx, row in enumerate(rows):
                if label_col_idx is not None and len(row) > label_col_idx:
                    label = str(row[label_col_idx]).strip()
                else:
                    label = str(row_idx + 1)

                val_str = row[col_idx] if len(row) > col_idx else ""
                value = self._parse_numeric(val_str.strip())
                if value is None:
                    continue

                points.append(DataPoint(label=label, value=value, confidence=0.95))

            if points:
                series_list.append(DataSeries(name=series_name, points=points))

        return series_list

    def _map_pixel_to_value(
        self,
        pixel: float,
        mapping: AxisMapping,
        inverted: bool,
    ) -> MappingResult:
        """Internal pixel to value mapping."""
        extrapolated = False
        clamped = False
        
        # Check extrapolation
        if pixel < mapping.pixel_min:
            extrapolated = True
            if self.config.clamp_to_axis_range:
                limit = mapping.pixel_min - self.config.extrapolation_limit * (
                    mapping.pixel_max - mapping.pixel_min
                )
                if pixel < limit:
                    pixel = limit
                    clamped = True
        
        elif pixel > mapping.pixel_max:
            extrapolated = True
            if self.config.clamp_to_axis_range:
                limit = mapping.pixel_max + self.config.extrapolation_limit * (
                    mapping.pixel_max - mapping.pixel_min
                )
                if pixel > limit:
                    pixel = limit
                    clamped = True
        
        # Map value
        value = mapping.pixel_to_value(pixel, inverted)
        
        # Round if integer scale
        if self.config.round_integers and mapping.is_integer_scale:
            value = round(value)
        else:
            value = round(value, self.config.decimal_places)
        
        # Confidence based on calibration quality and extrapolation
        confidence = mapping.confidence
        if extrapolated:
            confidence *= 0.7
        if clamped:
            confidence *= 0.5
        
        return MappingResult(
            pixel_value=pixel,
            mapped_value=value,
            confidence=confidence,
            extrapolated=extrapolated,
            clamped=clamped,
        )
    
    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse numeric value from text."""
        # Clean text
        cleaned = text.strip()
        
        # Remove common suffixes
        for suffix in ['%', 'k', 'K', 'm', 'M', 'b', 'B']:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-1]
                break
        
        # Handle common OCR errors
        cleaned = cleaned.replace('O', '0').replace('l', '1').replace('I', '1')
        cleaned = cleaned.replace(',', '')  # Remove thousands separator
        
        try:
            value = float(cleaned)
            
            # Apply suffix multiplier
            if text.strip().endswith(('k', 'K')):
                value *= 1000
            elif text.strip().endswith(('m', 'M')):
                value *= 1000000
            elif text.strip().endswith(('b', 'B')):
                value *= 1000000000
            
            return value
        except ValueError:
            return None
    
    def _check_integer_scale(
        self,
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> bool:
        """Check if scale values are integers."""
        if min_val is None or max_val is None:
            return False
        
        return (
            abs(min_val - round(min_val)) < 0.01 and
            abs(max_val - round(max_val)) < 0.01
        )
    
    def _is_horizontal_bar(self, elements: List[ChartElement]) -> bool:
        """Detect if bar chart is horizontal."""
        if len(elements) < 2:
            return False
        
        # Horizontal bars: consistent X variation, stacked Y positions
        y_positions = [e.center.y for e in elements]
        y_std = np.std(y_positions)
        
        x_positions = [e.center.x for e in elements]
        x_std = np.std(x_positions)
        
        # If Y positions are more spread than X, likely horizontal
        return y_std > x_std * 2
    
    def _extract_x_labels(self, texts: List[OCRText]) -> List[str]:
        """Extract X-axis labels from OCR texts."""
        x_labels = []
        
        for text in texts:
            if text.role == "xlabel":
                x_labels.append(text.text)
            elif text.role == "value":
                # Could be axis tick label
                x_labels.append(text.text)
        
        return x_labels
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of current calibration state."""
        return {
            "is_calibrated": self._is_calibrated,
            "x_axis": {
                "calibrated": self.x_mapping is not None,
                "range": [
                    self.x_mapping.value_min if self.x_mapping else None,
                    self.x_mapping.value_max if self.x_mapping else None,
                ],
                "confidence": self.x_mapping.confidence if self.x_mapping else 0,
                "scale_type": self.x_mapping.scale_type.value if self.x_mapping else None,
            } if self.x_mapping else None,
            "y_axis": {
                "calibrated": self.y_mapping is not None,
                "range": [
                    self.y_mapping.value_min if self.y_mapping else None,
                    self.y_mapping.value_max if self.y_mapping else None,
                ],
                "confidence": self.y_mapping.confidence if self.y_mapping else 0,
                "scale_type": self.y_mapping.scale_type.value if self.y_mapping else None,
            } if self.y_mapping else None,
        }
