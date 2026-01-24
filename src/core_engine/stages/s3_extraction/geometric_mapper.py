"""
Geometric Mapper Module

Maps pixel coordinates to data values using calibration from OCR axis labels.

Key features:
- Linear and logarithmic scale detection
- Least squares fitting for robust calibration
- Pixel-to-value and value-to-pixel conversion
- Sub-pixel accuracy support

Reference: docs/instruction_p2_research.md - Section 5.2
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ...schemas.extraction import PointFloat, Polyline, ScaleMapping
from ...schemas.stage_outputs import OCRText

logger = logging.getLogger(__name__)


class MapperConfig(BaseModel):
    """Configuration for geometric mapping."""
    
    # Calibration settings
    min_calibration_points: int = Field(
        default=2,
        ge=2,
        description="Minimum points required for calibration"
    )
    max_fit_error: float = Field(
        default=5.0,
        gt=0,
        description="Maximum allowed fit error (pixels)"
    )
    
    # Scale detection
    auto_detect_scale: bool = Field(
        default=True,
        description="Auto-detect linear vs logarithmic scale"
    )
    log_base: float = Field(
        default=10.0,
        gt=1,
        description="Base for logarithmic scale"
    )
    log_detection_threshold: float = Field(
        default=0.1,
        gt=0,
        description="R-squared improvement threshold for log detection"
    )
    
    # Coordinate system
    y_axis_inverted: bool = Field(
        default=True,
        description="Y-axis increases downward (image coordinates)"
    )


@dataclass
class CalibrationResult:
    """Result of axis calibration."""
    
    scale: ScaleMapping
    r_squared: float
    residual_std: float
    calibration_points: List[Tuple[float, float]]  # (pixel, value)


class GeometricMapper:
    """
    Maps between pixel coordinates and data values.
    
    Uses OCR axis labels to build calibration model,
    then applies to all extracted geometric features.
    
    Example:
        config = MapperConfig(auto_detect_scale=True)
        mapper = GeometricMapper(config)
        mapper.calibrate_y_axis(y_tick_labels)
        values = mapper.pixel_to_value_y(pixel_positions)
    """
    
    def __init__(self, config: Optional[MapperConfig] = None):
        """
        Initialize mapper.
        
        Args:
            config: Mapping configuration (uses defaults if None)
        """
        self.config = config or MapperConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.x_scale: Optional[ScaleMapping] = None
        self.y_scale: Optional[ScaleMapping] = None
        
        # Plot area boundaries (in pixels)
        self.plot_x_min: float = 0
        self.plot_x_max: float = 0
        self.plot_y_min: float = 0
        self.plot_y_max: float = 0
    
    def calibrate_y_axis(
        self,
        tick_values: List[Tuple[float, float]],
    ) -> Optional[CalibrationResult]:
        """
        Calibrate Y-axis mapping from tick labels.
        
        Args:
            tick_values: List of (pixel_y, value) from OCR
        
        Returns:
            CalibrationResult or None if insufficient data
        """
        if len(tick_values) < self.config.min_calibration_points:
            self.logger.warning(
                f"Insufficient calibration points for Y-axis: "
                f"{len(tick_values)} < {self.config.min_calibration_points}"
            )
            return None
        
        # Sort by pixel position
        sorted_ticks = sorted(tick_values, key=lambda x: x[0])
        
        pixels = np.array([t[0] for t in sorted_ticks])
        values = np.array([t[1] for t in sorted_ticks])
        
        # Try linear fit first
        linear_result = self._fit_linear(pixels, values)
        
        # Try log fit if configured
        log_result = None
        if self.config.auto_detect_scale and np.all(values > 0):
            log_result = self._fit_logarithmic(pixels, values)
        
        # Choose better fit
        if log_result and linear_result:
            if log_result.r_squared > linear_result.r_squared + self.config.log_detection_threshold:
                self.y_scale = log_result.scale
                self.logger.info(
                    f"Y-axis: logarithmic scale detected | "
                    f"R2={log_result.r_squared:.4f}"
                )
                return log_result
        
        if linear_result:
            self.y_scale = linear_result.scale
            self.logger.info(
                f"Y-axis: linear scale | R2={linear_result.r_squared:.4f}"
            )
            return linear_result
        
        return None
    
    def calibrate_x_axis(
        self,
        tick_values: List[Tuple[float, float]],
    ) -> Optional[CalibrationResult]:
        """
        Calibrate X-axis mapping from tick labels.
        
        Args:
            tick_values: List of (pixel_x, value) from OCR
        
        Returns:
            CalibrationResult or None if insufficient data
        """
        if len(tick_values) < self.config.min_calibration_points:
            self.logger.warning(
                f"Insufficient calibration points for X-axis: "
                f"{len(tick_values)} < {self.config.min_calibration_points}"
            )
            return None
        
        # Sort by pixel position
        sorted_ticks = sorted(tick_values, key=lambda x: x[0])
        
        pixels = np.array([t[0] for t in sorted_ticks])
        values = np.array([t[1] for t in sorted_ticks])
        
        # Try linear fit first
        linear_result = self._fit_linear(pixels, values)
        
        # Try log fit if configured
        log_result = None
        if self.config.auto_detect_scale and np.all(values > 0):
            log_result = self._fit_logarithmic(pixels, values)
        
        # Choose better fit
        if log_result and linear_result:
            if log_result.r_squared > linear_result.r_squared + self.config.log_detection_threshold:
                self.x_scale = log_result.scale
                self.logger.info(
                    f"X-axis: logarithmic scale detected | "
                    f"R2={log_result.r_squared:.4f}"
                )
                return log_result
        
        if linear_result:
            self.x_scale = linear_result.scale
            self.logger.info(
                f"X-axis: linear scale | R2={linear_result.r_squared:.4f}"
            )
            return linear_result
        
        return None
    
    def _fit_linear(
        self,
        pixels: np.ndarray,
        values: np.ndarray,
    ) -> Optional[CalibrationResult]:
        """
        Fit linear mapping: value = slope * pixel + intercept
        
        Uses least squares regression.
        """
        n = len(pixels)
        if n < 2:
            return None
        
        # Least squares: y = ax + b
        # Normal equations
        sum_x = np.sum(pixels)
        sum_y = np.sum(values)
        sum_xy = np.sum(pixels * values)
        sum_x2 = np.sum(pixels ** 2)
        
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        predicted = slope * pixels + intercept
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        residual_std = np.std(values - predicted)
        
        scale = ScaleMapping(
            slope=slope,
            intercept=intercept,
            is_logarithmic=False,
            num_calibration_points=n,
            fit_error=residual_std,
        )
        
        return CalibrationResult(
            scale=scale,
            r_squared=r_squared,
            residual_std=residual_std,
            calibration_points=list(zip(pixels.tolist(), values.tolist())),
        )
    
    def _fit_logarithmic(
        self,
        pixels: np.ndarray,
        values: np.ndarray,
    ) -> Optional[CalibrationResult]:
        """
        Fit logarithmic mapping: log(value) = slope * pixel + intercept
        """
        if np.any(values <= 0):
            return None
        
        log_values = np.log(values) / np.log(self.config.log_base)
        
        linear_result = self._fit_linear(pixels, log_values)
        
        if linear_result is None:
            return None
        
        # Update scale to be logarithmic
        scale = ScaleMapping(
            slope=linear_result.scale.slope,
            intercept=linear_result.scale.intercept,
            is_logarithmic=True,
            log_base=self.config.log_base,
            num_calibration_points=linear_result.scale.num_calibration_points,
            fit_error=linear_result.scale.fit_error,
        )
        
        return CalibrationResult(
            scale=scale,
            r_squared=linear_result.r_squared,
            residual_std=linear_result.residual_std,
            calibration_points=linear_result.calibration_points,
        )
    
    def pixel_to_value_x(self, pixel_x: float) -> Optional[float]:
        """Convert X pixel coordinate to data value."""
        if self.x_scale is None:
            return None
        return self.x_scale.pixel_to_value(pixel_x)
    
    def pixel_to_value_y(self, pixel_y: float) -> Optional[float]:
        """Convert Y pixel coordinate to data value."""
        if self.y_scale is None:
            return None
        
        # Handle inverted Y axis (image coordinates)
        if self.config.y_axis_inverted:
            # In image coords, higher pixel = lower value
            # The scale was fitted with this in mind
            pass
        
        return self.y_scale.pixel_to_value(pixel_y)
    
    def point_to_values(
        self,
        point: PointFloat,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert pixel point to data values.
        
        Args:
            point: Pixel coordinates
        
        Returns:
            (x_value, y_value) tuple
        """
        x_val = self.pixel_to_value_x(point.x)
        y_val = self.pixel_to_value_y(point.y)
        return (x_val, y_val)
    
    def polyline_to_values(
        self,
        polyline: Polyline,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Convert polyline vertices to data values.
        
        Args:
            polyline: Input polyline in pixel coordinates
        
        Returns:
            List of (x_value, y_value) tuples
        """
        return [self.point_to_values(pt) for pt in polyline.points]
    
    def set_plot_boundaries(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """
        Set plot area boundaries for normalization.
        
        Args:
            x_min: Left edge pixel
            x_max: Right edge pixel
            y_min: Top edge pixel
            y_max: Bottom edge pixel
        """
        self.plot_x_min = x_min
        self.plot_x_max = x_max
        self.plot_y_min = y_min
        self.plot_y_max = y_max
    
    def normalize_point(self, point: PointFloat) -> PointFloat:
        """
        Normalize point to [0, 1] range within plot area.
        
        Args:
            point: Pixel coordinates
        
        Returns:
            Normalized coordinates (0 = left/top, 1 = right/bottom)
        """
        width = self.plot_x_max - self.plot_x_min
        height = self.plot_y_max - self.plot_y_min
        
        if width <= 0 or height <= 0:
            return point
        
        norm_x = (point.x - self.plot_x_min) / width
        norm_y = (point.y - self.plot_y_min) / height
        
        return PointFloat(x=norm_x, y=norm_y)
    
    def estimate_value_from_bar_height(
        self,
        bar_top_y: float,
        bar_bottom_y: float,
        baseline_value: float = 0.0,
    ) -> Optional[float]:
        """
        Estimate bar value from top/bottom pixel positions.
        
        Args:
            bar_top_y: Y pixel of bar top
            bar_bottom_y: Y pixel of bar bottom (baseline)
            baseline_value: Value at baseline (usually 0)
        
        Returns:
            Estimated bar value
        """
        if self.y_scale is None:
            return None
        
        top_value = self.pixel_to_value_y(bar_top_y)
        
        if top_value is None:
            return None
        
        return top_value
