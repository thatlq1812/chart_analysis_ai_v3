"""
Geometric Mapper Module

Maps pixel coordinates to data values using calibration from OCR axis labels.

Key features:
- Linear and logarithmic scale detection
- RANSAC robust fitting for outlier rejection (OCR errors)
- Least squares fitting with confidence scoring
- Pixel-to-value and value-to-pixel conversion
- Sub-pixel accuracy support
- Hough-based axis line detection for improved calibration

Reference: docs/instruction_p2_research.md - Section 5.2
Enhancement: instruction_003.md - RANSAC for robust calibration
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ...schemas.extraction import PointFloat, Polyline, ScaleMapping
from ...schemas.stage_outputs import OCRText

logger = logging.getLogger(__name__)


class FittingMethod(str, Enum):
    """Method for fitting calibration model."""
    LEAST_SQUARES = "least_squares"
    RANSAC = "ransac"
    THEIL_SEN = "theil_sen"  # Robust median-based estimator


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
    
    # Robust fitting (RANSAC)
    fitting_method: FittingMethod = Field(
        default=FittingMethod.RANSAC,
        description="Fitting method: 'least_squares', 'ransac', or 'theil_sen'"
    )
    ransac_min_samples: int = Field(
        default=2,
        ge=2,
        description="Minimum samples for RANSAC model fitting"
    )
    ransac_residual_threshold: float = Field(
        default=10.0,
        gt=0,
        description="Maximum residual (pixels) for inlier classification"
    )
    ransac_max_trials: int = Field(
        default=100,
        ge=10,
        description="Maximum RANSAC iterations"
    )
    ransac_stop_probability: float = Field(
        default=0.99,
        gt=0,
        lt=1,
        description="Probability threshold for early stopping"
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
    
    # Axis line detection
    detect_axis_lines: bool = Field(
        default=True,
        description="Use Hough transform to detect axis lines"
    )
    hough_threshold: int = Field(
        default=50,
        ge=10,
        description="Minimum votes for Hough line detection"
    )
    hough_min_line_length: int = Field(
        default=50,
        ge=10,
        description="Minimum line length for HoughLinesP"
    )
    hough_max_line_gap: int = Field(
        default=10,
        ge=1,
        description="Maximum gap between line segments"
    )
    axis_angle_tolerance: float = Field(
        default=5.0,
        ge=0,
        le=45,
        description="Maximum angle (degrees) deviation from horizontal/vertical"
    )


@dataclass
class AxisLineResult:
    """Result of axis line detection."""
    
    x_axis_line: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    y_axis_line: Optional[Tuple[int, int, int, int]] = None
    origin: Optional[Tuple[int, int]] = None  # Intersection point
    x_axis_y_position: Optional[int] = None  # Y pixel of X-axis
    y_axis_x_position: Optional[int] = None  # X pixel of Y-axis
    all_horizontal_lines: List[Tuple[int, int, int, int]] = field(default_factory=list)
    all_vertical_lines: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Result of axis calibration."""
    
    scale: ScaleMapping
    r_squared: float
    residual_std: float
    calibration_points: List[Tuple[float, float]]  # (pixel, value)
    inlier_mask: Optional[np.ndarray] = None  # RANSAC inlier mask
    outliers_removed: int = 0  # Number of outliers filtered
    confidence: float = 1.0  # Calibration confidence [0-1]


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
        
        # Detected axis lines
        self.axis_lines: Optional[AxisLineResult] = None
        
        # Plot area boundaries (in pixels)
        self.plot_x_min: float = 0
        self.plot_x_max: float = 0
        self.plot_y_min: float = 0
        self.plot_y_max: float = 0
    
    def detect_axis_lines(
        self,
        binary_image: np.ndarray,
        chart_id: str = "unknown",
    ) -> AxisLineResult:
        """
        Detect axis lines using Hough transform.
        
        This helps identify:
        1. The exact pixel positions of X and Y axes
        2. The plot area boundaries
        3. The origin point (if axes meet)
        
        Args:
            binary_image: Binary image (edges or preprocessed)
            chart_id: Chart identifier for logging
        
        Returns:
            AxisLineResult with detected axis information
        """
        if not self.config.detect_axis_lines:
            return AxisLineResult()
        
        h, w = binary_image.shape[:2]
        
        # Edge detection if input is not already edges
        if binary_image.max() > 1:
            edges = cv2.Canny(binary_image, 50, 150)
        else:
            edges = binary_image
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )
        
        if lines is None:
            self.logger.debug(f"No lines detected | chart_id={chart_id}")
            return AxisLineResult()
        
        horizontal_lines = []
        vertical_lines = []
        angle_tol = self.config.axis_angle_tolerance
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Compute angle
            if abs(x2 - x1) < 1e-10:
                angle = 90.0
            else:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Classify as horizontal or vertical
            if angle < angle_tol or angle > (180 - angle_tol):
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(angle - 90) < angle_tol:
                vertical_lines.append((x1, y1, x2, y2))
        
        self.logger.debug(
            f"Lines detected | chart_id={chart_id} | "
            f"horizontal={len(horizontal_lines)} | vertical={len(vertical_lines)}"
        )
        
        # Find the most likely X-axis (longest horizontal line in bottom half)
        x_axis_line = None
        x_axis_y = None
        best_x_axis_length = 0
        
        for x1, y1, x2, y2 in horizontal_lines:
            y_pos = (y1 + y2) / 2
            # X-axis should be in bottom 70% of image
            if y_pos > h * 0.3:
                length = abs(x2 - x1)
                if length > best_x_axis_length:
                    best_x_axis_length = length
                    x_axis_line = (x1, y1, x2, y2)
                    x_axis_y = int(y_pos)
        
        # Find the most likely Y-axis (longest vertical line in left half)
        y_axis_line = None
        y_axis_x = None
        best_y_axis_length = 0
        
        for x1, y1, x2, y2 in vertical_lines:
            x_pos = (x1 + x2) / 2
            # Y-axis should be in left 50% of image
            if x_pos < w * 0.5:
                length = abs(y2 - y1)
                if length > best_y_axis_length:
                    best_y_axis_length = length
                    y_axis_line = (x1, y1, x2, y2)
                    y_axis_x = int(x_pos)
        
        # Find origin (intersection of axes)
        origin = None
        if x_axis_y is not None and y_axis_x is not None:
            origin = (y_axis_x, x_axis_y)
        
        result = AxisLineResult(
            x_axis_line=x_axis_line,
            y_axis_line=y_axis_line,
            origin=origin,
            x_axis_y_position=x_axis_y,
            y_axis_x_position=y_axis_x,
            all_horizontal_lines=horizontal_lines,
            all_vertical_lines=vertical_lines,
        )
        
        self.axis_lines = result
        
        # Update plot boundaries based on detected axes
        if x_axis_line is not None:
            self.plot_y_max = x_axis_y
        if y_axis_line is not None:
            self.plot_x_min = y_axis_x
        
        self.logger.info(
            f"Axis lines detected | chart_id={chart_id} | "
            f"x_axis_y={x_axis_y} | y_axis_x={y_axis_x} | origin={origin}"
        )
        
        return result
    
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
        
        # Try linear fit using configured method
        linear_result = self._fit_with_method(pixels, values, is_log=False)
        
        # Try log fit if configured
        log_result = None
        if self.config.auto_detect_scale and np.all(values > 0):
            log_result = self._fit_with_method(pixels, values, is_log=True)
        
        # Choose better fit
        if log_result and linear_result:
            if log_result.r_squared > linear_result.r_squared + self.config.log_detection_threshold:
                self.y_scale = log_result.scale
                self.logger.info(
                    f"Y-axis: logarithmic scale detected | "
                    f"R2={log_result.r_squared:.4f} | "
                    f"outliers={log_result.outliers_removed}"
                )
                return log_result
        
        if linear_result:
            self.y_scale = linear_result.scale
            self.logger.info(
                f"Y-axis: linear scale | R2={linear_result.r_squared:.4f} | "
                f"outliers={linear_result.outliers_removed}"
            )
            return linear_result
        
        return None
    
    def detect_scale_pattern(
        self,
        values: List[float],
    ) -> Dict[str, Any]:
        """
        Detect special scale patterns from tick values.
        
        Detects:
        - Logarithmic (1, 10, 100, 1000)
        - Percentage (0, 25, 50, 75, 100)
        - Scientific notation (1e-3, 1e-2, 1e-1)
        - Date/time patterns (could extend)
        
        Args:
            values: List of numeric tick values
        
        Returns:
            Dict with pattern info: {
                'type': 'linear' | 'logarithmic' | 'percentage' | 'scientific',
                'base': log base if logarithmic,
                'confidence': pattern confidence,
                'normalized': whether values are 0-1 or 0-100
            }
        """
        if len(values) < 2:
            return {'type': 'linear', 'confidence': 0.0}
        
        sorted_vals = sorted(values)
        
        # Check for percentage pattern (0, 25, 50, 75, 100 or 0, 20, 40, 60, 80, 100)
        if self._is_percentage_scale(sorted_vals):
            return {
                'type': 'percentage',
                'confidence': 0.9,
                'normalized': max(sorted_vals) <= 1.01,  # 0-1 vs 0-100
            }
        
        # Check for logarithmic pattern (powers of 10, 2, or e)
        log_pattern = self._detect_log_pattern(sorted_vals)
        if log_pattern['is_log']:
            return {
                'type': 'logarithmic',
                'base': log_pattern['base'],
                'confidence': log_pattern['confidence'],
            }
        
        # Check for scientific notation pattern
        if self._is_scientific_scale(sorted_vals):
            return {
                'type': 'scientific',
                'confidence': 0.8,
            }
        
        # Default to linear
        return {'type': 'linear', 'confidence': 0.7}
    
    def _is_percentage_scale(self, values: List[float]) -> bool:
        """Check if values represent a percentage scale."""
        if len(values) < 3:
            return False
        
        # Check 0-100 pattern
        if abs(min(values)) < 0.1 and abs(max(values) - 100) < 1:
            # Check for common percentage intervals (20, 25, 10)
            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            if len(diffs) >= 2:
                avg_diff = sum(diffs) / len(diffs)
                if avg_diff in [10, 20, 25] or abs(avg_diff - 20) < 2 or abs(avg_diff - 25) < 2:
                    return True
        
        # Check 0-1 pattern (normalized)
        if abs(min(values)) < 0.01 and abs(max(values) - 1.0) < 0.01:
            return True
        
        return False
    
    def _detect_log_pattern(self, values: List[float]) -> Dict[str, Any]:
        """Detect logarithmic scale pattern."""
        result = {'is_log': False, 'base': 10, 'confidence': 0.0}
        
        if len(values) < 3:
            return result
        
        # Filter positive values
        pos_values = [v for v in values if v > 0]
        if len(pos_values) < 3:
            return result
        
        # Check for base-10 pattern (1, 10, 100, 1000...)
        log10_vals = [np.log10(v) for v in pos_values]
        log10_rounded = [round(v) for v in log10_vals]
        
        if all(abs(v - r) < 0.1 for v, r in zip(log10_vals, log10_rounded)):
            # Values are close to powers of 10
            diffs = [log10_rounded[i+1] - log10_rounded[i] for i in range(len(log10_rounded)-1)]
            if len(set(diffs)) == 1 and diffs[0] in [1, 2]:
                result['is_log'] = True
                result['base'] = 10
                result['confidence'] = 0.95
                return result
        
        # Check for base-2 pattern (1, 2, 4, 8, 16...)
        log2_vals = [np.log2(v) for v in pos_values if v > 0]
        log2_rounded = [round(v) for v in log2_vals]
        
        if all(abs(v - r) < 0.1 for v, r in zip(log2_vals, log2_rounded)):
            diffs = [log2_rounded[i+1] - log2_rounded[i] for i in range(len(log2_rounded)-1)]
            if len(set(diffs)) == 1 and diffs[0] in [1, 2]:
                result['is_log'] = True
                result['base'] = 2
                result['confidence'] = 0.9
                return result
        
        # Check for general exponential growth (ratio between consecutive values)
        if len(pos_values) >= 3:
            ratios = [pos_values[i+1] / pos_values[i] for i in range(len(pos_values)-1)]
            avg_ratio = sum(ratios) / len(ratios)
            ratio_variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)
            
            if ratio_variance < 0.1 * avg_ratio ** 2 and avg_ratio > 1.5:
                result['is_log'] = True
                result['base'] = avg_ratio
                result['confidence'] = 0.7
        
        return result
    
    def _is_scientific_scale(self, values: List[float]) -> bool:
        """Check if values follow scientific notation pattern."""
        if len(values) < 3:
            return False
        
        # Check if values span multiple orders of magnitude with consistent pattern
        pos_values = [abs(v) for v in values if v != 0]
        if len(pos_values) < 3:
            return False
        
        log_vals = [np.log10(v) for v in pos_values]
        span = max(log_vals) - min(log_vals)
        
        # Scientific notation typically spans 3+ orders of magnitude
        return span >= 3
    
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
        
        # Try linear fit using configured method
        linear_result = self._fit_with_method(pixels, values, is_log=False)
        
        # Try log fit if configured
        log_result = None
        if self.config.auto_detect_scale and np.all(values > 0):
            log_result = self._fit_with_method(pixels, values, is_log=True)
        
        # Choose better fit
        if log_result and linear_result:
            if log_result.r_squared > linear_result.r_squared + self.config.log_detection_threshold:
                self.x_scale = log_result.scale
                self.logger.info(
                    f"X-axis: logarithmic scale detected | "
                    f"R2={log_result.r_squared:.4f} | "
                    f"outliers={log_result.outliers_removed}"
                )
                return log_result
        
        if linear_result:
            self.x_scale = linear_result.scale
            self.logger.info(
                f"X-axis: linear scale | R2={linear_result.r_squared:.4f} | "
                f"outliers={linear_result.outliers_removed}"
            )
            return linear_result
        
        return None
    
    def _fit_with_method(
        self,
        pixels: np.ndarray,
        values: np.ndarray,
        is_log: bool = False,
    ) -> Optional[CalibrationResult]:
        """
        Fit calibration model using configured method.
        
        Args:
            pixels: Pixel coordinates array
            values: Data values array
            is_log: Whether to fit logarithmic scale
        
        Returns:
            CalibrationResult or None if fitting fails
        """
        # Transform values for log scale
        if is_log:
            if np.any(values <= 0):
                return None
            fit_values = np.log(values) / np.log(self.config.log_base)
        else:
            fit_values = values
        
        # Select fitting method
        method = self.config.fitting_method
        
        if method == FittingMethod.RANSAC:
            result = self._fit_ransac(pixels, fit_values)
        elif method == FittingMethod.THEIL_SEN:
            result = self._fit_theil_sen(pixels, fit_values)
        else:
            result = self._fit_least_squares(pixels, fit_values)
        
        if result is None:
            return result
        
        # Update scale for logarithmic
        if is_log:
            result.scale.is_logarithmic = True
            result.scale.log_base = self.config.log_base
        
        return result
    
    def _fit_ransac(
        self,
        pixels: np.ndarray,
        values: np.ndarray,
    ) -> Optional[CalibrationResult]:
        """
        Fit linear mapping using RANSAC for robust outlier rejection.
        
        RANSAC (RANdom SAmple Consensus) iteratively:
        1. Selects random subset of points
        2. Fits model to subset
        3. Counts inliers within threshold
        4. Keeps best model
        
        This is robust against OCR errors that produce outlier values.
        """
        n = len(pixels)
        if n < self.config.ransac_min_samples:
            return self._fit_least_squares(pixels, values)
        
        best_inliers = None
        best_num_inliers = 0
        best_slope = None
        best_intercept = None
        
        threshold = self.config.ransac_residual_threshold
        min_samples = self.config.ransac_min_samples
        max_trials = self.config.ransac_max_trials
        stop_prob = self.config.ransac_stop_probability
        
        for trial in range(max_trials):
            # Random sample
            idx = np.random.choice(n, min_samples, replace=False)
            sample_pixels = pixels[idx]
            sample_values = values[idx]
            
            # Fit model to sample
            if len(np.unique(sample_pixels)) < 2:
                continue
            
            # Simple 2-point line fit
            if min_samples == 2:
                dx = sample_pixels[1] - sample_pixels[0]
                if abs(dx) < 1e-10:
                    continue
                slope = (sample_values[1] - sample_values[0]) / dx
                intercept = sample_values[0] - slope * sample_pixels[0]
            else:
                # Least squares on sample
                A = np.vstack([sample_pixels, np.ones(len(sample_pixels))]).T
                try:
                    slope, intercept = np.linalg.lstsq(A, sample_values, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
            
            # Count inliers
            predicted = slope * pixels + intercept
            residuals = np.abs(values - predicted)
            inliers = residuals < threshold
            num_inliers = np.sum(inliers)
            
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_slope = slope
                best_intercept = intercept
                
                # Early stopping if enough inliers found
                inlier_ratio = num_inliers / n
                if inlier_ratio > stop_prob:
                    break
        
        if best_inliers is None or best_num_inliers < self.config.min_calibration_points:
            # Fallback to least squares
            self.logger.debug("RANSAC failed to find enough inliers, using least squares")
            return self._fit_least_squares(pixels, values)
        
        # Refit using all inliers for better accuracy
        inlier_pixels = pixels[best_inliers]
        inlier_values = values[best_inliers]
        
        A = np.vstack([inlier_pixels, np.ones(len(inlier_pixels))]).T
        try:
            slope, intercept = np.linalg.lstsq(A, inlier_values, rcond=None)[0]
        except np.linalg.LinAlgError:
            slope, intercept = best_slope, best_intercept
        
        # Compute R-squared on inliers
        predicted = slope * inlier_pixels + intercept
        ss_res = np.sum((inlier_values - predicted) ** 2)
        ss_tot = np.sum((inlier_values - np.mean(inlier_values)) ** 2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        residual_std = np.std(inlier_values - predicted)
        outliers_removed = int(n - best_num_inliers)  # cast: np.int64 -> int
        
        # Confidence based on inlier ratio and R-squared
        inlier_ratio = best_num_inliers / n
        confidence = inlier_ratio * r_squared
        
        if outliers_removed > 0:
            self.logger.info(
                f"RANSAC removed {outliers_removed} outliers | "
                f"inliers={best_num_inliers}/{n} | R2={r_squared:.4f}"
            )
        
        scale = ScaleMapping(
            slope=slope,
            intercept=intercept,
            is_logarithmic=False,
            num_calibration_points=best_num_inliers,
            fit_error=residual_std,
        )
        
        return CalibrationResult(
            scale=scale,
            r_squared=r_squared,
            residual_std=residual_std,
            calibration_points=list(zip(inlier_pixels.tolist(), inlier_values.tolist())),
            inlier_mask=best_inliers,
            outliers_removed=outliers_removed,
            confidence=confidence,
        )
    
    def _fit_theil_sen(
        self,
        pixels: np.ndarray,
        values: np.ndarray,
    ) -> Optional[CalibrationResult]:
        """
        Fit linear mapping using Theil-Sen estimator.
        
        Theil-Sen is a robust median-based estimator that:
        1. Computes slopes between all pairs of points
        2. Takes median slope
        3. Takes median intercept
        
        Robust to up to 29.3% outliers.
        """
        n = len(pixels)
        if n < 2:
            return None
        
        # Compute all pairwise slopes
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pixels[j] - pixels[i]
                if abs(dx) > 1e-10:
                    slope = (values[j] - values[i]) / dx
                    slopes.append(slope)
        
        if not slopes:
            return self._fit_least_squares(pixels, values)
        
        # Median slope
        slope = np.median(slopes)
        
        # Median intercept
        intercepts = values - slope * pixels
        intercept = np.median(intercepts)
        
        # R-squared
        predicted = slope * pixels + intercept
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        
        if ss_tot < 1e-10:
            r_squared = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        residual_std = np.std(values - predicted)
        
        # Identify outliers (points with large residuals)
        residuals = np.abs(values - predicted)
        threshold = 2.0 * residual_std
        inliers = residuals < threshold
        outliers_removed = int(n - np.sum(inliers))  # cast: np.int64 -> int
        
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
            inlier_mask=inliers,
            outliers_removed=outliers_removed,
            confidence=r_squared,
        )
    
    def _fit_least_squares(
        self,
        pixels: np.ndarray,
        values: np.ndarray,
    ) -> Optional[CalibrationResult]:
        """
        Fit linear mapping using ordinary least squares.
        
        value = slope * pixel + intercept
        """
        n = len(pixels)
        if n < 2:
            return None
        
        # Least squares: y = ax + b using normal equations
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
            inlier_mask=np.ones(n, dtype=bool),
            outliers_removed=0,
            confidence=r_squared,
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
