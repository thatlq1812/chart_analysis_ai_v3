"""
Element Detector Module

Detects discrete chart elements: bars, markers, pie slices.

These elements are detected using contour analysis rather than
skeletonization, which is better suited for continuous lines.

Key features:
- Bar rectangle detection using contour approximation
- Marker (circle, square, triangle) detection using Hough transform
- Pie slice detection using angular analysis
- Color extraction for series grouping
- Watershed segmentation for separating merged bars
- Projection analysis for bar detection in complex images

Reference: docs/instruction_p2_research.md - Section 3.2

Version 2.0 - Added advanced bar separation algorithms
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy import ndimage

from ...schemas.common import Color
from ...schemas.extraction import (
    BarRectangle,
    DataMarker,
    MarkerType,
    PieSlice,
    PointFloat,
)


class BarSeparationMethod(str, Enum):
    """Method for separating merged bars."""
    CONTOUR_ONLY = "contour_only"  # Original method
    WATERSHED = "watershed"  # Watershed segmentation
    PROJECTION = "projection"  # Vertical/Horizontal projection
    MORPHOLOGICAL = "morphological"  # Erosion-based separation
    HYBRID = "hybrid"  # Combine multiple methods

logger = logging.getLogger(__name__)


class ElementDetectorConfig(BaseModel):
    """Configuration for element detection."""
    
    # Bar detection
    detect_bars: bool = Field(default=True, description="Detect bar rectangles")
    min_bar_area: int = Field(default=100, ge=1, description="Minimum bar area in pixels")
    max_bar_area_ratio: float = Field(
        default=0.4,
        ge=0.01,
        le=1.0,
        description="Maximum bar area as ratio of image area (filter background)"
    )
    bar_aspect_ratio_min: float = Field(default=0.03, gt=0, description="Min aspect ratio for bars")
    bar_aspect_ratio_max: float = Field(default=30.0, gt=0, description="Max aspect ratio for bars")
    
    # Advanced bar separation
    bar_separation_method: BarSeparationMethod = Field(
        default=BarSeparationMethod.HYBRID,
        description="Method for separating merged bars"
    )
    min_bar_gap: int = Field(default=3, ge=1, description="Minimum gap between bars (pixels)")
    projection_threshold: float = Field(default=0.2, ge=0, le=1, description="Threshold for projection analysis")
    watershed_markers_dist: int = Field(default=10, ge=1, description="Distance transform threshold for watershed")
    
    # Marker detection
    detect_markers: bool = Field(default=True, description="Detect data point markers")
    min_marker_size: int = Field(default=5, ge=1, description="Minimum marker size")
    max_marker_size: int = Field(default=50, ge=1, description="Maximum marker size")
    
    # Pie slice detection
    detect_pie_slices: bool = Field(default=True, description="Detect pie chart slices")
    min_slice_angle: float = Field(default=0.05, gt=0, description="Minimum slice angle (radians)")
    
    # Contour filtering
    contour_approx_epsilon: float = Field(
        default=0.02,
        gt=0,
        description="Contour approximation epsilon (fraction of perimeter)"
    )
    
    # Color extraction
    extract_colors: bool = Field(default=True, description="Extract dominant colors")
    
    # [FIX] Color-based bar detection (better for bar charts)
    use_color_segmentation: bool = Field(
        default=True,
        description="Use color segmentation to detect colored bars (recommended for bar charts)"
    )
    color_saturation_threshold: int = Field(
        default=30,
        ge=0,
        le=255,
        description="Minimum saturation to consider as colored (not grayscale)"
    )


@dataclass
class ElementDetectionResult:
    """Result of element detection."""
    
    bars: List[BarRectangle]
    markers: List[DataMarker]
    slices: List[PieSlice]
    contours_analyzed: int


class ElementDetector:
    """
    Detects discrete chart elements from binary/color images.
    
    Uses contour analysis for bars and slices, Hough transform
    for circular markers, and shape analysis for other markers.
    
    Example:
        config = ElementDetectorConfig(detect_bars=True)
        detector = ElementDetector(config)
        result = detector.detect(binary_image, color_image)
    """
    
    def __init__(self, config: Optional[ElementDetectorConfig] = None):
        """
        Initialize detector.
        
        Args:
            config: Detection configuration (uses defaults if None)
        """
        self.config = config or ElementDetectorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray] = None,
        chart_id: str = "unknown",
        chart_type: Optional[str] = None,
    ) -> ElementDetectionResult:
        """
        Detect all discrete elements.
        
        Args:
            binary_image: Binary image (foreground=255)
            color_image: Optional BGR image for color extraction
            chart_id: Chart identifier for logging
            chart_type: Optional chart type hint for routing detection strategy
        
        Returns:
            ElementDetectionResult with detected elements
        """
        self.logger.debug(f"Element detection started | chart_id={chart_id} | chart_type={chart_type}")
        
        bars = []
        markers = []
        slices = []
        
        # [FIX] Route detection based on chart type to avoid false positives
        # Skip bar detection for chart types that don't have bars
        skip_bar_detection = chart_type in ("line", "scatter", "pie", "area")
        
        # [FIX] Additional heuristic: if chart_type is unknown, check if image looks like line/scatter
        # This helps when classifier fails but visual features are clear
        if not skip_bar_detection and chart_type in (None, "unknown") and color_image is not None:
            if self._looks_like_line_or_scatter(binary_image, color_image, chart_id):
                skip_bar_detection = True
                self.logger.debug(f"Heuristic detected line/scatter pattern | chart_id={chart_id}")
        
        # Use advanced bar detection if configured and appropriate for chart type
        if self.config.detect_bars and not skip_bar_detection:
            method = self.config.bar_separation_method
            
            if method == BarSeparationMethod.HYBRID:
                bars = self._detect_bars_hybrid(binary_image, color_image, chart_id)
            elif method == BarSeparationMethod.WATERSHED:
                bars = self._detect_bars_watershed(binary_image, color_image, chart_id)
            elif method == BarSeparationMethod.PROJECTION:
                bars = self._detect_bars_projection(binary_image, color_image, chart_id)
            elif method == BarSeparationMethod.MORPHOLOGICAL:
                bars = self._detect_bars_morphological(binary_image, color_image, chart_id)
            else:
                # Fallback to contour-only
                bars = self._detect_bars_contour(binary_image, color_image, chart_id)
        
        # Find contours for markers and slices
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.config.min_bar_area:
                continue
            
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self.config.contour_approx_epsilon * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            
            # Classify by shape - only markers since bars handled above
            num_vertices = len(approx)
            
            if num_vertices >= 6 and self.config.detect_markers:
                # Potential circular marker
                marker = self._analyze_circle(contour, color_image)
                if marker is not None:
                    markers.append(marker)
            
            elif num_vertices == 3 and self.config.detect_markers:
                # Triangle marker
                marker = self._analyze_triangle(approx, color_image)
                if marker is not None:
                    markers.append(marker)
        
        # Detect circular markers using Hough transform
        if self.config.detect_markers:
            hough_markers = self._detect_circles_hough(binary_image, color_image)
            markers.extend(hough_markers)
        
        self.logger.info(
            f"Element detection complete | chart_id={chart_id} | "
            f"bars={len(bars)} | markers={len(markers)} | slices={len(slices)}"
        )
        
        return ElementDetectionResult(
            bars=bars,
            markers=markers,
            slices=slices,
            contours_analyzed=len(contours),
        )
    
    # ========== Advanced Bar Detection Methods ==========
    
    def _detect_bars_contour(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
        chart_id: str,
    ) -> List[BarRectangle]:
        """
        Original contour-based bar detection.
        
        Simple but fails when bars merge into single contour.
        """
        bars = []
        
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_bar_area:
                continue
            
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self.config.contour_approx_epsilon * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            
            if len(approx) == 4:
                bar = self._analyze_rectangle(approx, contour, color_image)
                if bar is not None:
                    bars.append(bar)
        
        self.logger.debug(f"Contour method found {len(bars)} bars | chart_id={chart_id}")
        return bars
    
    def _detect_bars_watershed(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
        chart_id: str,
    ) -> List[BarRectangle]:
        """
        Use watershed segmentation to separate merged bars.
        
        Watershed treats the image as topography and finds
        basin boundaries to separate touching regions.
        """
        bars = []
        
        # Distance transform to find centers
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # Normalize and threshold to find sure foreground
        _, sure_fg = cv2.threshold(
            dist_transform, 
            self.config.watershed_markers_dist * 0.3,
            255, 
            cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)
        
        # Find sure background
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary_image, kernel, iterations=3)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Background is 1, not 0
        markers[unknown == 255] = 0  # Mark unknown as 0
        
        # Need 3-channel image for watershed
        if color_image is not None:
            img_for_watershed = color_image.copy()
        else:
            img_for_watershed = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        
        # Apply watershed
        markers = cv2.watershed(img_for_watershed, markers)
        
        # Extract each segmented region
        unique_labels = np.unique(markers)
        for label in unique_labels:
            if label <= 1:  # Skip background and boundary
                continue
            
            # Create mask for this region
            mask = np.uint8(markers == label) * 255
            
            # Find contours of this region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.min_bar_area:
                    continue
                
                # Get bounding rect and check if bar-like
                x, y, w, h = cv2.boundingRect(contour)
                bar = self._create_bar_from_bbox(x, y, w, h, contour, color_image)
                if bar is not None:
                    bars.append(bar)
        
        self.logger.debug(f"Watershed method found {len(bars)} bars | chart_id={chart_id}")
        return bars
    
    def _detect_bars_projection(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
        chart_id: str,
    ) -> List[BarRectangle]:
        """
        Use projection profiles to detect bars.
        
        Projects pixels onto X and Y axes to find bar boundaries
        based on pixel density changes.
        """
        bars = []
        h, w = binary_image.shape
        
        # Vertical projection (sum columns) - finds vertical bars
        v_proj = np.sum(binary_image > 0, axis=0).astype(float)
        v_proj = v_proj / h  # Normalize
        
        # Horizontal projection (sum rows) - finds horizontal bars
        h_proj = np.sum(binary_image > 0, axis=1).astype(float)
        h_proj = h_proj / w  # Normalize
        
        # Detect if predominantly vertical or horizontal bars
        v_variance = np.var(v_proj)
        h_variance = np.var(h_proj)
        
        if v_variance > h_variance:
            # Vertical bars - find column groups
            bars.extend(self._bars_from_projection(
                binary_image, v_proj, 'vertical', color_image
            ))
        else:
            # Horizontal bars - find row groups
            bars.extend(self._bars_from_projection(
                binary_image, h_proj, 'horizontal', color_image
            ))
        
        self.logger.debug(f"Projection method found {len(bars)} bars | chart_id={chart_id}")
        return bars
    
    def _bars_from_projection(
        self,
        binary_image: np.ndarray,
        projection: np.ndarray,
        orientation: str,
        color_image: Optional[np.ndarray],
    ) -> List[BarRectangle]:
        """Extract bars from projection profile."""
        bars = []
        h, w = binary_image.shape
        threshold = self.config.projection_threshold
        
        # Find regions above threshold
        above_thresh = projection > threshold
        
        # Find transitions (edges of bars)
        diff = np.diff(above_thresh.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if above_thresh[0]:
            starts = np.insert(starts, 0, 0)
        if above_thresh[-1]:
            ends = np.append(ends, len(projection))
        
        # Create bars from each region
        for start, end in zip(starts, ends):
            if end - start < self.config.min_bar_gap:
                continue
            
            if orientation == 'vertical':
                # Extract this column range
                col_slice = binary_image[:, start:end]
                row_sum = np.sum(col_slice > 0, axis=1)
                
                # Find vertical extent
                row_above = row_sum > 0
                row_indices = np.where(row_above)[0]
                if len(row_indices) == 0:
                    continue
                
                y_min = row_indices[0]
                y_max = row_indices[-1]
                
                bar = self._create_bar_from_bbox(
                    start, y_min, end - start, y_max - y_min + 1,
                    None, color_image
                )
            else:
                # Horizontal bar
                row_slice = binary_image[start:end, :]
                col_sum = np.sum(row_slice > 0, axis=0)
                
                col_above = col_sum > 0
                col_indices = np.where(col_above)[0]
                if len(col_indices) == 0:
                    continue
                
                x_min = col_indices[0]
                x_max = col_indices[-1]
                
                bar = self._create_bar_from_bbox(
                    x_min, start, x_max - x_min + 1, end - start,
                    None, color_image
                )
            
            if bar is not None:
                bars.append(bar)
        
        return bars
    
    def _detect_bars_morphological(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
        chart_id: str,
    ) -> List[BarRectangle]:
        """
        Use morphological operations to separate merged bars.
        
        Applies erosion to break connections between touching bars,
        then finds individual components.
        """
        bars = []
        
        # Multiple erosion passes to break connections
        eroded = binary_image.copy()
        
        # Erosion with small kernel
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(eroded, kernel, iterations=2)
        
        # Find connected components after erosion
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=8
        )
        
        # For each component, grow back using original image as mask
        for label in range(1, num_labels):  # Skip background (0)
            # Get component stats
            x, y, w, h, area = stats[label]
            
            if area < 10:  # Skip tiny fragments
                continue
            
            # Create mask from this component
            component_mask = (labels == label).astype(np.uint8) * 255
            
            # Dilate back to original size, limited by original binary
            dilated = cv2.dilate(component_mask, kernel, iterations=3)
            final_mask = cv2.bitwise_and(dilated, binary_image)
            
            # Find bounding rect of final region
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.config.min_bar_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                bar = self._create_bar_from_bbox(x, y, w, h, contour, color_image)
                if bar is not None:
                    bars.append(bar)
        
        self.logger.debug(f"Morphological method found {len(bars)} bars | chart_id={chart_id}")
        return bars
    
    def _detect_bars_by_color(
        self,
        color_image: np.ndarray,
        chart_id: str,
    ) -> List[BarRectangle]:
        """
        Detect bars using color segmentation in HSV space.
        
        [NEW] This method is specifically designed for colored bar charts.
        It identifies saturated (colored) regions and filters them by
        rectangular shape and size constraints.
        
        Algorithm:
        1. Convert to HSV color space
        2. Create mask for saturated pixels (colors, not grayscale)
        3. Find contours in the saturation mask
        4. Filter contours by area and aspect ratio
        5. Return valid bar rectangles
        
        Args:
            color_image: BGR color image
            chart_id: For logging context
            
        Returns:
            List of detected bar rectangles
        """
        bars = []
        h, w = color_image.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Saturation channel - high saturation = colored pixels
        saturation = hsv[:, :, 1]
        
        # Threshold saturation to find colored regions
        sat_threshold = self.config.color_saturation_threshold
        _, sat_mask = cv2.threshold(saturation, sat_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations (conservative to avoid merging thin bars)
        kernel_small = np.ones((2, 2), np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = h * w
        max_bar_area = img_area * self.config.max_bar_area_ratio
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip too small or too large
            if area < self.config.min_bar_area:
                continue
            if area > max_bar_area:
                continue
            
            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Check aspect ratio (allow both vertical and horizontal bars)
            aspect = bw / bh if bh > 0 else 0
            inv_aspect = bh / bw if bw > 0 else 0
            
            # Valid bar: either wide-short or tall-narrow
            is_valid_aspect = (
                self.config.bar_aspect_ratio_min <= aspect <= self.config.bar_aspect_ratio_max
                or self.config.bar_aspect_ratio_min <= inv_aspect <= self.config.bar_aspect_ratio_max
            )
            
            if not is_valid_aspect:
                continue
            
            # Check fill ratio (bar should be mostly filled)
            rect_area = bw * bh
            fill_ratio = area / rect_area if rect_area > 0 else 0
            if fill_ratio < 0.5:  # Must be at least 50% filled
                continue
            
            # [FIX] Filter out legend boxes
            # Legend boxes are typically:
            # 1. Small and nearly square (aspect ratio close to 1)
            # 2. Located at the right edge or top of the chart
            is_legend_like = self._is_likely_legend_box(x, y, bw, bh, w, h, area)
            if is_legend_like:
                self.logger.debug(
                    f"Skipping legend-like box | x={x}, y={y}, w={bw}, h={bh}, area={area}"
                )
                continue
            
            # Extract dominant color
            color = self._extract_dominant_color(contour, color_image)
            
            bar = BarRectangle(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + bw),
                y_max=float(y + bh),
                color=color,
            )
            bars.append(bar)
        
        self.logger.debug(
            f"Color segmentation found {len(bars)} bars | "
            f"chart_id={chart_id} | sat_threshold={sat_threshold}"
        )
        return bars
    
    def _is_likely_legend_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        img_width: int,
        img_height: int,
        area: float,
    ) -> bool:
        """
        Check if a detected box is likely a legend indicator, not a bar.
        
        Legend boxes are typically:
        1. Small and nearly square (aspect ratio close to 1)
        2. Located at the right edge (legend column) or top-right corner
        3. Much smaller than typical bars
        
        Args:
            x, y, w, h: Bounding box of the detected region
            img_width, img_height: Image dimensions
            area: Contour area
            
        Returns:
            True if the box appears to be a legend indicator
        """
        # Check aspect ratio (legend boxes are usually square or nearly square)
        aspect_ratio = w / h if h > 0 else 0
        is_nearly_square = 0.7 <= aspect_ratio <= 1.4
        
        # Check size (legend boxes are typically small, under 500-1000 pixels area)
        is_small = area < 800  # Legend boxes are usually ~200-400 pixels
        
        # Check position (legend is usually on the right side of the chart)
        right_margin_threshold = img_width * 0.75
        is_on_right_side = x > right_margin_threshold
        
        # Combined check: small, square, and on the right = legend
        if is_nearly_square and is_small and is_on_right_side:
            return True
        
        # Also check for very small boxes anywhere (likely labels/markers)
        if is_nearly_square and area < 400:
            return True
        
        return False
    
    def _is_likely_stacked(
        self,
        bars: List[BarRectangle],
        color_image: np.ndarray,
    ) -> bool:
        """
        Detect if the bar chart is likely a stacked bar chart.
        
        Stacked bar characteristics:
        1. Multiple distinct colors in the image
        2. Bars that are very tall relative to width (multiple segments merged)
        3. Each detected bar contains multiple colors
        
        Args:
            bars: Currently detected bars
            color_image: Original color image
            
        Returns:
            True if the chart appears to be stacked
        """
        if len(bars) < 2:
            return False
        
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return False
        
        h, w = color_image.shape[:2]
        
        # Check 1: Look for multiple colors WITHIN each bar region
        # This is the key indicator for stacked bars - each bar contains multiple colors
        bars_with_multiple_colors = 0
        
        for bar in bars:
            x1, y1 = max(0, int(bar.x_min)), max(0, int(bar.y_min))
            x2, y2 = min(w, int(bar.x_max)), min(h, int(bar.y_max))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            region = color_image[y1:y2, x1:x2]
            pixels = region.reshape(-1, 3)
            
            if len(pixels) < 100:
                continue
            
            # Sample pixels for K-Means
            n_samples = min(1000, len(pixels))
            sample_idx = np.random.choice(len(pixels), n_samples, replace=False)
            sample_pixels = pixels[sample_idx]
            
            try:
                # Try k=4 clusters per bar
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
                kmeans.fit(sample_pixels)
                centers = kmeans.cluster_centers_.astype(np.uint8)
                
                # Count non-background colors in this bar
                bar_colors = 0
                for c in centers:
                    b_val, g_val, r_val = int(c[0]), int(c[1]), int(c[2])
                    # Skip white, black, gray
                    if min(b_val, g_val, r_val) > 240:
                        continue
                    if max(b_val, g_val, r_val) < 20:
                        continue
                    if abs(b_val - g_val) < 20 and abs(g_val - r_val) < 20 and abs(b_val - r_val) < 20:
                        continue
                    bar_colors += 1
                
                if bar_colors >= 2:  # This bar has 2+ distinct colors = stacked
                    bars_with_multiple_colors += 1
                    
            except Exception:
                continue
        
        # If most bars have multiple colors, it's stacked
        if bars_with_multiple_colors >= len(bars) * 0.5:
            self.logger.debug(
                f"Stacked detection: {bars_with_multiple_colors}/{len(bars)} bars have multiple colors"
            )
            return True
        
        # Check 2: Bars are unusually tall (height > 2x width) - merged segments
        tall_bars = sum(1 for bar in bars if (bar.y_max - bar.y_min) > 2 * (bar.x_max - bar.x_min))
        if tall_bars >= len(bars) * 0.5:
            self.logger.debug(
                f"Stacked detection: {tall_bars}/{len(bars)} bars are very tall"
            )
            return True
        
        return False

    def _detect_stacked_bars_by_kmeans(
        self,
        color_image: np.ndarray,
        chart_id: str,
        n_colors: int = 8,
        min_area: int = 500,
    ) -> List[BarRectangle]:
        """
        Detect stacked bar segments using K-Means color clustering.
        
        [NEW] This method uses K-Means to cluster pixels by color,
        then finds contours within each color cluster. This allows
        separation of stacked segments that share boundaries.
        
        Algorithm:
        1. Apply K-Means clustering on pixel colors
        2. For each color cluster (excluding white/black):
           a. Create binary mask for that color
           b. Find contours in the mask
           c. Filter by area and shape
        3. Return all detected bar segments
        
        Args:
            color_image: BGR color image
            chart_id: For logging context
            n_colors: Number of K-Means clusters
            min_area: Minimum segment area
            
        Returns:
            List of detected bar rectangles with color info
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            self.logger.warning(
                "scikit-learn not available for K-Means detection | "
                f"chart_id={chart_id}"
            )
            return []
        
        bars = []
        h, w = color_image.shape[:2]
        
        # Reshape image for K-Means
        pixels = color_image.reshape(-1, 3).astype(np.float32)
        
        # Apply K-Means (use k=8 to oversegment, then filter)
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(np.uint8)
        except Exception as e:
            self.logger.warning(
                f"K-Means failed | chart_id={chart_id} | error={e}"
            )
            return []
        
        # Create label image
        labels_2d = labels.reshape(h, w)
        
        img_area = h * w
        max_bar_area = img_area * self.config.max_bar_area_ratio
        
        # Process each color cluster
        for cluster_idx in range(n_colors):
            center_color = centers[cluster_idx]
            b_val, g_val, r_val = int(center_color[0]), int(center_color[1]), int(center_color[2])
            
            # Skip white background (high values in all channels)
            if b_val > 240 and g_val > 240 and r_val > 240:
                continue
            
            # Skip black (axis lines)
            if b_val < 15 and g_val < 15 and r_val < 15:
                continue
            
            # Skip gray (text, minor elements)
            if abs(b_val - g_val) < 20 and abs(g_val - r_val) < 20 and b_val < 200:
                continue
            
            # Create mask for this cluster
            mask = (labels_2d == cluster_idx).astype(np.uint8) * 255
            
            # Find contours in this color mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                if area > max_bar_area:
                    continue
                
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Filter by aspect ratio
                aspect = bw / bh if bh > 0 else 0
                if aspect > 5 or aspect < 0.1:  # Too horizontal or vertical = noise
                    continue
                
                # Skip legend-like boxes
                if self._is_likely_legend_box(x, y, bw, bh, w, h, area):
                    continue
                
                # Create color object
                color = Color(r=r_val, g=g_val, b=b_val)
                
                bar = BarRectangle(
                    x_min=float(x),
                    y_min=float(y),
                    x_max=float(x + bw),
                    y_max=float(y + bh),
                    color=color,
                )
                bars.append(bar)
        
        self.logger.debug(
            f"K-Means stacked detection found {len(bars)} segments | "
            f"chart_id={chart_id} | n_colors={n_colors}"
        )
        return bars
    
    def _detect_pie_slices_by_kmeans(
        self,
        color_image: np.ndarray,
        chart_id: str,
        n_colors: int = 8,
        min_area: int = 1000,
    ) -> List[PieSlice]:
        """
        Detect pie chart slices using K-Means color clustering.
        
        [NEW] Similar to stacked bar detection, but returns PieSlice
        objects with angular information.
        
        Args:
            color_image: BGR color image
            chart_id: For logging context
            n_colors: Number of K-Means clusters
            min_area: Minimum slice area
            
        Returns:
            List of detected pie slices with color and angle info
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            self.logger.warning(
                "scikit-learn not available for K-Means detection | "
                f"chart_id={chart_id}"
            )
            return []
        
        slices = []
        h, w = color_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Estimate pie radius (use half of smaller dimension)
        estimated_radius = min(h, w) // 2 - 20
        
        # Reshape image for K-Means
        pixels = color_image.reshape(-1, 3).astype(np.float32)
        
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(np.uint8)
        except Exception as e:
            self.logger.warning(
                f"K-Means failed for pie slices | chart_id={chart_id} | error={e}"
            )
            return []
        
        labels_2d = labels.reshape(h, w)
        
        for cluster_idx in range(n_colors):
            center_color = centers[cluster_idx]
            b_val, g_val, r_val = int(center_color[0]), int(center_color[1]), int(center_color[2])
            
            # Skip white/black/gray
            if b_val > 240 and g_val > 240 and r_val > 240:
                continue
            if b_val < 15 and g_val < 15 and r_val < 15:
                continue
            if abs(b_val - g_val) < 20 and abs(g_val - r_val) < 20 and b_val < 200:
                continue
            
            mask = (labels_2d == cluster_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                
                # Calculate centroid
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate angle from pie center (in radians for schema)
                angle_start = math.atan2(cy - center_y, cx - center_x)
                
                # Estimate slice proportion based on area
                total_area = math.pi * estimated_radius * estimated_radius
                proportion = area / total_area if total_area > 0 else 0
                angle_span = proportion * 2 * math.pi
                angle_end = angle_start + angle_span
                
                color = Color(r=r_val, g=g_val, b=b_val)
                
                # Create PieSlice matching schema
                slice_obj = PieSlice(
                    center=PointFloat(x=float(center_x), y=float(center_y)),
                    radius_outer=float(estimated_radius),
                    radius_inner=0.0,  # Full pie, not donut
                    angle_start=angle_start,
                    angle_end=angle_end,
                    color=color,
                )
                slices.append(slice_obj)
        
        self.logger.debug(
            f"K-Means pie detection found {len(slices)} slices | chart_id={chart_id}"
        )
        return slices
    
    def _detect_bars_hybrid(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
        chart_id: str,
        chart_type: Optional[str] = None,
    ) -> List[BarRectangle]:
        """
        Combine multiple methods and vote on results.
        
        [FIX] Updated to prioritize color segmentation for colored bar charts.
        [NEW] Uses K-Means for stacked bars when chart_type indicates stacking.
        [NEW] Auto-detects stacked bars when color segmentation finds few bars
              but image has multiple distinct colors.
        Uses contour, watershed, projection, and color methods,
        then merges results based on overlap and confidence.
        """
        h, w = binary_image.shape[:2]
        
        # [NEW] For stacked charts, use K-Means color clustering
        # K-Means can separate touching segments by color
        if chart_type in ("stacked", "stacked_bar", "100_stacked") and color_image is not None:
            kmeans_bars = self._detect_stacked_bars_by_kmeans(color_image, chart_id)
            if len(kmeans_bars) >= 3:  # Stacked should have multiple segments
                self.logger.debug(
                    f"K-Means stacked detection found {len(kmeans_bars)} bars | chart_id={chart_id}"
                )
                return kmeans_bars
        
        # [FIX] Try color segmentation first (best for bar charts with colored bars)
        color_bars = []
        if self.config.use_color_segmentation and color_image is not None:
            color_bars = self._detect_bars_by_color(color_image, chart_id)
            if len(color_bars) >= 2:
                self.logger.debug(f"Color segmentation found {len(color_bars)} bars | chart_id={chart_id}")
                
                # [NEW] Auto-detect stacked bars:
                # If we found few bars but they might be stacked (overlapping X positions),
                # try K-Means for better separation
                # [FIX v2.1] Only try K-Means if chart_type explicitly indicates stacking
                # Auto-detection was causing false positives due to grid lines being
                # counted as "multiple colors" within bar regions
                if chart_type in ("stacked", "stacked_bar", "100_stacked"):
                    if self._is_likely_stacked(color_bars, color_image):
                        self.logger.debug(
                            f"Auto-detected stacked pattern | chart_id={chart_id} | "
                            f"trying K-Means for better separation"
                        )
                        kmeans_bars = self._detect_stacked_bars_by_kmeans(color_image, chart_id)
                        # [FIX v2.1] Only accept K-Means if result is reasonable
                        # (not more than 3x the original count - avoids grid line noise)
                        if len(color_bars) < len(kmeans_bars) <= len(color_bars) * 3:
                            self.logger.debug(
                                f"K-Means improved detection: {len(color_bars)} -> {len(kmeans_bars)} bars | "
                                f"chart_id={chart_id}"
                            )
                            return kmeans_bars
                        else:
                            self.logger.debug(
                                f"K-Means rejected (count explosion): {len(color_bars)} -> {len(kmeans_bars)} bars | "
                                f"chart_id={chart_id}"
                            )
                
                return color_bars
        
        # Try each traditional method
        contour_bars = self._detect_bars_contour(binary_image, color_image, chart_id)
        watershed_bars = self._detect_bars_watershed(binary_image, color_image, chart_id)
        projection_bars = self._detect_bars_projection(binary_image, color_image, chart_id)
        
        self.logger.debug(
            f"Hybrid detection | chart_id={chart_id} | "
            f"contour={len(contour_bars)} | watershed={len(watershed_bars)} | "
            f"projection={len(projection_bars)}"
        )
        
        # If contour found reasonable count, it's probably correct
        if 2 <= len(contour_bars) <= 20:
            return contour_bars
        
        # If contour failed (0 or 1 bar), watershed probably separated them
        if len(contour_bars) <= 1 and len(watershed_bars) > len(contour_bars):
            return watershed_bars
        
        # If projection found different count, merge unique bars
        all_bars = contour_bars + watershed_bars + projection_bars
        
        if not all_bars:
            return []
        
        # Merge overlapping bars
        merged_bars = self._merge_overlapping_bars(all_bars)
        
        return merged_bars
    
    def _merge_overlapping_bars(
        self,
        bars: List[BarRectangle],
        iou_threshold: float = 0.5,
    ) -> List[BarRectangle]:
        """
        Merge bars that significantly overlap.
        
        Uses IoU (Intersection over Union) to detect duplicates.
        """
        if len(bars) <= 1:
            return bars
        
        # Sort by area (largest first)
        bars = sorted(bars, key=lambda b: (b.x_max - b.x_min) * (b.y_max - b.y_min), reverse=True)
        
        merged = []
        used = [False] * len(bars)
        
        for i, bar in enumerate(bars):
            if used[i]:
                continue
            
            used[i] = True
            
            # Check overlap with remaining bars
            for j in range(i + 1, len(bars)):
                if used[j]:
                    continue
                
                iou = self._calculate_iou(bar, bars[j])
                if iou > iou_threshold:
                    used[j] = True  # Skip this bar (duplicate)
            
            merged.append(bar)
        
        return merged
    
    def _calculate_iou(self, bar1: BarRectangle, bar2: BarRectangle) -> float:
        """Calculate Intersection over Union of two bars."""
        # Intersection
        x_left = max(bar1.x_min, bar2.x_min)
        y_top = max(bar1.y_min, bar2.y_min)
        x_right = min(bar1.x_max, bar2.x_max)
        y_bottom = min(bar1.y_max, bar2.y_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Union
        area1 = (bar1.x_max - bar1.x_min) * (bar1.y_max - bar1.y_min)
        area2 = (bar2.x_max - bar2.x_min) * (bar2.y_max - bar2.y_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_bar_from_bbox(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        contour: Optional[np.ndarray],
        color_image: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[BarRectangle]:
        """Create BarRectangle from bounding box if valid."""
        if w <= 0 or h <= 0:
            return None
        
        # Check aspect ratio
        aspect = w / h
        if not (self.config.bar_aspect_ratio_min <= aspect <= self.config.bar_aspect_ratio_max):
            inv_aspect = h / w
            if not (self.config.bar_aspect_ratio_min <= inv_aspect <= self.config.bar_aspect_ratio_max):
                return None
        
        bar_area = w * h
        
        # Check minimum area
        if bar_area < self.config.min_bar_area:
            return None
        
        # [FIX] Check maximum area (filter background/oversized contours)
        if image_shape is not None:
            img_h, img_w = image_shape
            img_area = img_h * img_w
            max_bar_area = img_area * self.config.max_bar_area_ratio
            if bar_area > max_bar_area:
                self.logger.debug(
                    f"Rejected oversized bar: {w}x{h} = {bar_area} > max {max_bar_area:.0f}"
                )
                return None
        elif color_image is not None:
            img_h, img_w = color_image.shape[:2]
            img_area = img_h * img_w
            max_bar_area = img_area * self.config.max_bar_area_ratio
            if bar_area > max_bar_area:
                self.logger.debug(
                    f"Rejected oversized bar: {w}x{h} = {bar_area} > max {max_bar_area:.0f}"
                )
                return None
        
        # Extract color
        color = None
        if self.config.extract_colors and color_image is not None and contour is not None:
            color = self._extract_dominant_color(contour, color_image)
        elif self.config.extract_colors and color_image is not None:
            # Sample color from center of bbox
            cx, cy = x + w // 2, y + h // 2
            if 0 <= cy < color_image.shape[0] and 0 <= cx < color_image.shape[1]:
                bgr = color_image[cy, cx]
                color = Color(r=int(bgr[2]), g=int(bgr[1]), b=int(bgr[0]))
        
        return BarRectangle(
            x_min=float(x),
            y_min=float(y),
            x_max=float(x + w),
            y_max=float(y + h),
            color=color,
        )
    
    # ========== End Advanced Bar Detection ==========
    
    def _analyze_rectangle(
        self,
        approx: np.ndarray,
        contour: np.ndarray,
        color_image: Optional[np.ndarray],
    ) -> Optional[BarRectangle]:
        """Analyze 4-vertex contour as potential bar rectangle."""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Check aspect ratio
        aspect = w / h if h > 0 else 0
        if not (self.config.bar_aspect_ratio_min <= aspect <= self.config.bar_aspect_ratio_max):
            # Also check inverse (vertical bars)
            inv_aspect = h / w if w > 0 else 0
            if not (self.config.bar_aspect_ratio_min <= inv_aspect <= self.config.bar_aspect_ratio_max):
                return None
        
        # Check if approximately rectangular (angles ~90 degrees)
        if not self._is_rectangle(approx):
            return None
        
        # Extract color
        color = None
        if self.config.extract_colors and color_image is not None:
            color = self._extract_dominant_color(contour, color_image)
        
        return BarRectangle(
            x_min=float(x),
            y_min=float(y),
            x_max=float(x + w),
            y_max=float(y + h),
            color=color,
        )
    
    def _is_rectangle(self, approx: np.ndarray) -> bool:
        """Check if 4-point polygon is approximately rectangular."""
        if len(approx) != 4:
            return False
        
        # Calculate angles at each corner
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            p3 = approx[(i + 2) % 4][0]
            
            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = math.acos(np.clip(cos_angle, -1, 1))
            angles.append(abs(angle - math.pi / 2))
        
        # All angles should be close to 90 degrees
        return all(a < 0.3 for a in angles)  # ~17 degree tolerance
    
    def _analyze_circle(
        self,
        contour: np.ndarray,
        color_image: Optional[np.ndarray],
    ) -> Optional[DataMarker]:
        """Analyze contour as potential circular marker."""
        # Fit minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Check size
        diameter = 2 * radius
        if not (self.config.min_marker_size <= diameter <= self.config.max_marker_size):
            return None
        
        # Check circularity
        area = cv2.contourArea(contour)
        expected_area = math.pi * radius * radius
        circularity = area / expected_area if expected_area > 0 else 0
        
        if circularity < 0.7:
            return None
        
        # Extract color
        color = None
        if self.config.extract_colors and color_image is not None:
            color = self._extract_dominant_color(contour, color_image)
        
        return DataMarker(
            center=PointFloat(x=cx, y=cy),
            marker_type=MarkerType.CIRCLE,
            size=diameter,
            color=color,
        )
    
    def _analyze_triangle(
        self,
        approx: np.ndarray,
        color_image: Optional[np.ndarray],
    ) -> Optional[DataMarker]:
        """Analyze 3-point polygon as triangle marker."""
        if len(approx) != 3:
            return None
        
        # Calculate center
        points = approx.reshape(-1, 2)
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        
        # Calculate size (average distance from center to vertices)
        distances = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in points]
        size = 2 * np.mean(distances)
        
        if not (self.config.min_marker_size <= size <= self.config.max_marker_size):
            return None
        
        # Extract color
        color = None
        if self.config.extract_colors and color_image is not None:
            mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [approx], 0, 255, -1)
            color = self._color_from_mask(mask, color_image)
        
        return DataMarker(
            center=PointFloat(x=cx, y=cy),
            marker_type=MarkerType.TRIANGLE,
            size=size,
            color=color,
        )
    
    def _detect_circles_hough(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
    ) -> List[DataMarker]:
        """Detect circular markers using Hough Circle Transform."""
        markers = []
        
        # Apply slight blur for better Hough detection
        blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.config.min_marker_size,
            param1=50,
            param2=30,
            minRadius=self.config.min_marker_size // 2,
            maxRadius=self.config.max_marker_size // 2,
        )
        
        if circles is None:
            return markers
        
        circles = np.round(circles[0, :]).astype(int)
        
        for cx, cy, r in circles:
            diameter = 2 * r
            
            # Extract color
            color = None
            if self.config.extract_colors and color_image is not None:
                mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                color = self._color_from_mask(mask, color_image)
            
            markers.append(DataMarker(
                center=PointFloat(x=float(cx), y=float(cy)),
                marker_type=MarkerType.CIRCLE,
                size=float(diameter),
                color=color,
            ))
        
        return markers
    
    def _extract_dominant_color(
        self,
        contour: np.ndarray,
        color_image: np.ndarray,
    ) -> Optional[Color]:
        """Extract dominant color inside contour."""
        # Create mask
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        return self._color_from_mask(mask, color_image)
    
    def _color_from_mask(
        self,
        mask: np.ndarray,
        color_image: np.ndarray,
    ) -> Optional[Color]:
        """Get mean color from masked region."""
        # Get pixels inside mask
        mean_bgr = cv2.mean(color_image, mask=mask)[:3]
        
        return Color(
            r=int(mean_bgr[2]),
            g=int(mean_bgr[1]),
            b=int(mean_bgr[0]),
        )
    
    def detect_pie_center(
        self,
        binary_image: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Detect pie chart center and radius.
        
        Args:
            binary_image: Binary image
        
        Returns:
            (center_x, center_y, radius) or None
        """
        # Find largest circular contour
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Fit circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        
        # Check if circular enough
        area = cv2.contourArea(largest)
        expected = math.pi * radius * radius
        circularity = area / expected if expected > 0 else 0
        
        if circularity < 0.5:
            return None
        
        return (cx, cy, radius)

    def _looks_like_line_or_scatter(
        self,
        binary_image: np.ndarray,
        color_image: np.ndarray,
        chart_id: str,
    ) -> bool:
        """
        Heuristic to detect if an image looks like a line or scatter chart.
        
        Used when classifier returns 'unknown' to avoid false positive bars.
        Key insight: Line charts have colored regions with LOW SOLIDITY
        (thin lines with gaps) while bar charts have HIGH SOLIDITY (filled rectangles).
        
        Returns:
            True if image appears to be line/scatter chart
        """
        h, w = binary_image.shape[:2]
        
        # Use color segmentation to find colored elements
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Detect colored regions (saturation > 30)
        color_mask = (saturation > 30).astype(np.uint8) * 255
        
        # Find contours of colored regions
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return False
        
        # Analyze colored contours
        total_colored_area = 0
        low_solidity_area = 0
        high_solidity_rectangles = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Too small to analyze
                continue
            
            total_colored_area += area
            
            # Get convex hull to compute solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Get bounding rect properties
            x, y, bw, bh = cv2.boundingRect(contour)
            rect_area = bw * bh
            extent = area / rect_area if rect_area > 0 else 0
            
            # Classify based on solidity:
            # - Low solidity (< 0.3) = line pattern (thin with gaps)
            # - High solidity (> 0.7) + high extent = filled rectangle = bar
            
            if solidity < 0.3:
                low_solidity_area += area
            elif solidity > 0.7 and extent > 0.7 and area > 500:
                high_solidity_rectangles += 1
        
        # Heuristic decision:
        # If most colored area has low solidity → line/scatter chart
        # If many high-solidity rectangles → bar chart
        
        if total_colored_area == 0:
            return False
        
        low_solidity_ratio = low_solidity_area / total_colored_area
        
        looks_like_line = (
            low_solidity_ratio > 0.5  # More than half is line-like
            or (low_solidity_ratio > 0.3 and high_solidity_rectangles == 0)
        )
        
        if looks_like_line:
            self.logger.debug(
                f"Line/scatter heuristic triggered | chart_id={chart_id} | "
                f"low_solidity_ratio={low_solidity_ratio:.2f} | "
                f"high_solidity_rects={high_solidity_rectangles}"
            )
        
        return looks_like_line
    
    def validate_elements(
        self,
        chart_type: str,
        bars: List[BarRectangle],
        markers: List[DataMarker],
        slices: List[PieSlice],
        image_shape: Tuple[int, int],
    ) -> dict:
        """
        Validate detected elements against chart type expectations.
        
        Performs cross-validation checks:
        - Bar chart should have bars, not many markers
        - Line/scatter should have markers, not bars
        - Pie chart should have slices
        - Element positions should be consistent
        
        Args:
            chart_type: Detected chart type
            bars: Detected bars
            markers: Detected markers
            slices: Detected slices
            image_shape: (height, width) of image
        
        Returns:
            Dict with validation results:
            {
                'is_valid': bool,
                'confidence': float,
                'issues': List[str],
                'suggestions': List[str]
            }
        """
        h, w = image_shape
        issues = []
        suggestions = []
        confidence = 1.0
        
        # Chart type specific validation
        if chart_type in ('bar', 'histogram'):
            if len(bars) == 0:
                issues.append("Bar/histogram chart detected but no bars found")
                confidence *= 0.5
                suggestions.append("Check if bars merged into background or use color segmentation")
            elif len(bars) < 2:
                issues.append(f"Only {len(bars)} bar(s) found, expected more for bar chart")
                confidence *= 0.7
            
            # Bars should have consistent width (for grouped bars)
            if len(bars) >= 2:
                widths = [bar.width for bar in bars]
                width_std = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
                if width_std > 0.5:
                    issues.append(f"Bar widths inconsistent (CV={width_std:.2f})")
                    confidence *= 0.9
            
            # Bars should be aligned (same baseline)
            if len(bars) >= 2:
                baselines = [bar.y + bar.height for bar in bars]  # Bottom of bars
                baseline_range = max(baselines) - min(baselines)
                if baseline_range > h * 0.1:
                    issues.append(f"Bar baselines not aligned (range={baseline_range:.0f}px)")
                    confidence *= 0.8
        
        elif chart_type in ('line', 'area'):
            if len(markers) == 0 and len(bars) > 0:
                issues.append("Line chart detected but found bars instead of markers")
                confidence *= 0.6
                suggestions.append("Classification might be incorrect, check for bar chart")
            
            # Check if markers are roughly aligned (following a trend)
            if len(markers) >= 3:
                xs = [m.center.x for m in markers]
                ys = [m.center.y for m in markers]
                if not self._check_trend_alignment(xs, ys):
                    issues.append("Markers don't follow expected line pattern")
                    confidence *= 0.8
        
        elif chart_type == 'scatter':
            if len(markers) < 5:
                issues.append(f"Scatter plot should have many points, found only {len(markers)}")
                confidence *= 0.7
            
            if len(bars) > 0:
                issues.append("Scatter plot detected but found bars")
                confidence *= 0.6
        
        elif chart_type == 'pie':
            if len(slices) == 0:
                issues.append("Pie chart detected but no slices found")
                confidence *= 0.4
                suggestions.append("Check color segmentation for pie detection")
            else:
                # Slices should sum to ~360 degrees (or 2*pi radians)
                total_angle = sum(s.end_angle - s.start_angle for s in slices)
                expected_total = 2 * np.pi
                if abs(total_angle - expected_total) > 0.2 * expected_total:
                    issues.append(f"Slice angles don't sum to 360° (got {np.degrees(total_angle):.1f}°)")
                    confidence *= 0.7
        
        elif chart_type == 'box':
            # Box plots have specific structure
            if len(bars) < 2:
                issues.append("Box plot should have multiple boxes")
                confidence *= 0.6
        
        elif chart_type == 'heatmap':
            # Heatmaps should have grid-like bars
            if len(bars) > 0:
                # Check for grid alignment
                xs = sorted(set(int(bar.x) for bar in bars))
                ys = sorted(set(int(bar.y) for bar in bars))
                
                if len(xs) < 2 or len(ys) < 2:
                    issues.append("Heatmap should have grid structure")
                    confidence *= 0.7
        
        # General element sanity checks
        if len(bars) + len(markers) + len(slices) == 0:
            issues.append("No chart elements detected")
            confidence = 0.1
            suggestions.append("Check image preprocessing settings")
        
        # Check for out-of-bounds elements
        out_of_bounds = 0
        for bar in bars:
            if bar.x < 0 or bar.y < 0 or bar.x + bar.width > w or bar.y + bar.height > h:
                out_of_bounds += 1
        for marker in markers:
            if marker.center.x < 0 or marker.center.y < 0 or marker.center.x > w or marker.center.y > h:
                out_of_bounds += 1
        
        if out_of_bounds > 0:
            issues.append(f"{out_of_bounds} element(s) out of bounds")
            confidence *= 0.9
        
        return {
            'is_valid': len(issues) == 0 or confidence >= 0.5,
            'confidence': max(0.0, min(1.0, confidence)),
            'issues': issues,
            'suggestions': suggestions,
        }
    
    def _check_trend_alignment(
        self,
        xs: List[float],
        ys: List[float],
        threshold: float = 0.3,
    ) -> bool:
        """Check if points roughly follow a trend (for line charts)."""
        if len(xs) < 3:
            return True
        
        # Sort by x and check if y values have some structure
        points = sorted(zip(xs, ys), key=lambda p: p[0])
        sorted_ys = [p[1] for p in points]
        
        # Compute simple linear regression R²
        xs_arr = np.array([p[0] for p in points])
        ys_arr = np.array(sorted_ys)
        
        if len(np.unique(xs_arr)) < 2:
            return True
        
        # Fit line
        A = np.vstack([xs_arr, np.ones(len(xs_arr))]).T
        try:
            m, c = np.linalg.lstsq(A, ys_arr, rcond=None)[0]
        except np.linalg.LinAlgError:
            return True
        
        predicted = m * xs_arr + c
        ss_res = np.sum((ys_arr - predicted) ** 2)
        ss_tot = np.sum((ys_arr - np.mean(ys_arr)) ** 2)
        
        if ss_tot < 1e-10:
            return True
        
        r_squared = 1.0 - (ss_res / ss_tot)
        
        # For trend alignment, we expect some correlation
        # But not perfect (that would be very rare)
        return r_squared > threshold or r_squared < -threshold  # Includes negative trends
