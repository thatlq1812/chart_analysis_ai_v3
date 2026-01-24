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
    bar_aspect_ratio_min: float = Field(default=0.1, gt=0, description="Min aspect ratio for bars")
    bar_aspect_ratio_max: float = Field(default=10.0, gt=0, description="Max aspect ratio for bars")
    
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
    ) -> ElementDetectionResult:
        """
        Detect all discrete elements.
        
        Args:
            binary_image: Binary image (foreground=255)
            color_image: Optional BGR image for color extraction
            chart_id: Chart identifier for logging
        
        Returns:
            ElementDetectionResult with detected elements
        """
        self.logger.debug(f"Element detection started | chart_id={chart_id}")
        
        bars = []
        markers = []
        slices = []
        
        # Use advanced bar detection if configured
        if self.config.detect_bars:
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
    
    def _detect_bars_hybrid(
        self,
        binary_image: np.ndarray,
        color_image: Optional[np.ndarray],
        chart_id: str,
    ) -> List[BarRectangle]:
        """
        Combine multiple methods and vote on results.
        
        Uses contour, watershed, and projection methods,
        then merges results based on overlap and confidence.
        """
        # Try each method
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
        
        # Check minimum area
        if w * h < self.config.min_bar_area:
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
