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

Reference: docs/instruction_p2_research.md - Section 3.2
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ...schemas.common import Color
from ...schemas.extraction import (
    BarRectangle,
    DataMarker,
    MarkerType,
    PieSlice,
    PointFloat,
)

logger = logging.getLogger(__name__)


class ElementDetectorConfig(BaseModel):
    """Configuration for element detection."""
    
    # Bar detection
    detect_bars: bool = Field(default=True, description="Detect bar rectangles")
    min_bar_area: int = Field(default=100, ge=1, description="Minimum bar area in pixels")
    bar_aspect_ratio_min: float = Field(default=0.1, gt=0, description="Min aspect ratio for bars")
    bar_aspect_ratio_max: float = Field(default=10.0, gt=0, description="Max aspect ratio for bars")
    
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
        
        # Find contours
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
            
            # Classify by shape
            num_vertices = len(approx)
            
            if num_vertices == 4 and self.config.detect_bars:
                # Potential bar rectangle
                bar = self._analyze_rectangle(approx, contour, color_image)
                if bar is not None:
                    bars.append(bar)
            
            elif num_vertices >= 6 and self.config.detect_markers:
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
