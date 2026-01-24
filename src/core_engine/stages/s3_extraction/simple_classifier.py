"""
Simple Image-based Chart Classifier.

Uses image-level features instead of detected elements for classification.
This approach is more robust when element detection is weak.

Author: That Le
Date: 2025-01-21
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ...schemas.enums import ChartType

logger = logging.getLogger(__name__)


class SimpleClassifierConfig(BaseModel):
    """Configuration for simple chart classifier."""
    
    # Edge analysis
    edge_canny_low: int = Field(default=50, ge=0)
    edge_canny_high: int = Field(default=150, ge=0)
    line_kernel_length: int = Field(default=25, ge=5)
    
    # Color analysis
    color_cluster_k: int = Field(default=5, ge=2, le=20)
    min_saturation: int = Field(default=30, ge=0, le=255)
    
    # Circular detection
    hough_dp: float = Field(default=1.2, gt=0)
    hough_min_dist: int = Field(default=50, ge=10)
    hough_param1: int = Field(default=100, ge=10)
    hough_param2: int = Field(default=30, ge=10)
    min_radius: int = Field(default=20, ge=5)
    max_radius: int = Field(default=300, ge=10)
    
    # Classification thresholds
    pie_circularity_threshold: float = Field(default=0.4, ge=0, le=1)
    bar_edge_ratio_threshold: float = Field(default=1.5, ge=1)
    scatter_marker_threshold: int = Field(default=10, ge=3)


@dataclass
class SimpleClassificationResult:
    """Result of simple classification."""
    
    chart_type: ChartType
    confidence: float
    features: Dict[str, float]
    reasoning: str


class SimpleChartClassifier:
    """
    Simple image-based chart classifier.
    
    Uses image-level features:
    - Edge orientation (horizontal vs vertical vs diagonal)
    - Color distribution (number of distinct colors)
    - Circular structure detection
    - Grid pattern detection
    
    Example:
        classifier = SimpleChartClassifier()
        result = classifier.classify(image)
    """
    
    def __init__(self, config: Optional[SimpleClassifierConfig] = None):
        """
        Initialize classifier.
        
        Args:
            config: Classifier configuration
        """
        self.config = config or SimpleClassifierConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def classify(
        self,
        image: np.ndarray,
        chart_id: str = "unknown",
    ) -> SimpleClassificationResult:
        """
        Classify chart type from image.
        
        Args:
            image: BGR color image
            chart_id: Chart identifier for logging
        
        Returns:
            SimpleClassificationResult with type and confidence
        """
        self.logger.debug(f"Simple classification started | chart_id={chart_id}")
        
        # Compute features
        features = self._compute_features(image)
        
        # Score each chart type
        scores = {
            ChartType.PIE: self._score_pie(features),
            ChartType.BAR: self._score_bar(features),
            ChartType.LINE: self._score_line(features),
            ChartType.SCATTER: self._score_scatter(features),
        }
        
        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, scores, best_type)
        
        self.logger.info(
            f"Simple classification complete | chart_id={chart_id} | "
            f"type={best_type.value} | confidence={best_score:.2f}"
        )
        
        return SimpleClassificationResult(
            chart_type=best_type,
            confidence=best_score,
            features=features,
            reasoning=reasoning,
        )
    
    def _compute_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute image-level features."""
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # Convert color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Edge orientation analysis
        edges = cv2.Canny(gray, self.config.edge_canny_low, self.config.edge_canny_high)
        h_ratio, v_ratio, d_ratio = self._analyze_edge_orientation(edges)
        
        # 2. Color analysis
        n_colors, color_coverage = self._analyze_colors(hsv)
        
        # 3. Circular structure detection
        circularity, n_circles = self._detect_circles(gray, image.shape[:2])
        
        # 4. Grid pattern (common in line charts)
        grid_score = self._detect_grid_pattern(edges)
        
        # 5. Small blob detection (markers in scatter plots)
        n_markers = self._count_small_blobs(gray)
        
        # 6. Rectangular region detection
        n_rects, rect_coverage = self._detect_rectangles(gray)
        
        return {
            "h_edge_ratio": h_ratio,
            "v_edge_ratio": v_ratio,
            "d_edge_ratio": d_ratio,
            "n_colors": float(n_colors),
            "color_coverage": color_coverage,
            "circularity": circularity,
            "n_circles": float(n_circles),
            "grid_score": grid_score,
            "n_markers": float(n_markers),
            "n_rectangles": float(n_rects),
            "rect_coverage": rect_coverage,
        }
    
    def _analyze_edge_orientation(
        self,
        edges: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Analyze edge orientations.
        
        Returns:
            Tuple of (horizontal_ratio, vertical_ratio, diagonal_ratio)
        """
        # Morphological line detection
        klen = self.config.line_kernel_length
        
        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        h_count = np.sum(h_lines > 0)
        
        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        v_count = np.sum(v_lines > 0)
        
        # Diagonal lines (45 degrees)
        d_kernel = np.eye(klen, dtype=np.uint8)
        d_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, d_kernel)
        d_count = np.sum(d_lines > 0)
        
        total = h_count + v_count + d_count + 1  # +1 to avoid division by zero
        
        return h_count / total, v_count / total, d_count / total
    
    def _analyze_colors(
        self,
        hsv: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Analyze color distribution.
        
        Returns:
            Tuple of (number_of_colors, color_coverage_ratio)
        """
        h, w = hsv.shape[:2]
        
        # Filter out low saturation (gray/white/black)
        saturation = hsv[:, :, 1]
        mask = saturation > self.config.min_saturation
        colored_ratio = np.sum(mask) / (h * w)
        
        if colored_ratio < 0.05:
            # Mostly grayscale
            return 1, colored_ratio
        
        # Quantize hue to count distinct colors
        hue = hsv[:, :, 0][mask]
        if len(hue) == 0:
            return 1, colored_ratio
        
        # Bin hue into 12 segments (30 degrees each)
        hue_bins = np.histogram(hue, bins=12, range=(0, 180))[0]
        n_colors = np.sum(hue_bins > len(hue) * 0.02)  # At least 2% of pixels
        
        return max(1, n_colors), colored_ratio
    
    def _detect_circles(
        self,
        gray: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, int]:
        """
        Detect circular structures using edge-based contour analysis.
        
        Uses Canny edges + contour circularity for robust detection.
        A pie chart typically has arc-shaped edges forming circular patterns.
        
        Returns:
            Tuple of (circularity_score, number_of_circular_contours)
        """
        h, w = image_shape
        total_area = h * w
        
        # Use Canny edges instead of binary threshold
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circular_count = 0
        max_circularity = 0.0
        
        # Calculate circularity for all contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Min area
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity: 4*pi*area / perimeter^2
            # Perfect circle = 1.0
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Check if this contour is a large, highly circular shape
            # (pie charts have a dominant circular contour)
            if circularity > 0.6 and area / total_area > 0.1:
                circular_count += 1
                max_circularity = max(max_circularity, circularity * area / total_area)
        
        # Also check for arc-like contours (partial circles in pie charts)
        arc_score = self._detect_arcs(contours, image_shape)
        
        circularity_score = max(max_circularity, arc_score * 0.8)
        
        return min(1.0, circularity_score), circular_count
    
    def _detect_arcs(
        self,
        contours: List,
        image_shape: Tuple[int, int],
    ) -> float:
        """
        Detect arc-like contours (common in pie charts).
        
        Returns:
            Arc score (0-1)
        """
        h, w = image_shape
        center = (w // 2, h // 2)
        
        arc_count = 0
        total_arc_length = 0
        
        for contour in contours:
            if len(contour) < 10:
                continue
            
            # Fit ellipse if enough points
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (ex, ey), (ma, mi), angle = ellipse
                    
                    # Check if ellipse center is near image center
                    # (pie chart arcs converge toward center)
                    dist_to_center = math.sqrt((ex - center[0])**2 + (ey - center[1])**2)
                    max_dist = math.sqrt(w**2 + h**2) / 4
                    
                    if dist_to_center < max_dist and ma > 30 and mi > 30:
                        arc_count += 1
                        total_arc_length += cv2.arcLength(contour, False)
                except:
                    pass
        
        # Score based on number of arcs converging to center
        if arc_count >= 3:
            return min(1.0, arc_count / 10 + total_arc_length / (2 * math.pi * min(h, w)))
        
        return 0.0
    
    def _detect_grid_pattern(self, edges: np.ndarray) -> float:
        """
        Detect grid pattern (common in line charts).
        
        Returns:
            Grid score (0-1)
        """
        h, w = edges.shape
        
        # Sum along rows and columns
        row_sums = np.sum(edges, axis=1)
        col_sums = np.sum(edges, axis=0)
        
        # Find peaks (grid lines)
        row_threshold = np.mean(row_sums) * 2
        col_threshold = np.mean(col_sums) * 2
        
        row_peaks = np.sum(row_sums > row_threshold)
        col_peaks = np.sum(col_sums > col_threshold)
        
        # Grid charts typically have multiple evenly-spaced peaks
        # Normalize by image dimensions
        grid_score = min(1.0, (row_peaks + col_peaks) / (h + w) * 10)
        
        return grid_score
    
    def _count_small_blobs(self, gray: np.ndarray) -> int:
        """
        Count small circular blobs (markers in scatter plots).
        
        Returns:
            Number of detected markers
        """
        # Setup SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect on both original and inverted
        keypoints1 = detector.detect(gray)
        keypoints2 = detector.detect(255 - gray)
        
        return len(keypoints1) + len(keypoints2)
    
    def _detect_rectangles(
        self,
        gray: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Detect rectangular regions (bars).
        
        Returns:
            Tuple of (number_of_rectangles, coverage_ratio)
        """
        h, w = gray.shape
        
        # Threshold to find distinct regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rect_count = 0
        rect_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Min area
                continue
            
            # Check if rectangular
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            if len(approx) == 4:
                # Check aspect ratio
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect = max(cw / ch, ch / cw) if min(cw, ch) > 0 else 0
                
                if 1.5 <= aspect <= 10:
                    rect_count += 1
                    rect_area += area
        
        coverage = rect_area / (h * w)
        return rect_count, coverage
    
    def _score_pie(self, features: Dict[str, float]) -> float:
        """Score likelihood of pie chart."""
        score = 0.0
        
        # Strong indicator: circular structure
        score += 0.5 * features["circularity"]
        
        # Multiple colors typical
        if features["n_colors"] >= 3:
            score += 0.2
        
        # High color coverage
        score += 0.2 * features["color_coverage"]
        
        # Penalty for grid pattern (rare in pie charts)
        score -= 0.2 * features["grid_score"]
        
        # Penalty for many rectangles
        if features["n_rectangles"] > 3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_bar(self, features: Dict[str, float]) -> float:
        """Score likelihood of bar chart."""
        score = 0.0
        
        # Strong indicator: vertical or horizontal edges dominant
        edge_ratio = max(features["v_edge_ratio"], features["h_edge_ratio"])
        score += 0.3 * edge_ratio * 3  # Amplify
        
        # Rectangles detected
        if features["n_rectangles"] >= 2:
            score += 0.3
        score += 0.2 * min(1.0, features["rect_coverage"] * 5)
        
        # Multiple colors
        if features["n_colors"] >= 2:
            score += 0.1
        
        # Penalty for circular structure
        score -= 0.3 * features["circularity"]
        
        # Penalty for many markers (likely scatter)
        if features["n_markers"] > 20:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_line(self, features: Dict[str, float]) -> float:
        """Score likelihood of line chart."""
        score = 0.0
        
        # Grid pattern common in line charts
        score += 0.3 * features["grid_score"]
        
        # Diagonal edges (lines are often diagonal)
        score += 0.2 * features["d_edge_ratio"] * 5
        
        # Some markers (data points)
        if 3 <= features["n_markers"] <= 30:
            score += 0.2
        
        # Few distinct colors
        if features["n_colors"] <= 4:
            score += 0.1
        
        # Penalty for circular structure
        score -= 0.3 * features["circularity"]
        
        # Penalty for many rectangles
        if features["n_rectangles"] > 5:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_scatter(self, features: Dict[str, float]) -> float:
        """Score likelihood of scatter plot."""
        score = 0.0
        
        # Strong indicator: many small markers
        if features["n_markers"] >= self.config.scatter_marker_threshold:
            score += 0.4 * min(1.0, features["n_markers"] / 50)
        
        # Grid pattern common
        score += 0.2 * features["grid_score"]
        
        # Few rectangles
        if features["n_rectangles"] <= 2:
            score += 0.1
        
        # Penalty for circular structure
        score -= 0.3 * features["circularity"]
        
        # Penalty for dominant edge orientation (scatter is more random)
        max_edge = max(features["h_edge_ratio"], features["v_edge_ratio"])
        if max_edge > 0.5:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_reasoning(
        self,
        features: Dict[str, float],
        scores: Dict[ChartType, float],
        best_type: ChartType,
    ) -> str:
        """Generate human-readable reasoning."""
        reasons = []
        
        if best_type == ChartType.PIE:
            if features["circularity"] > 0.3:
                reasons.append(f"Strong circular structure (score={features['circularity']:.2f})")
            if features["n_colors"] >= 3:
                reasons.append(f"Multiple colors detected ({int(features['n_colors'])})")
        
        elif best_type == ChartType.BAR:
            if features["n_rectangles"] >= 2:
                reasons.append(f"Rectangular regions detected ({int(features['n_rectangles'])})")
            edge_ratio = max(features["v_edge_ratio"], features["h_edge_ratio"])
            if edge_ratio > 0.3:
                reasons.append(f"Strong H/V edges (ratio={edge_ratio:.2f})")
        
        elif best_type == ChartType.LINE:
            if features["grid_score"] > 0.3:
                reasons.append(f"Grid pattern detected (score={features['grid_score']:.2f})")
            if features["d_edge_ratio"] > 0.1:
                reasons.append(f"Diagonal edges present (ratio={features['d_edge_ratio']:.2f})")
        
        elif best_type == ChartType.SCATTER:
            if features["n_markers"] >= 10:
                reasons.append(f"Many small markers ({int(features['n_markers'])})")
        
        if not reasons:
            reasons.append("Classification based on combined feature scores")
        
        return f"Classified as {best_type.value}: " + "; ".join(reasons)
