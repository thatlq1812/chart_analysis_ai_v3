"""
Simple Image-based Chart Classifier.

Uses image-level features instead of detected elements for classification.
This approach is more robust when element detection is weak.

Features (v2.0):
- Edge orientation (horizontal vs vertical vs diagonal)
- Color distribution (number of distinct colors)
- Circular structure detection
- Grid pattern detection
- **NEW** Texture features (grayscale-robust)
- **NEW** Local Binary Patterns (LBP)
- **NEW** Shape descriptors (Hu moments)

Author: That Le
Date: 2025-01-21
Updated: 2025-01-XX - Added grayscale-robust features
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
    
    # Grayscale-robust features (v2.0)
    use_texture_features: bool = Field(default=True, description="Use LBP and GLCM features")
    use_shape_features: bool = Field(default=True, description="Use Hu moments")
    lbp_radius: int = Field(default=3, ge=1, description="LBP radius")
    lbp_points: int = Field(default=24, ge=8, description="LBP neighbor points")


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
        """Compute image-level features including grayscale-robust features."""
        h, w = image.shape[:2]
        total_pixels = h * w
        
        # Convert color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect if image is grayscale
        is_grayscale = self._is_grayscale(image)
        
        # 1. Edge orientation analysis
        edges = cv2.Canny(gray, self.config.edge_canny_low, self.config.edge_canny_high)
        h_ratio, v_ratio, d_ratio = self._analyze_edge_orientation(edges)
        
        # 2. Color analysis (reduced weight for grayscale)
        n_colors, color_coverage = self._analyze_colors(hsv)
        
        # 3. Circular structure detection
        circularity, n_circles = self._detect_circles(gray, image.shape[:2])
        
        # 4. Grid pattern (common in line charts)
        grid_score = self._detect_grid_pattern(edges)
        
        # 5. Small blob detection (markers in scatter plots)
        n_markers = self._count_small_blobs(gray)
        
        # 6. Rectangular region detection
        n_rects, rect_coverage = self._detect_rectangles(gray)
        
        # === NEW: Grayscale-robust features (v2.0) ===
        
        # 7. Texture features (LBP-based)
        texture_uniformity, texture_contrast = 0.0, 0.0
        if self.config.use_texture_features:
            texture_uniformity, texture_contrast = self._compute_texture_features(gray)
        
        # 8. Shape features (Hu moments)
        hu_elongation, hu_compactness = 0.0, 0.0
        if self.config.use_shape_features:
            hu_elongation, hu_compactness = self._compute_shape_features(gray)
        
        # 9. Gradient histogram (grayscale-robust edge direction)
        grad_h_ratio, grad_v_ratio = self._compute_gradient_histogram(gray)
        
        # 10. Connected component statistics
        n_components, avg_component_area = self._analyze_connected_components(gray)
        
        # 11. Axis line detection (for line/bar charts)
        has_x_axis, has_y_axis = self._detect_axes(gray, edges)
        
        # 12. Symmetry score (pie charts are often symmetric)
        symmetry_score = self._compute_symmetry(gray)
        
        return {
            # Original features
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
            # New grayscale-robust features
            "texture_uniformity": texture_uniformity,
            "texture_contrast": texture_contrast,
            "hu_elongation": hu_elongation,
            "hu_compactness": hu_compactness,
            "grad_h_ratio": grad_h_ratio,
            "grad_v_ratio": grad_v_ratio,
            "n_components": float(n_components),
            "avg_component_area": avg_component_area,
            "has_x_axis": float(has_x_axis),
            "has_y_axis": float(has_y_axis),
            "symmetry_score": symmetry_score,
            "is_grayscale": float(is_grayscale),
        }
    
    # ========== NEW: Grayscale-robust Feature Methods ==========
    
    def _is_grayscale(self, image: np.ndarray) -> bool:
        """Check if image is effectively grayscale."""
        if len(image.shape) == 2:
            return True
        
        # Check if RGB channels are nearly identical
        b, g, r = cv2.split(image)
        diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
        diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))
        
        return diff_rg < 10 and diff_rb < 10
    
    def _compute_texture_features(
        self,
        gray: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute texture features using Local Binary Patterns (LBP).
        
        These features work on grayscale images and capture local texture patterns.
        
        Returns:
            Tuple of (uniformity, contrast)
        """
        h, w = gray.shape
        
        # Simple LBP implementation
        radius = self.config.lbp_radius
        padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius, cv2.BORDER_REFLECT)
        
        # Sample 8 neighbors at radius distance
        center = padded[radius:-radius, radius:-radius]
        
        # Compare with neighbors
        lbp = np.zeros_like(center, dtype=np.uint8)
        
        # 8 neighbor positions
        for i, (dy, dx) in enumerate([
            (-radius, 0), (-radius, radius), (0, radius), (radius, radius),
            (radius, 0), (radius, -radius), (0, -radius), (-radius, -radius)
        ]):
            neighbor = padded[radius+dy:h+radius+dy, radius+dx:w+radius+dx]
            lbp += (neighbor >= center).astype(np.uint8) * (1 << i)
        
        # Compute LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        
        # Uniformity: sum of squared histogram values (higher = more uniform texture)
        uniformity = np.sum(hist ** 2)
        
        # Contrast: weighted variance based on LBP values
        bin_centers = np.arange(256)
        mean_val = np.sum(hist * bin_centers)
        contrast = np.sqrt(np.sum(hist * (bin_centers - mean_val) ** 2))
        contrast = contrast / 128  # Normalize to ~[0, 1]
        
        return uniformity, min(1.0, contrast)
    
    def _compute_shape_features(
        self,
        gray: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute shape features using Hu moments.
        
        Returns:
            Tuple of (elongation, compactness)
        """
        # Threshold to get binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find largest contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, 0.0
        
        # Use moments from the entire image
        moments = cv2.moments(binary)
        
        if moments["m00"] == 0:
            return 0.0, 0.0
        
        # Compute Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform (Hu moments can vary by many orders of magnitude)
        hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # Elongation: ratio of principal axes (from central moments)
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        
        # Eigenvalues of covariance matrix
        delta = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
        lambda1 = (mu20 + mu02 + delta) / 2
        lambda2 = (mu20 + mu02 - delta) / 2
        
        if lambda2 > 0:
            elongation = np.sqrt(lambda1 / lambda2)
            elongation = min(10.0, elongation) / 10  # Normalize
        else:
            elongation = 1.0
        
        # Compactness: 4*pi*area / perimeter^2
        # For largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter > 0:
            compactness = 4 * math.pi * area / (perimeter ** 2)
        else:
            compactness = 0.0
        
        return elongation, min(1.0, compactness)
    
    def _compute_gradient_histogram(
        self,
        gray: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute gradient direction histogram.
        
        More robust than Canny edges for grayscale analysis.
        
        Returns:
            Tuple of (horizontal_ratio, vertical_ratio)
        """
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi  # -180 to 180
        
        # Only consider significant gradients
        mag_threshold = np.percentile(magnitude, 90)
        mask = magnitude > mag_threshold
        
        if np.sum(mask) < 10:
            return 0.33, 0.33
        
        significant_dirs = direction[mask]
        
        # Count horizontal (near 0 or 180), vertical (near 90 or -90)
        h_count = np.sum(
            (np.abs(significant_dirs) < 20) | 
            (np.abs(significant_dirs) > 160)
        )
        v_count = np.sum(
            (np.abs(significant_dirs - 90) < 20) | 
            (np.abs(significant_dirs + 90) < 20)
        )
        
        total = len(significant_dirs)
        
        return h_count / total, v_count / total
    
    def _analyze_connected_components(
        self,
        gray: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Analyze connected components in thresholded image.
        
        Returns:
            Tuple of (num_components, avg_area_ratio)
        """
        h, w = gray.shape
        total_area = h * w
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Exclude background (label 0)
        if num_labels <= 1:
            return 0, 0.0
        
        areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        
        # Filter tiny components (noise)
        significant_areas = areas[areas > 50]
        
        n_components = len(significant_areas)
        avg_area = np.mean(significant_areas) if n_components > 0 else 0
        
        return n_components, avg_area / total_area
    
    def _detect_axes(
        self,
        gray: np.ndarray,
        edges: np.ndarray,
    ) -> Tuple[bool, bool]:
        """
        Detect presence of X and Y axes.
        
        Uses Hough line detection to find long horizontal/vertical lines
        near image edges (typical axis locations).
        
        Returns:
            Tuple of (has_x_axis, has_y_axis)
        """
        h, w = edges.shape
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=w // 4,
            maxLineGap=10
        )
        
        has_x_axis = False
        has_y_axis = False
        
        if lines is None:
            return False, False
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if horizontal (X axis)
            if abs(y2 - y1) < 10 and abs(x2 - x1) > w // 4:
                # X axis typically in bottom half
                if y1 > h // 2:
                    has_x_axis = True
            
            # Check if vertical (Y axis)
            if abs(x2 - x1) < 10 and abs(y2 - y1) > h // 4:
                # Y axis typically in left quarter
                if x1 < w // 3:
                    has_y_axis = True
        
        return has_x_axis, has_y_axis
    
    def _compute_symmetry(
        self,
        gray: np.ndarray,
    ) -> float:
        """
        Compute radial symmetry score.
        
        Pie charts typically have high radial symmetry.
        
        Returns:
            Symmetry score (0-1)
        """
        h, w = gray.shape
        
        # Crop to center square for symmetry analysis
        size = min(h, w)
        y_off = (h - size) // 2
        x_off = (w - size) // 2
        center_crop = gray[y_off:y_off+size, x_off:x_off+size]
        
        # Compare with 180-degree rotation
        rotated_180 = cv2.rotate(center_crop, cv2.ROTATE_180)
        diff_180 = np.mean(np.abs(center_crop.astype(float) - rotated_180.astype(float)))
        
        # Compare with 90-degree rotation (4-fold symmetry)
        rotated_90 = cv2.rotate(center_crop, cv2.ROTATE_90_CLOCKWISE)
        diff_90 = np.mean(np.abs(center_crop.astype(float) - rotated_90.astype(float)))
        
        # Normalize (lower diff = higher symmetry)
        symmetry_180 = 1 - min(1.0, diff_180 / 128)
        symmetry_90 = 1 - min(1.0, diff_90 / 128)
        
        return (symmetry_180 + symmetry_90) / 2
    
    # ========== End Grayscale-robust Features ==========
    
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
        """Score likelihood of pie chart (v2.0 with grayscale support)."""
        score = 0.0
        is_grayscale = features.get("is_grayscale", 0) > 0.5
        
        # Strong indicator: circular structure
        score += 0.4 * features["circularity"]
        
        # Symmetry (strong for pie charts)
        score += 0.2 * features.get("symmetry_score", 0)
        
        # Shape compactness (pie charts are compact)
        score += 0.15 * features.get("hu_compactness", 0)
        
        if not is_grayscale:
            # Multiple colors typical
            if features["n_colors"] >= 3:
                score += 0.15
            # High color coverage
            score += 0.1 * features["color_coverage"]
        else:
            # For grayscale: rely more on texture uniformity
            # Pie slices often have distinct uniform textures
            score += 0.15 * (1 - features.get("texture_uniformity", 0.5))
        
        # Penalty for grid pattern (rare in pie charts)
        score -= 0.2 * features["grid_score"]
        
        # Penalty for many rectangles
        if features["n_rectangles"] > 3:
            score -= 0.2
        
        # Penalty for axis presence (pie charts don't have axes)
        if features.get("has_x_axis", 0) or features.get("has_y_axis", 0):
            score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _score_bar(self, features: Dict[str, float]) -> float:
        """Score likelihood of bar chart (v2.0 with grayscale support)."""
        score = 0.0
        is_grayscale = features.get("is_grayscale", 0) > 0.5
        
        # Strong indicator: vertical or horizontal edges dominant
        edge_ratio = max(features["v_edge_ratio"], features["h_edge_ratio"])
        score += 0.25 * edge_ratio * 3  # Amplify
        
        # Gradient-based edge direction (grayscale-robust)
        grad_ratio = max(features.get("grad_h_ratio", 0), features.get("grad_v_ratio", 0))
        score += 0.15 * grad_ratio * 2
        
        # Rectangles detected
        if features["n_rectangles"] >= 2:
            score += 0.25
        score += 0.15 * min(1.0, features["rect_coverage"] * 5)
        
        # Connected components (bar charts have distinct bars)
        n_comp = features.get("n_components", 0)
        if 2 <= n_comp <= 15:
            score += 0.1
        
        if not is_grayscale:
            # Multiple colors
            if features["n_colors"] >= 2:
                score += 0.1
        
        # Axis presence (bar charts typically have axes)
        if features.get("has_x_axis", 0) or features.get("has_y_axis", 0):
            score += 0.1
        
        # Penalty for circular structure
        score -= 0.3 * features["circularity"]
        
        # Penalty for many markers (likely scatter)
        if features["n_markers"] > 20:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_line(self, features: Dict[str, float]) -> float:
        """Score likelihood of line chart (v2.0 with grayscale support)."""
        score = 0.0
        is_grayscale = features.get("is_grayscale", 0) > 0.5
        
        # Grid pattern common in line charts
        score += 0.25 * features["grid_score"]
        
        # Diagonal edges (lines are often diagonal)
        score += 0.15 * features["d_edge_ratio"] * 5
        
        # Some markers (data points)
        if 3 <= features["n_markers"] <= 30:
            score += 0.2
        
        # Axis presence (line charts typically have axes)
        if features.get("has_x_axis", 0) and features.get("has_y_axis", 0):
            score += 0.15
        elif features.get("has_x_axis", 0) or features.get("has_y_axis", 0):
            score += 0.08
        
        # Shape elongation (line charts have elongated patterns)
        score += 0.1 * features.get("hu_elongation", 0)
        
        if not is_grayscale:
            # Few distinct colors
            if features["n_colors"] <= 4:
                score += 0.1
        
        # Penalty for circular structure
        score -= 0.3 * features["circularity"]
        
        # Penalty for many rectangles
        if features["n_rectangles"] > 5:
            score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _score_scatter(self, features: Dict[str, float]) -> float:
        """Score likelihood of scatter plot (v2.0 with grayscale support)."""
        score = 0.0
        is_grayscale = features.get("is_grayscale", 0) > 0.5
        
        # Strong indicator: many small markers
        if features["n_markers"] >= self.config.scatter_marker_threshold:
            score += 0.35 * min(1.0, features["n_markers"] / 50)
        
        # Grid pattern common
        score += 0.15 * features["grid_score"]
        
        # Axis presence (scatter plots typically have axes)
        if features.get("has_x_axis", 0) and features.get("has_y_axis", 0):
            score += 0.15
        
        # Many connected components (individual points)
        n_comp = features.get("n_components", 0)
        if n_comp > 10:
            score += 0.15
        
        # Small average component area (points are small)
        avg_area = features.get("avg_component_area", 0)
        if 0 < avg_area < 0.01:
            score += 0.1
        
        # Few rectangles
        if features["n_rectangles"] <= 2:
            score += 0.1
        
        # Penalty for circular structure
        score -= 0.3 * features["circularity"]
        
        # Penalty for dominant edge orientation (scatter is more random)
        max_edge = max(features["h_edge_ratio"], features["v_edge_ratio"])
        if max_edge > 0.5:
            score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def _generate_reasoning(
        self,
        features: Dict[str, float],
        scores: Dict[ChartType, float],
        best_type: ChartType,
    ) -> str:
        """Generate human-readable reasoning (v2.0 with grayscale info)."""
        reasons = []
        is_grayscale = features.get("is_grayscale", 0) > 0.5
        
        # Note if grayscale
        if is_grayscale:
            reasons.append("Grayscale image (using texture/shape features)")
        
        if best_type == ChartType.PIE:
            if features["circularity"] > 0.3:
                reasons.append(f"Strong circular structure ({features['circularity']:.2f})")
            if features.get("symmetry_score", 0) > 0.5:
                reasons.append(f"High symmetry ({features.get('symmetry_score', 0):.2f})")
            if not is_grayscale and features["n_colors"] >= 3:
                reasons.append(f"Multiple colors ({int(features['n_colors'])})")
            if features.get("hu_compactness", 0) > 0.5:
                reasons.append(f"Compact shape ({features.get('hu_compactness', 0):.2f})")
        
        elif best_type == ChartType.BAR:
            if features["n_rectangles"] >= 2:
                reasons.append(f"Rectangles detected ({int(features['n_rectangles'])})")
            edge_ratio = max(features["v_edge_ratio"], features["h_edge_ratio"])
            if edge_ratio > 0.3:
                reasons.append(f"Strong H/V edges ({edge_ratio:.2f})")
            grad_ratio = max(features.get("grad_h_ratio", 0), features.get("grad_v_ratio", 0))
            if grad_ratio > 0.3:
                reasons.append(f"Gradient direction ({grad_ratio:.2f})")
            if features.get("has_x_axis", 0) or features.get("has_y_axis", 0):
                reasons.append("Axes detected")
        
        elif best_type == ChartType.LINE:
            if features["grid_score"] > 0.3:
                reasons.append(f"Grid pattern ({features['grid_score']:.2f})")
            if features["d_edge_ratio"] > 0.1:
                reasons.append(f"Diagonal edges ({features['d_edge_ratio']:.2f})")
            if features.get("has_x_axis", 0) and features.get("has_y_axis", 0):
                reasons.append("Both axes detected")
            if features.get("hu_elongation", 0) > 0.5:
                reasons.append(f"Elongated pattern ({features.get('hu_elongation', 0):.2f})")
        
        elif best_type == ChartType.SCATTER:
            if features["n_markers"] >= 10:
                reasons.append(f"Many markers ({int(features['n_markers'])})")
            n_comp = features.get("n_components", 0)
            if n_comp > 10:
                reasons.append(f"Many components ({int(n_comp)})")
            if features.get("has_x_axis", 0) and features.get("has_y_axis", 0):
                reasons.append("Both axes detected")
        
        if not reasons:
            reasons.append("Classification based on combined feature scores")
        
        return f"Classified as {best_type.value}: " + "; ".join(reasons)
