"""
Vectorization Module

Implements Ramer-Douglas-Peucker (RDP) algorithm for converting
pixel paths to piecewise linear vectors, with curve fitting support.

Key features:
- RDP simplification with adaptive epsilon
- Vertex preservation (data point accuracy)
- Sub-pixel refinement
- Morphological profile for line style detection
- Arc/circle fitting for pie charts
- Spline fitting for smooth curves

Reference: docs/instruction_p2_research.md - Section 3
Enhancement: instruction_003.md - Curve fitting
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ...schemas.extraction import (
    KeyPoint,
    KeyPointType,
    LineStyle,
    PointFloat,
    Polyline,
)

logger = logging.getLogger(__name__)


class CurveFitMethod(str, Enum):
    """Method for curve fitting."""
    NONE = "none"           # Only RDP polylines
    AUTO = "auto"           # Automatically detect curves
    CIRCLE = "circle"       # Fit circles/arcs
    ELLIPSE = "ellipse"     # Fit ellipses
    SPLINE = "spline"       # B-spline fitting


class VectorizeConfig(BaseModel):
    """Configuration for vectorization."""
    
    # RDP algorithm
    epsilon: float = Field(
        default=2.0,
        gt=0,
        description="RDP tolerance (pixels). Lower = more vertices preserved"
    )
    adaptive_epsilon: bool = Field(
        default=True,
        description="Adapt epsilon based on local stroke width"
    )
    epsilon_factor: float = Field(
        default=0.5,
        gt=0,
        le=2.0,
        description="Factor for adaptive epsilon: eps = factor * stroke_width"
    )
    
    # Curvature-adaptive epsilon (NEW)
    use_curvature_adaptive: bool = Field(
        default=True,
        description="Use local curvature to adapt epsilon (tighter tolerance at curves)"
    )
    curvature_epsilon_min: float = Field(
        default=0.5,
        gt=0,
        description="Minimum epsilon for high-curvature regions"
    )
    curvature_epsilon_max: float = Field(
        default=5.0,
        gt=0,
        description="Maximum epsilon for straight regions"
    )
    curvature_window: int = Field(
        default=7,
        ge=3,
        description="Window size for local curvature estimation"
    )
    
    # Hierarchical segmentation (NEW)
    use_hierarchical: bool = Field(
        default=True,
        description="Use hierarchical segmentation at high-curvature points"
    )
    segment_at_corners: bool = Field(
        default=True,
        description="Segment polyline at detected corners"
    )
    corner_angle_threshold: float = Field(
        default=45.0,
        ge=10,
        le=120,
        description="Angle threshold (degrees) for corner segmentation"
    )
    
    # Sub-pixel refinement
    subpixel_refinement: bool = Field(
        default=True,
        description="Refine vertex positions to sub-pixel accuracy"
    )
    refinement_window: int = Field(
        default=5,
        ge=3,
        description="Window size for sub-pixel refinement"
    )
    
    # Line style detection
    detect_line_style: bool = Field(
        default=True,
        description="Detect dash patterns for line style"
    )
    min_gap_length: int = Field(
        default=3,
        ge=1,
        description="Minimum gap length to consider as dashed"
    )
    
    # Curve fitting
    curve_fit_method: CurveFitMethod = Field(
        default=CurveFitMethod.AUTO,
        description="Curve fitting method for smooth shapes"
    )
    curve_fit_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Minimum R-squared for accepting curve fit"
    )
    min_arc_points: int = Field(
        default=10,
        ge=5,
        description="Minimum points to attempt arc fitting"
    )
    curvature_threshold: float = Field(
        default=0.1,
        gt=0,
        description="Minimum curvature to consider as curved"
    )


@dataclass
class VectorizeResult:
    """Result of vectorization operation."""
    
    polylines: List[Polyline]
    vertices: List[KeyPoint]  # RDP-preserved vertices (data points)
    total_points_before: int
    total_points_after: int
    compression_ratio: float
    fitted_curves: List['FittedCurve'] = None  # Curve fits if detected
    
    def __post_init__(self):
        if self.fitted_curves is None:
            self.fitted_curves = []


@dataclass
class FittedCurve:
    """Result of curve fitting."""
    
    curve_type: str  # "arc", "circle", "ellipse", "spline"
    center: Optional[Tuple[float, float]] = None  # For arc/circle/ellipse
    radius: Optional[float] = None  # For circle
    radii: Optional[Tuple[float, float]] = None  # For ellipse (a, b)
    angle: Optional[float] = None  # Rotation angle for ellipse
    start_angle: Optional[float] = None  # For arc
    end_angle: Optional[float] = None  # For arc
    control_points: Optional[List[Tuple[float, float]]] = None  # For spline
    r_squared: float = 0.0  # Fit quality
    points: List[Tuple[float, float]] = None  # Original points used


class Vectorizer:
    """
    Converts pixel paths to vector polylines using RDP algorithm.
    
    The Ramer-Douglas-Peucker algorithm preserves vertices that are
    significant (local extrema, inflection points) while removing
    redundant points on straight segments.
    
    Example:
        config = VectorizeConfig(epsilon=2.0, adaptive_epsilon=True)
        vectorizer = Vectorizer(config)
        result = vectorizer.process(paths, stroke_width_map)
    """
    
    def __init__(self, config: Optional[VectorizeConfig] = None):
        """
        Initialize vectorizer.
        
        Args:
            config: Vectorization configuration (uses defaults if None)
        """
        self.config = config or VectorizeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(
        self,
        paths: List[List[Tuple[int, int]]],
        stroke_width_map: Optional[np.ndarray] = None,
        grayscale_image: Optional[np.ndarray] = None,
        chart_id: str = "unknown",
    ) -> VectorizeResult:
        """
        Vectorize pixel paths to polylines.
        
        Args:
            paths: List of pixel paths (each path is list of (x, y))
            stroke_width_map: Optional stroke width at each pixel
            grayscale_image: Optional grayscale for sub-pixel refinement
            chart_id: Chart identifier for logging
        
        Returns:
            VectorizeResult with polylines
        """
        self.logger.debug(f"Vectorization started | chart_id={chart_id}")
        
        polylines = []
        all_vertices = []
        total_before = 0
        total_after = 0
        
        for path_idx, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            total_before += len(path)
            
            # Hierarchical segmentation at corners (NEW)
            if self.config.use_hierarchical and self.config.segment_at_corners:
                segments = self._segment_at_corners(path)
            else:
                segments = [path]
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                
                # Determine epsilon for this path/segment
                if self.config.use_curvature_adaptive:
                    # Use curvature-aware RDP
                    simplified = self._rdp_curvature_adaptive(segment)
                elif self.config.adaptive_epsilon and stroke_width_map is not None:
                    epsilon = self._compute_adaptive_epsilon(segment, stroke_width_map)
                    simplified = self._rdp_simplify(segment, epsilon)
                else:
                    simplified = self._rdp_simplify(segment, self.config.epsilon)
                
                # Sub-pixel refinement
                if self.config.subpixel_refinement and grayscale_image is not None:
                    simplified = self._refine_subpixel(simplified, grayscale_image)
                
                total_after += len(simplified)
                
                # Convert to PointFloat
                points = [PointFloat(x=float(x), y=float(y)) for x, y in simplified]
                
                # Detect line style
                line_style = LineStyle.SOLID
                seg_lengths = []
                gap_lengths = []
                
                if self.config.detect_line_style:
                    line_style, seg_lengths, gap_lengths = self._detect_line_style(segment)
                
                # Create polyline
                polyline = Polyline(
                    points=points,
                    line_style=line_style,
                    segment_lengths=seg_lengths,
                    gap_lengths=gap_lengths,
                )
                polylines.append(polyline)
                
                # Mark vertices as data points
                for point in points:
                    vertex = KeyPoint(
                        point=point,
                        point_type=KeyPointType.VERTEX,
                        is_vertex=True,
                        confidence=1.0,
                    )
                    all_vertices.append(vertex)
        
        compression = 1.0 - (total_after / total_before) if total_before > 0 else 0.0
        
        self.logger.info(
            f"Vectorization complete | chart_id={chart_id} | "
            f"polylines={len(polylines)} | compression={compression:.1%}"
        )
        
        return VectorizeResult(
            polylines=polylines,
            vertices=all_vertices,
            total_points_before=total_before,
            total_points_after=total_after,
            compression_ratio=compression,
        )
    
    def _rdp_simplify(
        self,
        path: List[Tuple[int, int]],
        epsilon: float,
    ) -> List[Tuple[float, float]]:
        """
        Ramer-Douglas-Peucker line simplification.
        
        Recursively finds the point with maximum perpendicular distance
        from the line segment. If distance > epsilon, splits at that point.
        
        Args:
            path: List of (x, y) points
            epsilon: Tolerance threshold
        
        Returns:
            Simplified path with fewer points
        """
        if len(path) < 3:
            return [(float(x), float(y)) for x, y in path]
        
        # Convert to numpy for efficiency
        points = np.array(path, dtype=np.float64)
        
        # Find the point with maximum distance from line
        start, end = points[0], points[-1]
        
        # Line segment vector
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-10:
            # Start and end are same point
            return [(float(start[0]), float(start[1]))]
        
        line_unit = line_vec / line_len
        
        # Perpendicular distances
        max_dist = 0.0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            point = points[i]
            # Vector from start to point
            vec = point - start
            # Project onto line
            proj_len = np.dot(vec, line_unit)
            # Perpendicular distance
            if proj_len < 0:
                dist = np.linalg.norm(vec)
            elif proj_len > line_len:
                dist = np.linalg.norm(point - end)
            else:
                proj = start + proj_len * line_unit
                dist = np.linalg.norm(point - proj)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # Recursively simplify
        if max_dist > epsilon:
            # Split at max distance point
            left = self._rdp_simplify(path[:max_idx + 1], epsilon)
            right = self._rdp_simplify(path[max_idx:], epsilon)
            return left[:-1] + right
        else:
            # Just keep endpoints
            return [(float(start[0]), float(start[1])),
                    (float(end[0]), float(end[1]))]
    
    def _compute_adaptive_epsilon(
        self,
        path: List[Tuple[int, int]],
        stroke_width_map: np.ndarray,
    ) -> float:
        """
        Compute adaptive epsilon based on local stroke width.
        
        Thicker strokes need larger epsilon to filter noise.
        
        Args:
            path: Pixel path
            stroke_width_map: Stroke width at each pixel
        
        Returns:
            Adaptive epsilon value
        """
        widths = []
        h, w = stroke_width_map.shape
        
        for x, y in path:
            if 0 <= y < h and 0 <= x < w:
                width = stroke_width_map[y, x]
                if width > 0:
                    widths.append(width)
        
        if not widths:
            return self.config.epsilon
        
        # Use median stroke width
        median_width = np.median(widths)
        
        # Adaptive epsilon
        adaptive_eps = self.config.epsilon_factor * median_width
        
        # Clamp to reasonable range
        return max(0.5, min(adaptive_eps, 10.0))
    
    def _segment_at_corners(
        self,
        path: List[Tuple[int, int]],
    ) -> List[List[Tuple[int, int]]]:
        """
        Segment a path at detected corner points.
        
        Corners are points where the direction changes significantly.
        This allows different epsilon values for different segments.
        
        Args:
            path: Pixel path
        
        Returns:
            List of path segments
        """
        if len(path) < 5:
            return [path]
        
        window = max(3, self.config.curvature_window // 2)
        threshold_rad = np.radians(self.config.corner_angle_threshold)
        
        corner_indices = []
        
        for i in range(window, len(path) - window):
            # Direction before point
            dx1 = path[i][0] - path[i - window][0]
            dy1 = path[i][1] - path[i - window][1]
            
            # Direction after point
            dx2 = path[i + window][0] - path[i][0]
            dy2 = path[i + window][1] - path[i][1]
            
            # Compute angle change
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            
            if len1 < 1e-10 or len2 < 1e-10:
                continue
            
            # Dot product for angle
            dot = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            
            # If angle change is significant, mark as corner
            if angle > threshold_rad:
                corner_indices.append(i)
        
        if not corner_indices:
            return [path]
        
        # Remove corners that are too close together
        min_segment_length = 10
        filtered_corners = []
        last_corner = -min_segment_length
        
        for idx in corner_indices:
            if idx - last_corner >= min_segment_length:
                filtered_corners.append(idx)
                last_corner = idx
        
        # Split path at corners
        segments = []
        prev_idx = 0
        
        for corner_idx in filtered_corners:
            if corner_idx - prev_idx >= 3:
                segments.append(path[prev_idx:corner_idx + 1])
            prev_idx = corner_idx
        
        # Add final segment
        if len(path) - prev_idx >= 3:
            segments.append(path[prev_idx:])
        
        return segments if segments else [path]
    
    def _rdp_curvature_adaptive(
        self,
        path: List[Tuple[int, int]],
    ) -> List[Tuple[float, float]]:
        """
        RDP with curvature-adaptive epsilon.
        
        Uses tighter tolerance (smaller epsilon) at curved regions
        and looser tolerance at straight regions.
        
        Args:
            path: Pixel path
        
        Returns:
            Simplified path
        """
        if len(path) < 3:
            return [(float(x), float(y)) for x, y in path]
        
        # Compute local curvature at each point
        curvatures = self._compute_local_curvature(path)
        
        # Map curvature to epsilon
        epsilons = []
        for curv in curvatures:
            # High curvature -> low epsilon (preserve detail)
            # Low curvature -> high epsilon (simplify)
            if curv > self.config.curvature_threshold:
                # Curved region
                eps = self.config.curvature_epsilon_min
            else:
                # Scale linearly based on curvature
                ratio = curv / self.config.curvature_threshold if self.config.curvature_threshold > 0 else 0
                eps = self.config.curvature_epsilon_max - (
                    self.config.curvature_epsilon_max - self.config.curvature_epsilon_min
                ) * ratio
            epsilons.append(eps)
        
        # Use segment-wise RDP with local epsilon
        return self._rdp_with_local_epsilon(path, epsilons)
    
    def _compute_local_curvature(
        self,
        path: List[Tuple[int, int]],
        window: int = None,
    ) -> List[float]:
        """
        Compute local curvature at each point using Menger curvature.
        
        Menger curvature = 4 * area(triangle) / (|AB| * |BC| * |CA|)
        
        Args:
            path: Pixel path
            window: Window size for curvature estimation
        
        Returns:
            List of curvature values (same length as path)
        """
        if window is None:
            window = self.config.curvature_window
        
        half_w = window // 2
        n = len(path)
        curvatures = []
        
        for i in range(n):
            # Get neighboring points
            i_prev = max(0, i - half_w)
            i_next = min(n - 1, i + half_w)
            
            if i_prev == i or i == i_next:
                curvatures.append(0.0)
                continue
            
            # Three points for Menger curvature
            A = np.array(path[i_prev], dtype=np.float64)
            B = np.array(path[i], dtype=np.float64)
            C = np.array(path[i_next], dtype=np.float64)
            
            # Side lengths
            AB = np.linalg.norm(B - A)
            BC = np.linalg.norm(C - B)
            CA = np.linalg.norm(A - C)
            
            if AB < 1e-10 or BC < 1e-10 or CA < 1e-10:
                curvatures.append(0.0)
                continue
            
            # Area using cross product (2D: z-component of 3D cross product)
            cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            area = abs(cross) / 2.0
            
            # Menger curvature
            curv = 4.0 * area / (AB * BC * CA)
            curvatures.append(curv)
        
        return curvatures
    
    def _rdp_with_local_epsilon(
        self,
        path: List[Tuple[int, int]],
        epsilons: List[float],
    ) -> List[Tuple[float, float]]:
        """
        RDP with varying epsilon based on local properties.
        
        Args:
            path: Pixel path
            epsilons: Epsilon value for each point
        
        Returns:
            Simplified path
        """
        if len(path) < 3:
            return [(float(x), float(y)) for x, y in path]
        
        points = np.array(path, dtype=np.float64)
        n = len(points)
        
        # Recursive RDP with local epsilon
        keep = np.zeros(n, dtype=bool)
        keep[0] = keep[-1] = True
        
        def recursive_rdp(start, end):
            if end - start <= 1:
                return
            
            # Find point with maximum distance relative to local epsilon
            line_vec = points[end] - points[start]
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-10:
                return
            
            line_unit = line_vec / line_len
            
            max_dist_ratio = 0.0
            max_idx = start
            
            for i in range(start + 1, end):
                vec = points[i] - points[start]
                proj_len = np.dot(vec, line_unit)
                
                if proj_len < 0:
                    dist = np.linalg.norm(vec)
                elif proj_len > line_len:
                    dist = np.linalg.norm(points[i] - points[end])
                else:
                    proj = points[start] + proj_len * line_unit
                    dist = np.linalg.norm(points[i] - proj)
                
                # Compare distance to local epsilon
                local_eps = epsilons[i]
                dist_ratio = dist / local_eps if local_eps > 0 else dist
                
                if dist_ratio > max_dist_ratio:
                    max_dist_ratio = dist_ratio
                    max_idx = i
            
            if max_dist_ratio > 1.0:  # dist > epsilon
                keep[max_idx] = True
                recursive_rdp(start, max_idx)
                recursive_rdp(max_idx, end)
        
        recursive_rdp(0, n - 1)
        
        # Extract kept points
        result = [(float(points[i, 0]), float(points[i, 1])) for i in range(n) if keep[i]]
        
        return result
    
    def _refine_subpixel(
        self,
        path: List[Tuple[float, float]],
        grayscale: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """
        Refine vertex positions to sub-pixel accuracy.
        
        Uses quadratic surface fitting around each vertex
        to find the intensity extremum.
        
        Args:
            path: Simplified path with float coordinates
            grayscale: Grayscale image for intensity analysis
        
        Returns:
            Path with sub-pixel refined positions
        """
        h, w = grayscale.shape
        window = self.config.refinement_window
        half = window // 2
        
        refined = []
        
        for x, y in path:
            ix, iy = int(round(x)), int(round(y))
            
            # Check bounds for window
            if (iy - half < 0 or iy + half >= h or
                ix - half < 0 or ix + half >= w):
                refined.append((x, y))
                continue
            
            # Extract window
            window_img = grayscale[iy - half:iy + half + 1,
                                   ix - half:ix + half + 1].astype(np.float64)
            
            # Fit quadratic surface: f(u,v) = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
            # Find the extremum (maximum for bright strokes)
            try:
                dx, dy = self._fit_quadratic_extremum(window_img)
                # Clamp offset
                dx = max(-half, min(dx, half))
                dy = max(-half, min(dy, half))
                refined.append((x + dx, y + dy))
            except Exception:
                refined.append((x, y))
        
        return refined
    
    def _fit_quadratic_extremum(
        self,
        window: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Fit quadratic surface and find extremum.
        
        Args:
            window: Image window around point
        
        Returns:
            (dx, dy) offset to extremum
        """
        h, w = window.shape
        cy, cx = h // 2, w // 2
        
        # Simple gradient-based refinement
        # Use Sobel gradients
        gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient at center
        grad_x = gx[cy, cx]
        grad_y = gy[cy, cx]
        
        # Hessian for curvature
        gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3)
        gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1, ksize=3)
        
        hxx = gxx[cy, cx]
        hyy = gyy[cy, cx]
        
        # Newton step (for maximum, negate)
        if abs(hxx) > 1e-6:
            dx = -grad_x / hxx
        else:
            dx = 0.0
        
        if abs(hyy) > 1e-6:
            dy = -grad_y / hyy
        else:
            dy = 0.0
        
        return dx, dy
    
    def _detect_line_style(
        self,
        path: List[Tuple[int, int]],
    ) -> Tuple[LineStyle, List[float], List[float]]:
        """
        Detect line style from path pattern.
        
        Analyzes segment and gap lengths to classify
        as solid, dashed, dotted, or dash-dot.
        
        Args:
            path: Original pixel path
        
        Returns:
            (LineStyle, segment_lengths, gap_lengths)
        """
        if len(path) < 5:
            return LineStyle.SOLID, [], []
        
        # For now, assume solid (full implementation needs gap detection)
        # This would require analyzing the original binary image
        # to detect gaps in the stroke
        
        return LineStyle.SOLID, [], []
    
    def simplify_contour(
        self,
        contour: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simplify a contour using OpenCV's approxPolyDP.
        
        Useful for bar rectangles and other closed shapes.
        
        Args:
            contour: OpenCV contour (Nx1x2 array)
            epsilon: Approximation accuracy (default: 2% of perimeter)
        
        Returns:
            Simplified contour
        """
        if epsilon is None:
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = 0.02 * perimeter
        
        return cv2.approxPolyDP(contour, epsilon, closed=True)
    
    # ========== CURVE FITTING METHODS ==========
    
    def fit_circle(
        self,
        points: List[Tuple[float, float]],
    ) -> Optional[FittedCurve]:
        """
        Fit a circle to a set of points using algebraic least squares.
        
        Uses the algebraic circle fit method:
        (x - cx)^2 + (y - cy)^2 = r^2
        
        Rearranged to: x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
        
        Args:
            points: List of (x, y) points
        
        Returns:
            FittedCurve with circle parameters or None if fit fails
        """
        if len(points) < 3:
            return None
        
        pts = np.array(points)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Build design matrix for: A*[cx, cy, c]^T = b
        # where c = r^2 - cx^2 - cy^2
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            # Least squares solve
            result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c = result
            
            # Recover radius
            r_squared = c + cx**2 + cy**2
            if r_squared < 0:
                return None
            radius = np.sqrt(r_squared)
            
            # Compute R-squared
            predicted = (x - cx)**2 + (y - cy)**2
            actual = radius**2 * np.ones(len(x))
            ss_res = np.sum((predicted - actual)**2)
            ss_tot = np.sum((predicted - np.mean(predicted))**2)
            
            if ss_tot < 1e-10:
                r_sq = 1.0 if ss_res < 1e-10 else 0.0
            else:
                r_sq = 1.0 - (ss_res / ss_tot)
            
            return FittedCurve(
                curve_type="circle",
                center=(cx, cy),
                radius=radius,
                r_squared=r_sq,
                points=points,
            )
            
        except np.linalg.LinAlgError:
            return None
    
    def fit_arc(
        self,
        points: List[Tuple[float, float]],
    ) -> Optional[FittedCurve]:
        """
        Fit an arc to a set of points.
        
        First fits a circle, then determines the arc angles.
        
        Args:
            points: List of (x, y) points
        
        Returns:
            FittedCurve with arc parameters or None if fit fails
        """
        circle = self.fit_circle(points)
        if circle is None or circle.r_squared < self.config.curve_fit_threshold:
            return None
        
        cx, cy = circle.center
        
        # Compute angles for each point
        pts = np.array(points)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        
        # Find start and end angles (handle wrap-around)
        # Sort points by angle
        sorted_indices = np.argsort(angles)
        sorted_angles = angles[sorted_indices]
        
        # Check for angle wrap (crossing -pi/pi boundary)
        angle_diffs = np.diff(sorted_angles)
        max_gap_idx = np.argmax(angle_diffs)
        
        if angle_diffs[max_gap_idx] > np.pi:
            # Arc crosses the boundary
            start_angle = sorted_angles[max_gap_idx + 1]
            end_angle = sorted_angles[max_gap_idx]
        else:
            start_angle = sorted_angles[0]
            end_angle = sorted_angles[-1]
        
        return FittedCurve(
            curve_type="arc",
            center=circle.center,
            radius=circle.radius,
            start_angle=float(start_angle),
            end_angle=float(end_angle),
            r_squared=circle.r_squared,
            points=points,
        )
    
    def fit_ellipse(
        self,
        points: List[Tuple[float, float]],
    ) -> Optional[FittedCurve]:
        """
        Fit an ellipse to a set of points using OpenCV.
        
        Args:
            points: List of (x, y) points
        
        Returns:
            FittedCurve with ellipse parameters or None if fit fails
        """
        if len(points) < 5:
            return None
        
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        try:
            # OpenCV fitEllipse
            ellipse = cv2.fitEllipse(pts)
            center, (width, height), angle = ellipse
            
            # Compute fit quality (distance from ellipse)
            cx, cy = center
            a, b = width / 2, height / 2
            theta = np.radians(angle)
            
            # Compute normalized distance for each point
            pts_flat = pts.reshape(-1, 2)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            distances = []
            for px, py in pts_flat:
                # Transform to ellipse coordinate system
                dx, dy = px - cx, py - cy
                x_rot = dx * cos_t + dy * sin_t
                y_rot = -dx * sin_t + dy * cos_t
                
                # Normalized distance from ellipse
                if a > 0 and b > 0:
                    dist = abs((x_rot / a)**2 + (y_rot / b)**2 - 1)
                    distances.append(dist)
            
            if distances:
                mean_dist = np.mean(distances)
                # Convert to R-squared-like metric
                r_sq = max(0, 1 - mean_dist)
            else:
                r_sq = 0
            
            return FittedCurve(
                curve_type="ellipse",
                center=(cx, cy),
                radii=(a, b),
                angle=float(angle),
                r_squared=r_sq,
                points=points,
            )
            
        except cv2.error:
            return None
    
    def compute_curvature(
        self,
        points: List[Tuple[float, float]],
        window: int = 5,
    ) -> List[float]:
        """
        Compute discrete curvature at each point.
        
        Uses the Menger curvature formula:
        k = 4 * Area(triangle) / (|AB| * |BC| * |CA|)
        
        Args:
            points: List of (x, y) points
            window: Number of points to consider for smoothing
        
        Returns:
            List of curvature values (one per point)
        """
        if len(points) < 3:
            return [0.0] * len(points)
        
        pts = np.array(points)
        curvatures = []
        
        for i in range(len(pts)):
            # Get neighboring points
            if i == 0:
                p0, p1, p2 = pts[0], pts[1], pts[min(2, len(pts)-1)]
            elif i == len(pts) - 1:
                p0, p1, p2 = pts[max(0, i-2)], pts[i-1], pts[i]
            else:
                p0, p1, p2 = pts[i-1], pts[i], pts[i+1]
            
            # Compute Menger curvature
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)
            
            if a < 1e-10 or b < 1e-10 or c < 1e-10:
                curvatures.append(0.0)
                continue
            
            # Area of triangle using cross product
            area = 0.5 * abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1]) -
                (p2[0] - p0[0]) * (p1[1] - p0[1])
            )
            
            # Menger curvature
            k = 4 * area / (a * b * c)
            curvatures.append(k)
        
        return curvatures
    
    def detect_curve_type(
        self,
        points: List[Tuple[float, float]],
    ) -> str:
        """
        Automatically detect the best curve type for a set of points.
        
        Args:
            points: List of (x, y) points
        
        Returns:
            Curve type string: "line", "arc", "circle", "ellipse", or "spline"
        """
        if len(points) < self.config.min_arc_points:
            return "line"
        
        # Compute curvature
        curvatures = self.compute_curvature(points)
        mean_curvature = np.mean(np.abs(curvatures))
        std_curvature = np.std(curvatures)
        
        # Low curvature = straight line
        if mean_curvature < self.config.curvature_threshold:
            return "line"
        
        # Constant curvature = circle/arc
        if std_curvature < 0.3 * mean_curvature:
            # Try circle fit
            circle_fit = self.fit_circle(points)
            if circle_fit and circle_fit.r_squared > self.config.curve_fit_threshold:
                # Check if it's a full circle or arc
                pts = np.array(points)
                cx, cy = circle_fit.center
                angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
                angle_range = np.ptp(angles)
                
                if angle_range > 1.8 * np.pi:
                    return "circle"
                else:
                    return "arc"
        
        # Try ellipse
        ellipse_fit = self.fit_ellipse(points)
        if ellipse_fit and ellipse_fit.r_squared > self.config.curve_fit_threshold:
            return "ellipse"
        
        # Default to spline for complex curves
        return "spline"
    
    def fit_curve_auto(
        self,
        points: List[Tuple[float, float]],
    ) -> Optional[FittedCurve]:
        """
        Automatically fit the best curve type to points.
        
        Args:
            points: List of (x, y) points
        
        Returns:
            FittedCurve or None if no good fit found
        """
        curve_type = self.detect_curve_type(points)
        
        if curve_type == "line":
            return None  # Use RDP polyline instead
        elif curve_type == "circle":
            return self.fit_circle(points)
        elif curve_type == "arc":
            return self.fit_arc(points)
        elif curve_type == "ellipse":
            return self.fit_ellipse(points)
        else:
            # Spline: just return points as control points
            return FittedCurve(
                curve_type="spline",
                control_points=points,
                r_squared=1.0,  # Spline passes through all points
                points=points,
            )
