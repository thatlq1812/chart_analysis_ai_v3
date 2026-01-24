"""
Vectorization Module

Implements Ramer-Douglas-Peucker (RDP) algorithm for converting
pixel paths to piecewise linear vectors.

Key features:
- RDP simplification with adaptive epsilon
- Vertex preservation (data point accuracy)
- Sub-pixel refinement
- Morphological profile for line style detection

Reference: docs/instruction_p2_research.md - Section 3
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass
class VectorizeResult:
    """Result of vectorization operation."""
    
    polylines: List[Polyline]
    vertices: List[KeyPoint]  # RDP-preserved vertices (data points)
    total_points_before: int
    total_points_after: int
    compression_ratio: float


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
            
            # Determine epsilon for this path
            if self.config.adaptive_epsilon and stroke_width_map is not None:
                epsilon = self._compute_adaptive_epsilon(path, stroke_width_map)
            else:
                epsilon = self.config.epsilon
            
            # Apply RDP simplification
            simplified = self._rdp_simplify(path, epsilon)
            
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
                line_style, seg_lengths, gap_lengths = self._detect_line_style(path)
            
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
