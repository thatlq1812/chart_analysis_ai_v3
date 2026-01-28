"""
Skeletonization Module

Implements topology-preserving skeletonization using Lee algorithm.
Converts thick strokes to 1-pixel width skeleton while maintaining
connectivity and structural integrity.

Key features:
- Homotopy-preserving thinning (preserves topology)
- Junction detection and classification
- Distance transform for stroke width estimation

Reference: docs/instruction_p2_research.md - Section 2.2
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field
from skimage.morphology import skeletonize, thin

from ...schemas.extraction import KeyPoint, KeyPointType, PointFloat

logger = logging.getLogger(__name__)


class SkeletonConfig(BaseModel):
    """Configuration for skeletonization."""
    
    # Algorithm selection
    method: str = Field(
        default="lee",
        description="Skeletonization method: 'lee', 'zhang', or 'medial'"
    )
    
    # Pre-processing (gap filling)
    fill_gaps: bool = Field(
        default=True,
        description="Apply morphological closing to fill small gaps before skeletonization"
    )
    gap_fill_kernel_size: int = Field(
        default=3,
        ge=1,
        le=7,
        description="Kernel size for gap filling (larger = fills bigger gaps)"
    )
    
    # Post-processing
    remove_spurs: bool = Field(
        default=True,
        description="Remove small spurs (artifacts)"
    )
    spur_length: int = Field(
        default=5,
        ge=1,
        description="Maximum spur length to remove"
    )
    use_improved_spur_removal: bool = Field(
        default=True,
        description="Use improved spur removal that preserves important branches"
    )
    spur_angle_threshold: float = Field(
        default=30.0,
        ge=0,
        le=90,
        description="Angle threshold (degrees) for spur vs branch detection"
    )
    
    # Junction handling
    detect_junctions: bool = Field(
        default=True,
        description="Detect and classify junction points"
    )
    detect_corners: bool = Field(
        default=True,
        description="Detect corner points (high curvature)"
    )
    corner_angle_threshold: float = Field(
        default=45.0,
        ge=10,
        le=90,
        description="Angle threshold (degrees) for corner detection"
    )
    
    # Distance transform
    compute_stroke_width: bool = Field(
        default=True,
        description="Compute stroke width from distance transform"
    )


@dataclass
class SkeletonResult:
    """Result of skeletonization operation."""
    
    skeleton: np.ndarray  # Binary skeleton image
    distance_map: np.ndarray  # Distance transform
    keypoints: List[KeyPoint]  # Detected keypoints
    stroke_width_map: np.ndarray  # Stroke width at each skeleton pixel


class Skeletonizer:
    """
    Topology-preserving skeletonization for chart images.
    
    Uses Lee algorithm (scikit-image) which guarantees that the
    number of connected components and holes remains unchanged.
    
    Example:
        config = SkeletonConfig(method="lee", remove_spurs=True)
        skeletonizer = Skeletonizer(config)
        result = skeletonizer.process(binary_image)
    """
    
    def __init__(self, config: Optional[SkeletonConfig] = None):
        """
        Initialize skeletonizer.
        
        Args:
            config: Skeletonization configuration (uses defaults if None)
        """
        self.config = config or SkeletonConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(
        self,
        binary_image: np.ndarray,
        distance_map: Optional[np.ndarray] = None,
        chart_id: str = "unknown",
    ) -> SkeletonResult:
        """
        Perform skeletonization on binary image.
        
        Args:
            binary_image: Binary image (foreground=255, background=0)
            distance_map: Optional precomputed distance transform
            chart_id: Chart identifier for logging
        
        Returns:
            SkeletonResult with skeleton and metadata
        """
        self.logger.debug(f"Skeletonization started | chart_id={chart_id}")
        
        # Pre-processing: Fill small gaps with morphological closing
        if self.config.fill_gaps:
            kernel_size = self.config.gap_fill_kernel_size
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            self.logger.debug(f"Gap filling applied | kernel_size={kernel_size}")
        
        # Normalize to boolean (scikit-image expects bool)
        binary_bool = binary_image > 127
        
        # Compute distance transform if needed
        if distance_map is None and self.config.compute_stroke_width:
            distance_map = cv2.distanceTransform(
                binary_image, cv2.DIST_L2, 5
            )
        elif distance_map is None:
            distance_map = np.zeros_like(binary_image, dtype=np.float32)
        
        # Perform skeletonization
        if self.config.method == "lee":
            # Lee algorithm (default) - best topology preservation
            skeleton = skeletonize(binary_bool, method="lee")
        elif self.config.method == "zhang":
            # Zhang-Suen algorithm - faster but may create artifacts
            skeleton = skeletonize(binary_bool, method="zhang")
        elif self.config.method == "medial":
            # Medial axis (different approach)
            from skimage.morphology import medial_axis
            skeleton, _ = medial_axis(binary_bool, return_distance=True)
        else:
            # Fallback to scikit-image thin
            skeleton = thin(binary_bool)
        
        # Convert to uint8
        skeleton_uint8 = (skeleton.astype(np.uint8) * 255)
        
        # Remove small spurs if configured
        if self.config.remove_spurs:
            if self.config.use_improved_spur_removal:
                skeleton_uint8 = self._remove_spurs_improved(
                    skeleton_uint8,
                    self.config.spur_length,
                    self.config.spur_angle_threshold,
                )
            else:
                skeleton_uint8 = self._remove_spurs(
                    skeleton_uint8, self.config.spur_length
                )
        
        # Detect keypoints
        keypoints = []
        if self.config.detect_junctions:
            keypoints = self._detect_keypoints(skeleton_uint8)
        
        # Compute stroke width at skeleton pixels
        stroke_width_map = np.zeros_like(skeleton_uint8, dtype=np.float32)
        if self.config.compute_stroke_width:
            stroke_width_map = self._compute_stroke_widths(
                skeleton_uint8, distance_map
            )
        
        self.logger.info(
            f"Skeletonization complete | chart_id={chart_id} | "
            f"keypoints={len(keypoints)}"
        )
        
        return SkeletonResult(
            skeleton=skeleton_uint8,
            distance_map=distance_map,
            keypoints=keypoints,
            stroke_width_map=stroke_width_map,
        )
    
    def _remove_spurs(
        self,
        skeleton: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Remove small spurs (branches) from skeleton.
        
        Spurs are short branches that are artifacts of skeletonization.
        
        Args:
            skeleton: Binary skeleton image
            max_length: Maximum spur length to remove
        
        Returns:
            Pruned skeleton
        """
        result = skeleton.copy()
        
        # Iteratively remove endpoints that create short branches
        for _ in range(max_length):
            # Find endpoints (pixels with exactly 1 neighbor)
            endpoints = self._find_endpoints(result)
            
            if len(endpoints) == 0:
                break
            
            # Remove endpoints
            for y, x in endpoints:
                # Check if removing this creates a spur
                neighbors = self._count_neighbors(result, x, y)
                if neighbors == 1:
                    result[y, x] = 0
        
        return result
    
    def _remove_spurs_improved(
        self,
        skeleton: np.ndarray,
        max_length: int,
        angle_threshold: float = 30.0,
    ) -> np.ndarray:
        """
        Remove small spurs using improved algorithm.
        
        Unlike basic spur removal, this method:
        1. Considers the angle between spur and main branch
        2. Preserves spurs that align with the main direction (likely data points)
        3. Uses branch length relative to total path length
        
        Args:
            skeleton: Binary skeleton image
            max_length: Maximum spur length to remove
            angle_threshold: Angle (degrees) below which spur is considered aligned
        
        Returns:
            Pruned skeleton
        """
        result = skeleton.copy()
        
        for iteration in range(max_length):
            # Find all endpoints
            endpoints = self._find_endpoints(result)
            
            if not endpoints:
                break
            
            removed = 0
            for y, x in endpoints:
                # Trace the branch from this endpoint
                branch = self._trace_branch(result, x, y, max_length + 2)
                
                if len(branch) < 2:
                    continue
                
                # Check if branch is short enough
                if len(branch) > max_length:
                    continue
                
                # Check if this is a genuine spur vs important feature
                if len(branch) >= 3:
                    # Compute branch direction
                    branch_dir = self._compute_direction(branch[:min(5, len(branch))])
                    
                    # Get direction at junction (if exists)
                    junction_x, junction_y = branch[-1]
                    main_dir = self._compute_main_direction(result, junction_x, junction_y, branch)
                    
                    if main_dir is not None:
                        # Compute angle between spur and main branch
                        angle = self._angle_between(branch_dir, main_dir)
                        
                        # If spur is roughly perpendicular, it's likely noise
                        # If roughly aligned, might be important (endpoint of line)
                        if angle < angle_threshold or angle > (180 - angle_threshold):
                            continue  # Keep this spur, it's aligned
                
                # Remove the spur (but keep junction point)
                for bx, by in branch[:-1]:  # Don't remove junction
                    result[by, bx] = 0
                removed += 1
            
            if removed == 0:
                break
        
        return result
    
    def _trace_branch(
        self,
        skeleton: np.ndarray,
        start_x: int,
        start_y: int,
        max_length: int,
    ) -> List[Tuple[int, int]]:
        """Trace a branch from endpoint until junction or max length."""
        h, w = skeleton.shape
        branch = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        
        x, y = start_x, start_y
        
        for _ in range(max_length):
            # Find unvisited neighbor
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < h and 0 <= nx < w:
                        if skeleton[ny, nx] > 0 and (nx, ny) not in visited:
                            neighbors.append((nx, ny))
            
            if len(neighbors) == 0:
                break  # Dead end
            
            if len(neighbors) > 1:
                # Reached a junction
                # Pick one neighbor to include as junction point
                x, y = neighbors[0]
                branch.append((x, y))
                break
            
            # Continue along single path
            x, y = neighbors[0]
            branch.append((x, y))
            visited.add((x, y))
            
            # Check if this is a junction
            neighbor_count = self._count_neighbors(skeleton, x, y)
            if neighbor_count >= 3:
                break  # Junction reached
        
        return branch
    
    def _compute_direction(self, points: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Compute direction vector from a list of points."""
        if len(points) < 2:
            return (1.0, 0.0)
        
        # Use first and last point
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        
        # Normalize
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return (1.0, 0.0)
        
        return (dx / length, dy / length)
    
    def _compute_main_direction(
        self,
        skeleton: np.ndarray,
        junction_x: int,
        junction_y: int,
        exclude_branch: List[Tuple[int, int]],
    ) -> Optional[Tuple[float, float]]:
        """Compute main direction at junction, excluding the spur branch."""
        h, w = skeleton.shape
        exclude_set = set(exclude_branch)
        
        # Find other branches at junction
        other_neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = junction_x + dx, junction_y + dy
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] > 0 and (nx, ny) not in exclude_set:
                        other_neighbors.append((nx, ny))
        
        if not other_neighbors:
            return None
        
        # Compute average direction of other branches
        directions = []
        for nx, ny in other_neighbors:
            dx = nx - junction_x
            dy = ny - junction_y
            length = np.sqrt(dx * dx + dy * dy)
            if length > 0:
                directions.append((dx / length, dy / length))
        
        if not directions:
            return None
        
        # Average direction
        avg_dx = sum(d[0] for d in directions) / len(directions)
        avg_dy = sum(d[1] for d in directions) / len(directions)
        
        length = np.sqrt(avg_dx * avg_dx + avg_dy * avg_dy)
        if length < 1e-10:
            return None
        
        return (avg_dx / length, avg_dy / length)
    
    def _angle_between(
        self,
        dir1: Tuple[float, float],
        dir2: Tuple[float, float],
    ) -> float:
        """Compute angle between two direction vectors (in degrees)."""
        dot = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        # Clamp to [-1, 1] to avoid numerical issues
        dot = max(-1.0, min(1.0, dot))
        angle_rad = np.arccos(dot)
        return np.degrees(angle_rad)

    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoint pixels (exactly 1 neighbor)."""
        endpoints = []
        
        # Pad image to handle borders
        padded = np.pad(skeleton, 1, mode="constant", constant_values=0)
        
        # Find all skeleton pixels
        ys, xs = np.where(padded > 0)
        
        for y, x in zip(ys, xs):
            # Count 8-connected neighbors
            neighbors = np.sum(padded[y-1:y+2, x-1:x+2] > 0) - 1
            if neighbors == 1:
                endpoints.append((y - 1, x - 1))  # Adjust for padding
        
        return endpoints
    
    def _count_neighbors(
        self,
        skeleton: np.ndarray,
        x: int,
        y: int,
    ) -> int:
        """Count 8-connected neighbors of a pixel."""
        h, w = skeleton.shape
        count = 0
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] > 0:
                        count += 1
        
        return count
    
    def _detect_keypoints(self, skeleton: np.ndarray) -> List[KeyPoint]:
        """
        Detect and classify keypoints in skeleton.
        
        Keypoint types:
        - Endpoint: 1 neighbor
        - Junction: 3+ neighbors
        - Corner: Sharp angle change (high curvature)
        
        Args:
            skeleton: Binary skeleton image
        
        Returns:
            List of detected keypoints
        """
        keypoints = []
        
        # Find all skeleton pixels
        ys, xs = np.where(skeleton > 0)
        
        for y, x in zip(ys, xs):
            neighbors = self._count_neighbors(skeleton, x, y)
            
            if neighbors == 1:
                # Endpoint
                keypoints.append(KeyPoint(
                    point=PointFloat(x=float(x), y=float(y)),
                    point_type=KeyPointType.ENDPOINT,
                    confidence=1.0,
                ))
            elif neighbors >= 3:
                # Junction
                keypoints.append(KeyPoint(
                    point=PointFloat(x=float(x), y=float(y)),
                    point_type=KeyPointType.JUNCTION,
                    confidence=1.0,
                ))
            elif neighbors == 2 and self.config.detect_corners:
                # Check if this is a corner (high curvature point)
                angle = self._compute_local_angle(skeleton, x, y)
                if angle is not None and angle < self.config.corner_angle_threshold:
                    keypoints.append(KeyPoint(
                        point=PointFloat(x=float(x), y=float(y)),
                        point_type=KeyPointType.CORNER,
                        confidence=min(1.0, (self.config.corner_angle_threshold - angle) / self.config.corner_angle_threshold),
                    ))
        
        return keypoints
    
    def _compute_local_angle(
        self,
        skeleton: np.ndarray,
        x: int,
        y: int,
        window: int = 3,
    ) -> Optional[float]:
        """
        Compute local angle at a skeleton point.
        
        Uses neighboring points in a window to determine the angle
        formed by the skeleton at this point.
        
        Args:
            skeleton: Binary skeleton image
            x, y: Point coordinates
            window: Window size for neighbor search
        
        Returns:
            Angle in degrees (0-180), or None if cannot compute
        """
        h, w = skeleton.shape
        
        # Find the two neighbors (for points with exactly 2 neighbors)
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] > 0:
                        neighbors.append((nx, ny))
        
        if len(neighbors) != 2:
            return None
        
        # Trace a few pixels in each direction for more stable angle
        dirs = []
        for nx, ny in neighbors:
            # Get direction vector from center to neighbor
            dx = nx - x
            dy = ny - y
            
            # Extend in this direction
            trace_x, trace_y = nx, ny
            for _ in range(window - 1):
                found = False
                for ddy in [-1, 0, 1]:
                    for ddx in [-1, 0, 1]:
                        if ddx == 0 and ddy == 0:
                            continue
                        nnx, nny = trace_x + ddx, trace_y + ddy
                        if 0 <= nny < h and 0 <= nnx < w:
                            if skeleton[nny, nnx] > 0 and (nnx, nny) != (x, y):
                                # Check if continuing in roughly same direction
                                if (nnx - x) * dx >= 0 or (nny - y) * dy >= 0:
                                    trace_x, trace_y = nnx, nny
                                    found = True
                                    break
                    if found:
                        break
                if not found:
                    break
            
            dirs.append((trace_x - x, trace_y - y))
        
        if len(dirs) != 2:
            return None
        
        # Compute angle between the two directions
        dx1, dy1 = dirs[0]
        dx2, dy2 = dirs[1]
        
        # Normalize
        len1 = np.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
        
        if len1 < 1e-10 or len2 < 1e-10:
            return None
        
        dx1, dy1 = dx1 / len1, dy1 / len1
        dx2, dy2 = dx2 / len2, dy2 / len2
        
        # Dot product gives cos(angle)
        # Note: vectors point outward from center, so we want angle between them
        dot = dx1 * dx2 + dy1 * dy2
        dot = max(-1.0, min(1.0, dot))
        
        # The angle between outward vectors; 180 = straight, 0/360 = fold back
        angle = np.degrees(np.arccos(dot))
        
        # Convert to interior angle (180 - angle gives sharpness)
        interior_angle = 180 - angle
        
        return interior_angle
        
        return keypoints
    
    def _compute_stroke_widths(
        self,
        skeleton: np.ndarray,
        distance_map: np.ndarray,
    ) -> np.ndarray:
        """
        Compute stroke width at each skeleton pixel.
        
        The distance transform value at skeleton pixel gives
        half the stroke width.
        
        Args:
            skeleton: Binary skeleton image
            distance_map: Distance transform of original binary
        
        Returns:
            Stroke width map (2 * distance_map at skeleton pixels)
        """
        stroke_width = np.zeros_like(distance_map)
        
        # Stroke width = 2 * distance at skeleton
        mask = skeleton > 0
        stroke_width[mask] = 2 * distance_map[mask]
        
        return stroke_width
    
    def trace_paths(
        self,
        skeleton: np.ndarray,
        keypoints: List[KeyPoint],
    ) -> List[List[Tuple[int, int]]]:
        """
        Trace connected paths between keypoints.
        
        Follows skeleton pixels to extract polylines.
        
        Args:
            skeleton: Binary skeleton image
            keypoints: Detected keypoints (endpoints, junctions)
        
        Returns:
            List of paths (each path is list of (x, y) tuples)
        """
        paths = []
        visited = np.zeros_like(skeleton, dtype=bool)
        
        # Mark keypoints
        keypoint_set = set()
        for kp in keypoints:
            keypoint_set.add((int(kp.point.x), int(kp.point.y)))
        
        # Start from each endpoint
        endpoints = [
            kp for kp in keypoints
            if kp.point_type == KeyPointType.ENDPOINT
        ]
        
        for start_kp in endpoints:
            start_x, start_y = int(start_kp.point.x), int(start_kp.point.y)
            
            if visited[start_y, start_x]:
                continue
            
            path = self._trace_single_path(
                skeleton, visited, start_x, start_y, keypoint_set
            )
            
            if len(path) >= 2:
                paths.append(path)
        
        return paths
    
    def _trace_single_path(
        self,
        skeleton: np.ndarray,
        visited: np.ndarray,
        start_x: int,
        start_y: int,
        keypoint_set: set,
    ) -> List[Tuple[int, int]]:
        """Trace a single path from starting point."""
        h, w = skeleton.shape
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        
        x, y = start_x, start_y
        
        while True:
            # Find unvisited neighbor
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    
                    if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                        path.append((nx, ny))
                        visited[ny, nx] = True
                        x, y = nx, ny
                        found = True
                        
                        # Stop at keypoints (junctions)
                        if (nx, ny) in keypoint_set and len(path) > 1:
                            return path
                        
                        break
                
                if found:
                    break
            
            if not found:
                break
        
        return path
