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
    
    # Junction handling
    detect_junctions: bool = Field(
        default=True,
        description="Detect and classify junction points"
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
        - Corner: Sharp angle change
        
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
