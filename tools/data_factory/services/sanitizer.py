"""
Image Sanitizer - Filter and validate extracted images

This service applies quality checks to filter out:
- Too small images
- Bad aspect ratios (very thin/wide)
- Blank/solid color images
- Non-chart images (photos, logos, etc.)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from ..config import DataFactoryConfig, QualityConfig
from ..schemas import ChartImage


class ImageSanitizer:
    """
    Validate and filter chart images based on quality criteria.
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        self.quality = config.quality
    
    def is_valid(self, image: Image.Image) -> Tuple[bool, List[str]]:
        """
        Check if an image passes quality filters.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid, list of validation notes)
        """
        notes: List[str] = []
        is_valid = True
        
        # Check dimensions
        width, height = image.size
        
        if width < self.quality.min_width:
            notes.append(f"Width too small: {width} < {self.quality.min_width}")
            is_valid = False
        
        if height < self.quality.min_height:
            notes.append(f"Height too small: {height} < {self.quality.min_height}")
            is_valid = False
        
        if width > self.quality.max_width:
            notes.append(f"Width too large: {width} > {self.quality.max_width}")
            is_valid = False
        
        if height > self.quality.max_height:
            notes.append(f"Height too large: {height} > {self.quality.max_height}")
            is_valid = False
        
        # Check aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        
        if aspect_ratio < self.quality.min_aspect_ratio:
            notes.append(f"Aspect ratio too small: {aspect_ratio:.2f} < {self.quality.min_aspect_ratio}")
            is_valid = False
        
        if aspect_ratio > self.quality.max_aspect_ratio:
            notes.append(f"Aspect ratio too large: {aspect_ratio:.2f} > {self.quality.max_aspect_ratio}")
            is_valid = False
        
        # Check if image is too uniform (blank/solid color)
        uniformity = self._calculate_uniformity(image)
        if uniformity > self.quality.max_uniformity:
            notes.append(f"Image too uniform (likely blank): {uniformity:.3f} > {self.quality.max_uniformity}")
            is_valid = False
        
        # Check unique colors
        unique_colors = self._count_unique_colors(image)
        if unique_colors < self.quality.min_unique_colors:
            notes.append(f"Too few colors: {unique_colors} < {self.quality.min_unique_colors}")
            is_valid = False
        
        return is_valid, notes
    
    def _calculate_uniformity(self, image: Image.Image) -> float:
        """
        Calculate image uniformity (how similar all pixels are).
        
        Returns:
            Float between 0 and 1, where 1 is completely uniform
        """
        try:
            # Convert to grayscale for uniformity check
            gray = image.convert("L")
            pixels = np.array(gray).flatten()
            
            # Calculate standard deviation
            std = np.std(pixels)
            
            # Normalize: 0 std = 1.0 uniformity, high std = 0.0 uniformity
            # Typical image std is around 50-80
            uniformity = 1.0 - min(std / 100.0, 1.0)
            
            return uniformity
            
        except Exception:
            return 0.0  # Assume not uniform on error
    
    def _count_unique_colors(self, image: Image.Image, sample_size: int = 10000) -> int:
        """
        Count approximate number of unique colors in image.
        
        Uses sampling for large images to improve performance.
        """
        try:
            # Convert to RGB
            rgb = image.convert("RGB")
            pixels = list(rgb.getdata())
            
            # Sample if too many pixels
            if len(pixels) > sample_size:
                import random
                random.seed(self.config.random_seed)
                pixels = random.sample(pixels, sample_size)
            
            # Count unique
            unique = len(set(pixels))
            
            return unique
            
        except Exception:
            return 100  # Assume enough colors on error
    
    def validate_chart_image(self, chart_image: ChartImage) -> bool:
        """
        Validate a ChartImage object.
        
        Updates the chart_image with validation results.
        
        Returns:
            True if image is valid
        """
        if not chart_image.image_path.exists():
            chart_image.is_valid = False
            chart_image.validation_notes.append("Image file not found")
            return False
        
        try:
            image = Image.open(chart_image.image_path)
            is_valid, notes = self.is_valid(image)
            
            chart_image.is_valid = is_valid
            chart_image.validation_notes.extend(notes)
            chart_image.quality_score = self._calculate_quality_score(image, notes)
            
            return is_valid
            
        except Exception as e:
            chart_image.is_valid = False
            chart_image.validation_notes.append(f"Failed to open image: {e}")
            return False
    
    def _calculate_quality_score(self, image: Image.Image, notes: List[str]) -> float:
        """
        Calculate overall quality score for an image.
        
        Returns:
            Float between 0 and 1
        """
        score = 1.0
        
        # Penalize for each validation issue
        score -= len(notes) * 0.2
        
        # Bonus for good size (larger is better for charts)
        width, height = image.size
        size_score = min((width * height) / (1000 * 1000), 1.0) * 0.1
        score += size_score
        
        # Bonus for good aspect ratio (close to golden ratio or square)
        aspect = width / height if height > 0 else 1
        if 0.8 <= aspect <= 1.5:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def filter_images(self, images: List[ChartImage]) -> Tuple[List[ChartImage], List[ChartImage]]:
        """
        Filter a list of images, separating valid from invalid.
        
        Args:
            images: List of ChartImage objects
            
        Returns:
            Tuple of (valid_images, invalid_images)
        """
        valid = []
        invalid = []
        
        for img in images:
            if self.validate_chart_image(img):
                valid.append(img)
            else:
                invalid.append(img)
        
        logger.info(f"Filtering complete | valid={len(valid)} | invalid={len(invalid)}")
        
        return valid, invalid
    
    def cleanup_invalid_images(self, images: List[ChartImage], delete_files: bool = False) -> int:
        """
        Remove invalid images from the list and optionally delete files.
        
        Args:
            images: List of ChartImage objects
            delete_files: If True, delete invalid image files
            
        Returns:
            Number of images removed
        """
        removed = 0
        
        for img in images:
            if not img.is_valid:
                if delete_files and img.image_path.exists():
                    img.image_path.unlink()
                    logger.debug(f"Deleted invalid image | image_id={img.image_id}")
                removed += 1
        
        logger.info(f"Cleanup complete | removed={removed} | deleted_files={delete_files}")
        return removed


class ChartDetector:
    """
    Heuristic-based chart detection.
    
    Uses image analysis to determine if an image is likely a chart
    vs. a photo, logo, or other non-chart image.
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
    
    def is_likely_chart(self, image: Image.Image) -> Tuple[bool, float, str]:
        """
        Determine if an image is likely a chart.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_chart, confidence, reason)
        """
        scores = {}
        
        # Check for chart-like characteristics
        scores["has_axes"] = self._check_axes(image)
        scores["has_grid"] = self._check_grid_lines(image)
        scores["has_bars"] = self._check_bars(image)
        scores["has_text"] = self._check_text_regions(image)
        scores["color_distribution"] = self._check_color_distribution(image)
        scores["edge_density"] = self._check_edge_density(image)
        
        # Calculate overall score
        total_score = sum(scores.values()) / len(scores)
        
        is_chart = total_score > 0.4
        
        # Generate reason
        top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        reason = f"Top features: {', '.join([f'{k}={v:.2f}' for k, v in top_features])}"
        
        return is_chart, total_score, reason
    
    def _check_axes(self, image: Image.Image) -> float:
        """Check for axis-like lines (horizontal and vertical)."""
        try:
            import cv2
            
            # Convert to grayscale
            gray = np.array(image.convert("L"))
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
            
            if lines is None:
                return 0.0
            
            # Count horizontal and vertical lines
            h_lines = 0
            v_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 10 or angle > 170:  # Horizontal
                    h_lines += 1
                elif 80 < angle < 100:  # Vertical
                    v_lines += 1
            
            # Score based on having both axes
            if h_lines > 0 and v_lines > 0:
                return min(1.0, (h_lines + v_lines) / 10)
            
            return 0.2 if (h_lines > 0 or v_lines > 0) else 0.0
            
        except Exception:
            return 0.3  # Default moderate score
    
    def _check_grid_lines(self, image: Image.Image) -> float:
        """Check for grid-like patterns."""
        try:
            import cv2
            
            gray = np.array(image.convert("L"))
            edges = cv2.Canny(gray, 30, 100)
            
            # Count edge pixels
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # Charts typically have moderate edge density
            if 0.02 < edge_ratio < 0.15:
                return 0.6
            elif 0.01 < edge_ratio < 0.2:
                return 0.3
            
            return 0.1
            
        except Exception:
            return 0.3
    
    def _check_bars(self, image: Image.Image) -> float:
        """Check for bar-like rectangular regions."""
        try:
            import cv2
            
            # Convert to grayscale
            gray = np.array(image.convert("L"))
            
            # Threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count rectangular contours
            rect_count = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                
                # Bar-like aspect ratios
                if aspect > 2 or aspect < 0.5:
                    area = w * h
                    if area > 100:  # Minimum size
                        rect_count += 1
            
            return min(1.0, rect_count / 10)
            
        except Exception:
            return 0.3
    
    def _check_text_regions(self, image: Image.Image) -> float:
        """Check for text-like regions (labels, titles)."""
        # Simple heuristic: charts usually have limited text regions
        # This is a placeholder - could use OCR for better detection
        return 0.5
    
    def _check_color_distribution(self, image: Image.Image) -> float:
        """Check if color distribution is chart-like."""
        try:
            # Convert to RGB
            rgb = np.array(image.convert("RGB"))
            
            # Get unique colors
            pixels = rgb.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            
            n_colors = len(unique_colors)
            
            # Charts typically have limited color palette
            if 10 < n_colors < 500:
                return 0.7
            elif 5 < n_colors < 1000:
                return 0.4
            
            return 0.2
            
        except Exception:
            return 0.3
    
    def _check_edge_density(self, image: Image.Image) -> float:
        """Check edge density pattern."""
        try:
            import cv2
            
            gray = np.array(image.convert("L"))
            edges = cv2.Canny(gray, 50, 150)
            
            density = np.sum(edges > 0) / edges.size
            
            # Charts have moderate edge density
            if 0.03 < density < 0.12:
                return 0.7
            elif 0.02 < density < 0.2:
                return 0.4
            
            return 0.2
            
        except Exception:
            return 0.3
