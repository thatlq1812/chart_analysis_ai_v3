"""
Image Preprocessor Module

Implements negative image transformation and adaptive preprocessing
for enhanced structural extraction from chart images.

Key techniques:
- Negative image inversion (background dark, strokes bright)
- Adaptive thresholding for non-uniform lighting
- White top-hat transform for thin stroke enhancement
- Denoising with edge preservation

Reference: docs/instruction_p2_research.md - Section 2
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreprocessConfig(BaseModel):
    """Configuration for image preprocessing."""
    
    # Negative transformation
    apply_negative: bool = Field(default=True, description="Apply negative transformation")
    
    # Adaptive thresholding
    adaptive_method: str = Field(
        default="gaussian",
        description="Adaptive method: 'gaussian' or 'mean'"
    )
    block_size: int = Field(
        default=11,
        ge=3,
        description="Block size for adaptive thresholding (must be odd)"
    )
    c_constant: int = Field(
        default=2,
        description="Constant subtracted from mean/weighted mean"
    )
    
    # Denoising
    apply_denoise: bool = Field(default=True, description="Apply denoising filter")
    denoise_strength: int = Field(
        default=10,
        ge=0,
        le=30,
        description="Denoising strength (h parameter)"
    )
    
    # Morphological enhancement
    apply_tophat: bool = Field(
        default=True,
        description="Apply white top-hat for stroke enhancement"
    )
    tophat_kernel_size: int = Field(default=5, ge=3, description="Top-hat kernel size")
    
    # Contrast enhancement
    apply_clahe: bool = Field(default=False, description="Apply CLAHE contrast enhancement")
    clahe_clip_limit: float = Field(default=2.0, gt=0, description="CLAHE clip limit")
    clahe_grid_size: int = Field(default=8, ge=1, description="CLAHE tile grid size")
    
    # [FIX] Post-negative cleaning (address artifacts before skeletonization)
    # Reference: User review - "Post-Negative Issue" analysis
    apply_text_masking: bool = Field(
        default=True,
        description="Mask text regions to prevent OCR artifacts in skeleton"
    )
    apply_grid_removal: bool = Field(
        default=True,
        description="Remove thin grid lines using morphological opening"
    )
    grid_removal_kernel_size: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Kernel size for grid line removal (smaller = more aggressive)"
    )
    apply_noise_filtering: bool = Field(
        default=True,
        description="Remove small connected components (salt-and-pepper noise)"
    )
    min_component_area: int = Field(
        default=20,
        ge=5,
        description="Minimum area (pixels) for connected components to keep"
    )
    
    # Morphology enhancement (existing)
    apply_morphology: bool = Field(
        default=False,
        description="Apply general morphological operations"
    )


@dataclass
class PreprocessResult:
    """Result of preprocessing operation."""
    
    negative_image: np.ndarray
    binary_image: np.ndarray
    grayscale_image: np.ndarray
    cleaned_image: np.ndarray  # [FIX] Post-cleaning binary (after masking/noise removal)
    operations_applied: list
    cleaning_stats: dict  # [FIX] Stats about cleaning operations


class ImagePreprocessor:
    """
    Preprocesses chart images for skeleton extraction.
    
    Implements the negative image + adaptive thresholding pipeline
    from Geo-SLM research for optimal structural extraction.
    
    Example:
        config = PreprocessConfig(apply_negative=True, apply_denoise=True)
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.process(image)
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(
        self,
        image: np.ndarray,
        chart_id: str = "unknown",
    ) -> PreprocessResult:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input BGR image (numpy array)
            chart_id: Chart identifier for logging
        
        Returns:
            PreprocessResult with processed images
        """
        operations = []
        
        self.logger.debug(f"Preprocessing started | chart_id={chart_id}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            operations.append("bgr_to_grayscale")
        else:
            gray = image.copy()
        
        # Step 1: Denoising (before negative to preserve edges)
        if self.config.apply_denoise:
            gray = self._denoise(gray)
            operations.append("denoise")
        
        # Step 2: Contrast enhancement (optional)
        if self.config.apply_clahe:
            gray = self._apply_clahe(gray)
            operations.append("clahe")
        
        # Step 3: Negative transformation
        if self.config.apply_negative:
            negative = self._to_negative(gray)
            operations.append("negative_transform")
        else:
            negative = gray.copy()
        
        # Step 4: White top-hat transform for thin stroke enhancement
        if self.config.apply_tophat:
            negative = self._white_tophat(negative)
            operations.append("white_tophat")
        
        # Step 5: Adaptive thresholding for binary image
        binary = self._adaptive_threshold(negative)
        operations.append("adaptive_threshold")
        
        # [FIX] Step 6: Post-negative cleaning (applied to binary, before skeleton)
        # This addresses the "Post-Negative Issue" where text, grid lines, and noise
        # create false artifacts in skeletonization.
        # Note: Full cleaning with text masking requires text_boxes from OCR,
        # which should be done via clean_for_skeleton() method after OCR runs.
        cleaning_stats = {
            "text_regions_masked": 0,
            "pixels_removed_by_grid_filter": 0,
            "components_removed_by_noise_filter": 0,
        }
        
        # Apply grid removal and noise filtering (text masking deferred to orchestrator)
        cleaned = binary.copy()
        if self.config.apply_grid_removal:
            before_pixels = np.sum(cleaned > 0)
            cleaned = self._remove_grid_lines(cleaned)
            after_pixels = np.sum(cleaned > 0)
            cleaning_stats["pixels_removed_by_grid_filter"] = int(before_pixels - after_pixels)
            operations.append("grid_removal")
        
        if self.config.apply_noise_filtering:
            cleaned, removed_count = self._remove_small_noise(cleaned)
            cleaning_stats["components_removed_by_noise_filter"] = removed_count
            operations.append("noise_filtering")
        
        self.logger.info(
            f"Preprocessing complete | chart_id={chart_id} | "
            f"operations={len(operations)} | "
            f"grid_pixels_removed={cleaning_stats['pixels_removed_by_grid_filter']} | "
            f"noise_components_removed={cleaning_stats['components_removed_by_noise_filter']}"
        )
        
        return PreprocessResult(
            negative_image=negative,
            binary_image=binary,
            grayscale_image=gray,
            cleaned_image=cleaned,  # [FIX] Use cleaned for skeleton
            operations_applied=operations,
            cleaning_stats=cleaning_stats,
        )
    
    def _to_negative(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to negative image.
        
        Inverts intensity: I_neg(x,y) = Max_val - I_src(x,y)
        This makes strokes bright on dark background.
        
        Args:
            image: Grayscale image
        
        Returns:
            Negative image
        """
        return cv2.bitwise_not(image)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising while preserving edges.
        
        Uses Non-local Means Denoising for grayscale.
        
        Args:
            image: Grayscale image
        
        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoising(
            image,
            h=self.config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Enhances local contrast while limiting noise amplification.
        
        Args:
            image: Grayscale image
        
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
        )
        return clahe.apply(image)
    
    def _white_tophat(self, image: np.ndarray) -> np.ndarray:
        """
        Apply white top-hat transform.
        
        Extracts bright structures (strokes) smaller than kernel.
        Formula: tophat(I) = I - opening(I)
        
        This removes low-frequency background variations and enhances
        thin bright strokes in the negative image.
        
        Args:
            image: Grayscale image (negative)
        
        Returns:
            Top-hat filtered image
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.tophat_kernel_size, self.config.tophat_kernel_size),
        )
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for binary conversion.
        
        Handles non-uniform lighting in scanned documents.
        
        Args:
            image: Grayscale image
        
        Returns:
            Binary image (0 or 255)
        """
        method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.config.adaptive_method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        
        # Ensure block_size is odd
        block_size = self.config.block_size
        if block_size % 2 == 0:
            block_size += 1
        
        return cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=self.config.c_constant,
        )
    
    def extract_color_mask(
        self,
        image: np.ndarray,
        target_color: Tuple[int, int, int],
        tolerance: int = 30,
    ) -> np.ndarray:
        """
        Extract mask for specific color (for series separation).
        
        Args:
            image: BGR image
            target_color: Target color (B, G, R)
            tolerance: Color tolerance
        
        Returns:
            Binary mask where target color is white
        """
        lower = np.array([max(0, c - tolerance) for c in target_color])
        upper = np.array([min(255, c + tolerance) for c in target_color])
        return cv2.inRange(image, lower, upper)
    
    def compute_distance_transform(self, binary: np.ndarray) -> np.ndarray:
        """
        Compute distance transform for stroke width estimation.
        
        Distance transform gives the distance from each foreground pixel
        to the nearest background pixel. The value at the skeleton
        indicates the stroke half-width.
        
        Args:
            binary: Binary image (strokes white)
        
        Returns:
            Distance transform image
        """
        return cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # ========================================================================
    # [FIX] Post-Negative Cleaning Methods
    # Reference: User review - "Post-Negative Issue" analysis
    # These methods address artifacts (text, grid, noise) that interfere with
    # skeletonization, causing false junctions and vector noise.
    # ========================================================================
    
    def clean_for_skeleton(
        self,
        binary_image: np.ndarray,
        text_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        chart_id: str = "unknown",
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply post-negative cleaning to prepare binary image for skeletonization.
        
        This is the FIX for the "Post-Negative Issue" where text, grid lines,
        and noise create false artifacts in skeleton extraction.
        
        Processing order:
            1. Mask text regions (if boxes provided)
            2. Remove thin grid lines (morphological opening)
            3. Remove small noise components (connected components filter)
        
        Args:
            binary_image: Binary image (foreground=255, background=0)
            text_boxes: List of (x, y, w, h) boxes to mask (from OCR)
            chart_id: Chart identifier for logging
        
        Returns:
            Tuple of (cleaned_image, stats_dict)
        """
        self.logger.debug(f"Post-negative cleaning started | chart_id={chart_id}")
        
        stats = {
            "text_regions_masked": 0,
            "pixels_removed_by_grid_filter": 0,
            "components_removed_by_noise_filter": 0,
        }
        
        cleaned = binary_image.copy()
        
        # Step 1: Mask text regions
        if self.config.apply_text_masking and text_boxes:
            cleaned, masked_count = self._mask_text_regions(cleaned, text_boxes)
            stats["text_regions_masked"] = masked_count
        
        # Step 2: Remove thin grid lines
        if self.config.apply_grid_removal:
            before_pixels = np.sum(cleaned > 0)
            cleaned = self._remove_grid_lines(cleaned)
            after_pixels = np.sum(cleaned > 0)
            stats["pixels_removed_by_grid_filter"] = int(before_pixels - after_pixels)
        
        # Step 3: Remove small noise components
        if self.config.apply_noise_filtering:
            cleaned, removed_count = self._remove_small_noise(cleaned)
            stats["components_removed_by_noise_filter"] = removed_count
        
        self.logger.info(
            f"Post-negative cleaning complete | chart_id={chart_id} | "
            f"text_masked={stats['text_regions_masked']} | "
            f"grid_pixels_removed={stats['pixels_removed_by_grid_filter']} | "
            f"noise_components_removed={stats['components_removed_by_noise_filter']}"
        )
        
        return cleaned, stats
    
    def _mask_text_regions(
        self,
        binary_image: np.ndarray,
        text_boxes: List[Tuple[int, int, int, int]],
        padding: int = 2,
    ) -> Tuple[np.ndarray, int]:
        """
        Mask text regions by filling with background (black).
        
        This prevents OCR text from creating skeleton artifacts like
        loops (from 'O', '0') or spurs (from serifs).
        
        Args:
            binary_image: Binary image to modify
            text_boxes: List of (x, y, w, h) bounding boxes
            padding: Extra padding around text boxes
        
        Returns:
            Tuple of (masked_image, count_of_masked_regions)
        """
        masked = binary_image.copy()
        h, w = masked.shape[:2]
        count = 0
        
        for box in text_boxes:
            x, y, bw, bh = box
            # Apply padding with bounds checking
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            
            # Fill with black (background)
            masked[y1:y2, x1:x2] = 0
            count += 1
        
        return masked, count
    
    def _remove_grid_lines(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Remove thin grid lines using morphological opening.
        
        Grid lines are typically thinner than data lines (1-2px vs 2-4px).
        Opening (erosion followed by dilation) removes thin structures
        while preserving thicker ones.
        
        Args:
            binary_image: Binary image (foreground=255)
        
        Returns:
            Binary image with grid lines removed
        """
        kernel_size = self.config.grid_removal_kernel_size
        
        # Use a small rectangular kernel to target thin lines
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # MORPH_OPEN = erosion followed by dilation
        # This removes features smaller than kernel while preserving larger ones
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    def _remove_small_noise(
        self,
        binary_image: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Remove small connected components (salt-and-pepper noise).
        
        After negative transform and thresholding, isolated pixels or
        small clusters may appear as noise. These create spurious
        skeleton branches if not removed.
        
        Uses connected components analysis with 8-connectivity.
        
        Args:
            binary_image: Binary image (foreground=255)
        
        Returns:
            Tuple of (cleaned_image, count_of_removed_components)
        """
        min_size = self.config.min_component_area
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        # Create output image
        cleaned = np.zeros_like(binary_image)
        
        # Component 0 is background, start from 1
        removed_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                cleaned[labels == i] = 255
            else:
                removed_count += 1
        
        return cleaned, removed_count
    
    def extract_text_boxes_for_masking(
        self,
        ocr_texts: List,  # List[OCRText] from ocr_engine
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert OCRText list to simple (x, y, w, h) boxes for masking.
        
        Helper method to bridge between OCR output and cleaning input.
        
        Args:
            ocr_texts: List of OCRText objects with bbox attributes
        
        Returns:
            List of (x, y, width, height) tuples
        """
        boxes = []
        for text in ocr_texts:
            if hasattr(text, 'bbox') and text.bbox is not None:
                bbox = text.bbox
                x = bbox.x_min
                y = bbox.y_min
                w = bbox.x_max - bbox.x_min
                h = bbox.y_max - bbox.y_min
                boxes.append((x, y, w, h))
        return boxes
