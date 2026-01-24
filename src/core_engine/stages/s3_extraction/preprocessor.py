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
from typing import Optional, Tuple

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


@dataclass
class PreprocessResult:
    """Result of preprocessing operation."""
    
    negative_image: np.ndarray
    binary_image: np.ndarray
    grayscale_image: np.ndarray
    operations_applied: list


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
        
        self.logger.info(
            f"Preprocessing complete | chart_id={chart_id} | "
            f"operations={len(operations)}"
        )
        
        return PreprocessResult(
            negative_image=negative,
            binary_image=binary,
            grayscale_image=gray,
            operations_applied=operations,
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
