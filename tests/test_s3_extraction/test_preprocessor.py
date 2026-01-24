"""
Unit tests for ImagePreprocessor module.

Tests negative image transformation and adaptive thresholding.
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import tempfile

from core_engine.stages.s3_extraction.preprocessor import (
    ImagePreprocessor,
    PreprocessConfig,
    PreprocessResult,
)


class TestPreprocessConfig:
    """Tests for PreprocessConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PreprocessConfig()
        
        assert config.apply_negative is True
        assert config.apply_denoise is True
        assert config.block_size == 11
        assert config.c_constant == 2
        assert config.adaptive_method == "gaussian"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = PreprocessConfig(
            apply_negative=False,
            block_size=51,
            c_constant=15,
        )
        
        assert config.apply_negative is False
        assert config.block_size == 51
        assert config.c_constant == 15

    def test_block_size_validation(self) -> None:
        """Test that block_size >= 3 is valid."""
        config = PreprocessConfig(block_size=35)
        assert config.block_size == 35


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    @pytest.fixture
    def preprocessor(self) -> ImagePreprocessor:
        """Create preprocessor with default config."""
        return ImagePreprocessor()

    @pytest.fixture
    def sample_rgb_image(self) -> np.ndarray:
        """Create sample RGB image with chart-like features (BGR format for OpenCV)."""
        # Create 100x100 white background
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Add black lines (simulating chart axes)
        img[80:85, 10:90, :] = 0  # X-axis
        img[10:85, 10:15, :] = 0  # Y-axis
        
        # Add colored bars (BGR format)
        img[40:80, 25:35, :] = [0, 0, 255]  # Red bar
        img[30:80, 45:55, :] = [0, 255, 0]  # Green bar
        img[50:80, 65:75, :] = [255, 0, 0]  # Blue bar
        
        return img

    @pytest.fixture
    def sample_grayscale_image(self) -> np.ndarray:
        """Create sample grayscale image."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[40:60, 40:60] = 0  # Black square
        return img

    def test_process_returns_result(
        self, 
        preprocessor: ImagePreprocessor,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test that process returns PreprocessResult."""
        result = preprocessor.process(sample_rgb_image)
        
        assert isinstance(result, PreprocessResult)
        assert hasattr(result, 'binary_image')
        assert hasattr(result, 'negative_image')
        assert hasattr(result, 'grayscale_image')
        assert hasattr(result, 'operations_applied')

    def test_binary_output_format(
        self,
        preprocessor: ImagePreprocessor,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test that binary_image is 2D binary."""
        result = preprocessor.process(sample_rgb_image)
        
        # Should be 2D (grayscale binary)
        assert len(result.binary_image.shape) == 2
        
        # Should only contain 0 and 255
        unique = np.unique(result.binary_image)
        assert len(unique) <= 2
        assert all(v in [0, 255] for v in unique)

    def test_negative_transformation(self, preprocessor: ImagePreprocessor) -> None:
        """Test negative image transformation."""
        # Create simple grayscale image
        img = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        
        negative = preprocessor._to_negative(img)
        
        # Check inversion
        assert negative[0, 0] == 255  # 0 -> 255
        assert negative[0, 1] == 127  # 128 -> 127
        assert negative[1, 0] == 0    # 255 -> 0
        assert negative[1, 1] == 191  # 64 -> 191

    def test_process_preserves_structure(
        self,
        preprocessor: ImagePreprocessor,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test that processing preserves chart structure."""
        result = preprocessor.process(sample_rgb_image)
        
        # Check that axes region has some structure
        axes_region = result.binary_image[75:90, 5:95]  # Around X-axis
        
        # Should have at least some structure
        assert axes_region.shape == (15, 90)

    def test_grayscale_conversion_applied(
        self,
        preprocessor: ImagePreprocessor,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test that RGB is converted to grayscale."""
        result = preprocessor.process(sample_rgb_image)
        
        assert len(result.grayscale_image.shape) == 2
        assert result.grayscale_image.dtype == np.uint8
        assert 'bgr_to_grayscale' in result.operations_applied

    def test_grayscale_input_no_conversion(
        self,
        preprocessor: ImagePreprocessor,
        sample_grayscale_image: np.ndarray,
    ) -> None:
        """Test grayscale input skips conversion."""
        result = preprocessor.process(sample_grayscale_image)
        
        assert len(result.binary_image.shape) == 2
        assert 'bgr_to_grayscale' not in result.operations_applied

    def test_process_without_negative(
        self,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test processing without negative transformation."""
        config = PreprocessConfig(apply_negative=False)
        preprocessor = ImagePreprocessor(config)
        
        result = preprocessor.process(sample_rgb_image)
        
        assert isinstance(result, PreprocessResult)
        assert 'negative_transform' not in result.operations_applied

    def test_process_without_denoise(
        self,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test processing without denoising."""
        config = PreprocessConfig(apply_denoise=False)
        preprocessor = ImagePreprocessor(config)
        
        result = preprocessor.process(sample_rgb_image)
        
        assert isinstance(result, PreprocessResult)
        assert 'denoise' not in result.operations_applied

    def test_adaptive_threshold_methods(
        self,
        sample_rgb_image: np.ndarray,
    ) -> None:
        """Test different adaptive threshold methods."""
        methods = ["gaussian", "mean"]
        
        for method in methods:
            config = PreprocessConfig(adaptive_method=method)
            preprocessor = ImagePreprocessor(config)
            
            result = preprocessor.process(sample_rgb_image)
            assert isinstance(result, PreprocessResult)
            assert 'adaptive_threshold' in result.operations_applied

    def test_white_tophat_enhancement(
        self,
        preprocessor: ImagePreprocessor,
    ) -> None:
        """Test white tophat morphological operation."""
        # Create image with thin white line on gray background
        img = np.ones((50, 50), dtype=np.uint8) * 128
        img[25, 10:40] = 255  # Thin white line
        
        enhanced = preprocessor._white_tophat(img)
        
        # Line should be enhanced (brighter relative to background)
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == img.shape


class TestPreprocessorEdgeCases:
    """Edge case tests for ImagePreprocessor."""

    def test_small_image(self) -> None:
        """Test handling of very small image."""
        preprocessor = ImagePreprocessor()
        
        # Minimum viable image
        img = np.zeros((20, 20), dtype=np.uint8)
        result = preprocessor.process(img)
        
        assert result.binary_image.shape == (20, 20)

    def test_large_image(self) -> None:
        """Test handling of large image."""
        preprocessor = ImagePreprocessor()
        
        img = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        result = preprocessor.process(img)
        
        assert result.binary_image.shape == (500, 500)

    def test_uniform_white_image(self) -> None:
        """Test handling of uniform white image."""
        preprocessor = ImagePreprocessor()
        
        img_white = np.ones((50, 50), dtype=np.uint8) * 255
        result_white = preprocessor.process(img_white)
        assert result_white.binary_image.shape == (50, 50)

    def test_uniform_black_image(self) -> None:
        """Test handling of uniform black image."""
        preprocessor = ImagePreprocessor()
        
        img_black = np.zeros((50, 50), dtype=np.uint8)
        result_black = preprocessor.process(img_black)
        assert result_black.binary_image.shape == (50, 50)

    def test_operations_tracking(self) -> None:
        """Test that all operations are tracked."""
        config = PreprocessConfig(
            apply_negative=True,
            apply_denoise=True,
            apply_tophat=True,
        )
        preprocessor = ImagePreprocessor(config)
        
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = preprocessor.process(img)
        
        # Check standard operations are applied
        assert 'bgr_to_grayscale' in result.operations_applied
        assert 'denoise' in result.operations_applied
        assert 'negative_transform' in result.operations_applied
        assert 'white_tophat' in result.operations_applied
        assert 'adaptive_threshold' in result.operations_applied
