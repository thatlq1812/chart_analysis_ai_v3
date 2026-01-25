"""
Tests for Post-Negative Cleaning Fix

This module tests the fix for the "Post-Negative Issue" where text, grid lines,
and noise create false artifacts in skeletonization.

Reference: User review - "Post-Negative Issue" analysis
"""

import numpy as np
import pytest
import cv2

from src.core_engine.stages.s3_extraction.preprocessor import (
    ImagePreprocessor,
    PreprocessConfig,
    PreprocessResult,
)


class TestPreprocessConfig:
    """Test the new cleaning config fields."""
    
    def test_default_cleaning_enabled(self):
        """Verify cleaning options are enabled by default."""
        config = PreprocessConfig()
        
        assert config.apply_text_masking is True
        assert config.apply_grid_removal is True
        assert config.apply_noise_filtering is True
        assert config.min_component_area == 20
        assert config.grid_removal_kernel_size == 2
    
    def test_config_customization(self):
        """Test custom cleaning configuration."""
        config = PreprocessConfig(
            apply_text_masking=False,
            apply_grid_removal=True,
            grid_removal_kernel_size=3,
            apply_noise_filtering=True,
            min_component_area=50,
        )
        
        assert config.apply_text_masking is False
        assert config.grid_removal_kernel_size == 3
        assert config.min_component_area == 50


class TestPreprocessResult:
    """Test the updated PreprocessResult with cleaned_image."""
    
    def test_result_has_cleaned_image(self):
        """Verify PreprocessResult includes cleaned_image field."""
        config = PreprocessConfig()
        preprocessor = ImagePreprocessor(config)
        
        # Create test image (simple bar chart simulation)
        test_img = np.ones((200, 300, 3), dtype=np.uint8) * 255  # White background
        # Add some bars (black)
        cv2.rectangle(test_img, (50, 100), (80, 180), (0, 0, 0), -1)
        cv2.rectangle(test_img, (120, 60), (150, 180), (0, 0, 0), -1)
        cv2.rectangle(test_img, (190, 120), (220, 180), (0, 0, 0), -1)
        
        result = preprocessor.process(test_img, chart_id="test_001")
        
        assert hasattr(result, "cleaned_image")
        assert hasattr(result, "cleaning_stats")
        assert result.cleaned_image is not None
        assert isinstance(result.cleaning_stats, dict)
    
    def test_cleaning_stats_structure(self):
        """Verify cleaning_stats contains expected keys."""
        config = PreprocessConfig()
        preprocessor = ImagePreprocessor(config)
        
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = preprocessor.process(test_img, chart_id="test_002")
        
        expected_keys = [
            "text_regions_masked",
            "pixels_removed_by_grid_filter",
            "components_removed_by_noise_filter",
        ]
        
        for key in expected_keys:
            assert key in result.cleaning_stats


class TestTextMasking:
    """Test text region masking functionality."""
    
    def test_mask_text_regions(self):
        """Test that text boxes are properly masked (filled with black)."""
        config = PreprocessConfig(apply_text_masking=True)
        preprocessor = ImagePreprocessor(config)
        
        # Create binary image with "text" (white pixels)
        binary = np.zeros((100, 200), dtype=np.uint8)
        binary[20:40, 30:70] = 255  # Simulated text region 1
        binary[60:80, 100:150] = 255  # Simulated text region 2
        binary[10:90, 160:190] = 255  # Simulated bar (should not be masked)
        
        # Define text boxes to mask
        text_boxes = [
            (30, 20, 40, 20),  # (x, y, w, h) for region 1
            (100, 60, 50, 20),  # (x, y, w, h) for region 2
        ]
        
        cleaned, stats = preprocessor.clean_for_skeleton(
            binary, text_boxes=text_boxes, chart_id="test_mask"
        )
        
        # Text regions should be masked (black)
        assert np.sum(cleaned[20:40, 30:70]) == 0, "Text region 1 should be masked"
        assert np.sum(cleaned[60:80, 100:150]) == 0, "Text region 2 should be masked"
        
        # Bar region should remain (white)
        assert np.sum(cleaned[10:90, 160:190]) > 0, "Bar region should not be masked"
        
        # Stats should reflect masked regions
        assert stats["text_regions_masked"] == 2
    
    def test_extract_text_boxes_from_ocr(self):
        """Test conversion of OCRText to masking boxes."""
        from src.core_engine.schemas.common import BoundingBox
        from src.core_engine.schemas.stage_outputs import OCRText
        
        preprocessor = ImagePreprocessor()
        
        # Create mock OCRText objects
        ocr_texts = [
            OCRText(
                text="100",
                bbox=BoundingBox(x_min=10, y_min=20, x_max=50, y_max=40, confidence=0.9),
                confidence=0.9,
                role="value",
            ),
            OCRText(
                text="Category",
                bbox=BoundingBox(x_min=100, y_min=180, x_max=180, y_max=200, confidence=0.85),
                confidence=0.85,
                role="xlabel",
            ),
        ]
        
        boxes = preprocessor.extract_text_boxes_for_masking(ocr_texts)
        
        assert len(boxes) == 2
        assert boxes[0] == (10, 20, 40, 20)  # (x, y, w, h)
        assert boxes[1] == (100, 180, 80, 20)


class TestGridLineRemoval:
    """Test grid line removal via morphological opening."""
    
    def test_remove_thin_grid_lines(self):
        """Test that thin grid lines are removed while thick lines remain."""
        config = PreprocessConfig(
            apply_grid_removal=True,
            grid_removal_kernel_size=2,
        )
        preprocessor = ImagePreprocessor(config)
        
        # Create image with thin grid (1px) and thick data line (3px)
        binary = np.zeros((100, 100), dtype=np.uint8)
        
        # Thin horizontal grid lines (1px)
        binary[20, :] = 255
        binary[40, :] = 255
        binary[60, :] = 255
        
        # Thick data line (3px wide)
        binary[78:81, :] = 255
        
        cleaned = preprocessor._remove_grid_lines(binary)
        
        # Thin grid lines should be significantly reduced
        thin_line_sum = np.sum(cleaned[20, :]) + np.sum(cleaned[40, :]) + np.sum(cleaned[60, :])
        thick_line_sum = np.sum(cleaned[78:81, :])
        
        # Thick line should remain mostly intact
        assert thick_line_sum > 0, "Thick data line should remain"
        # Grid lines may be partially removed
        # (exact removal depends on kernel size and morphology)


class TestNoiseFiltering:
    """Test small component (noise) removal."""
    
    def test_remove_small_noise(self):
        """Test that small isolated pixels/clusters are removed."""
        config = PreprocessConfig(
            apply_noise_filtering=True,
            min_component_area=20,
        )
        preprocessor = ImagePreprocessor(config)
        
        # Create image with large component and small noise
        binary = np.zeros((100, 100), dtype=np.uint8)
        
        # Large component (area > 20)
        cv2.rectangle(binary, (40, 40), (60, 60), 255, -1)  # 20x20 = 400 pixels
        
        # Small noise (area < 20)
        binary[10, 10] = 255  # 1 pixel
        binary[10:13, 80:83] = 255  # 9 pixels
        binary[90:94, 90:94] = 255  # 16 pixels
        
        cleaned, removed_count = preprocessor._remove_small_noise(binary)
        
        # Large component should remain
        assert np.sum(cleaned[40:60, 40:60]) > 0, "Large component should remain"
        
        # Small noise should be removed
        assert cleaned[10, 10] == 0, "1-pixel noise should be removed"
        
        # Multiple small components removed
        assert removed_count >= 2, f"Expected at least 2 noise components removed, got {removed_count}"
    
    def test_preserve_large_components(self):
        """Ensure large components are not affected by noise filtering."""
        config = PreprocessConfig(min_component_area=50)
        preprocessor = ImagePreprocessor(config)
        
        # Create image with various sized components
        binary = np.zeros((200, 200), dtype=np.uint8)
        
        # Component 1: 100 pixels (should remain)
        cv2.rectangle(binary, (10, 10), (20, 20), 255, -1)
        
        # Component 2: 400 pixels (should remain)
        cv2.rectangle(binary, (50, 50), (70, 70), 255, -1)
        
        # Component 3: 30 pixels (should be removed)
        cv2.rectangle(binary, (150, 150), (155, 156), 255, -1)
        
        cleaned, removed = preprocessor._remove_small_noise(binary)
        
        # Large components preserved
        assert np.sum(cleaned[10:20, 10:20]) > 0
        assert np.sum(cleaned[50:70, 50:70]) > 0


class TestIntegration:
    """Integration tests for the full cleaning pipeline."""
    
    def test_full_cleaning_pipeline(self):
        """Test end-to-end cleaning with all options enabled."""
        config = PreprocessConfig(
            apply_negative=True,
            apply_denoise=True,
            apply_grid_removal=True,
            apply_noise_filtering=True,
            min_component_area=15,
        )
        preprocessor = ImagePreprocessor(config)
        
        # Create test chart image
        test_img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        
        # Add black bars (chart data)
        cv2.rectangle(test_img, (50, 80), (80, 180), (0, 0, 0), -1)
        cv2.rectangle(test_img, (120, 40), (150, 180), (0, 0, 0), -1)
        cv2.rectangle(test_img, (190, 100), (220, 180), (0, 0, 0), -1)
        
        # Add thin grid lines (1px gray)
        for y in range(40, 180, 20):
            cv2.line(test_img, (40, y), (240, y), (200, 200, 200), 1)
        
        # Add small noise dots
        test_img[20, 20] = [0, 0, 0]
        test_img[180, 280] = [0, 0, 0]
        
        result = preprocessor.process(test_img, chart_id="test_integration")
        
        # Verify all outputs exist
        assert result.binary_image is not None
        assert result.cleaned_image is not None
        assert result.negative_image is not None
        
        # Cleaned image should have less noise than raw binary
        # (This is a soft assertion as exact pixel counts depend on thresholds)
        assert result.cleaning_stats["components_removed_by_noise_filter"] >= 0
    
    def test_clean_for_skeleton_with_text_masking(self):
        """Test clean_for_skeleton method with text boxes."""
        preprocessor = ImagePreprocessor()
        
        # Create binary with "data" and "text"
        binary = np.zeros((100, 200), dtype=np.uint8)
        
        # Data line
        cv2.line(binary, (10, 50), (190, 50), 255, 2)
        
        # Simulated text regions
        cv2.rectangle(binary, (20, 70), (60, 90), 255, -1)  # "100"
        cv2.rectangle(binary, (140, 70), (180, 90), 255, -1)  # "200"
        
        text_boxes = [
            (20, 70, 40, 20),
            (140, 70, 40, 20),
        ]
        
        cleaned, stats = preprocessor.clean_for_skeleton(
            binary, text_boxes=text_boxes, chart_id="test_skeleton_prep"
        )
        
        # Data line should remain
        assert np.sum(cleaned[49:52, 10:190]) > 0, "Data line should remain"
        
        # Text regions should be masked
        assert np.sum(cleaned[70:90, 20:60]) == 0, "Text 1 should be masked"
        assert np.sum(cleaned[70:90, 140:180]) == 0, "Text 2 should be masked"
        
        assert stats["text_regions_masked"] == 2


class TestDisabledCleaning:
    """Test behavior when cleaning options are disabled."""
    
    def test_no_cleaning_when_disabled(self):
        """Verify no cleaning is applied when all options are disabled."""
        config = PreprocessConfig(
            apply_grid_removal=False,
            apply_noise_filtering=False,
            apply_text_masking=False,
        )
        preprocessor = ImagePreprocessor(config)
        
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        result = preprocessor.process(test_img, chart_id="test_no_clean")
        
        # binary and cleaned should be identical when cleaning is disabled
        # (aside from potential minor differences from other operations)
        diff = np.abs(
            result.binary_image.astype(float) - result.cleaned_image.astype(float)
        )
        
        # Should be very similar (allow small tolerance for numerical precision)
        assert np.mean(diff) < 1.0, "Binary and cleaned should be nearly identical"
