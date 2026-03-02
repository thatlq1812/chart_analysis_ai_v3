#!/usr/bin/env python
"""
Test Element Detector with Color Segmentation Fix

Tests that the ElementDetector correctly detects colored bars
using HSV saturation-based color segmentation.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core_engine.stages.s3_extraction.element_detector import (
    ElementDetector, 
    ElementDetectorConfig,
)


def create_simple_bar_chart():
    """Create a simple bar chart with 4 colored bars."""
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (60, 50), (60, 350), (0, 0, 0), 2)
    cv2.line(img, (60, 350), (460, 350), (0, 0, 0), 2)
    
    # Draw 4 bars with different colors
    bars = [
        (100, 200, (66, 133, 244)),   # Blue
        (180, 280, (52, 168, 83)),    # Green
        (260, 150, (251, 188, 5)),    # Yellow
        (340, 320, (234, 67, 53)),    # Red
    ]
    
    for x, height, color in bars:
        cv2.rectangle(img, (x, 350 - height), (x + 50, 350), color, -1)
    
    # Add title
    cv2.putText(img, "Quarterly Sales 2025", (140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def create_grouped_bar_chart():
    """Create a grouped bar chart with 2 groups of 3 bars each.
    
    Note: Bars are spaced further apart to avoid merge issues.
    """
    img = np.ones((400, 700, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (60, 50), (60, 350), (0, 0, 0), 2)
    cv2.line(img, (60, 350), (660, 350), (0, 0, 0), 2)
    
    # Group 1: Q1 (bars with 10px gap between them)
    cv2.rectangle(img, (100, 150), (140, 350), (66, 133, 244), -1)   # Blue
    cv2.rectangle(img, (150, 200), (190, 350), (52, 168, 83), -1)    # Green
    cv2.rectangle(img, (200, 250), (240, 350), (234, 67, 53), -1)    # Red
    
    # Group 2: Q2 (bars with 10px gap between them)
    cv2.rectangle(img, (350, 100), (390, 350), (66, 133, 244), -1)   # Blue
    cv2.rectangle(img, (400, 180), (440, 350), (52, 168, 83), -1)    # Green
    cv2.rectangle(img, (450, 220), (490, 350), (234, 67, 53), -1)    # Red
    
    # Add title
    cv2.putText(img, "Product Sales by Quarter", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def create_horizontal_bar_chart():
    """Create a horizontal bar chart."""
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (80, 50), (80, 350), (0, 0, 0), 2)   # Y-axis
    cv2.line(img, (80, 350), (450, 350), (0, 0, 0), 2) # X-axis
    
    # Draw horizontal bars
    bars = [
        (70, 200, (66, 133, 244)),    # Blue
        (130, 300, (52, 168, 83)),    # Green
        (190, 150, (251, 188, 5)),    # Yellow
        (250, 350, (234, 67, 53)),    # Red
    ]
    
    for y, width, color in bars:
        cv2.rectangle(img, (80, y), (80 + width, y + 40), color, -1)
    
    # Add title
    cv2.putText(img, "Horizontal Bar Chart", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def create_thin_bars_chart():
    """Create a chart with thin bars (stress test for aspect ratio)."""
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (40, 50), (40, 350), (0, 0, 0), 2)
    cv2.line(img, (40, 350), (460, 350), (0, 0, 0), 2)
    
    # Draw thin bars (width=15, well-spaced at 50px apart)
    bars = [
        (60, 250, (66, 133, 244)),     # Blue
        (110, 280, (52, 168, 83)),     # Green
        (160, 200, (251, 188, 5)),     # Yellow
        (210, 320, (234, 67, 53)),     # Red
        (260, 150, (128, 0, 128)),     # Purple
        (310, 180, (0, 128, 128)),     # Teal
        (360, 220, (255, 165, 0)),     # Orange
    ]
    
    for x, height, color in bars:
        cv2.rectangle(img, (x, 350 - height), (x + 15, 350), color, -1)
    
    cv2.putText(img, "Thin Bars Chart", (160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def create_chart_with_grid():
    """Create a bar chart with background grid lines."""
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    
    # Draw grid lines
    for y in range(50, 351, 50):
        cv2.line(img, (60, y), (460, y), (200, 200, 200), 1)
    
    # Draw axes
    cv2.line(img, (60, 50), (60, 350), (0, 0, 0), 2)
    cv2.line(img, (60, 350), (460, 350), (0, 0, 0), 2)
    
    # Draw bars
    bars = [
        (100, 200, (66, 133, 244)),
        (180, 250, (52, 168, 83)),
        (260, 180, (251, 188, 5)),
        (340, 300, (234, 67, 53)),
    ]
    
    for x, height, color in bars:
        cv2.rectangle(img, (x, 350 - height), (x + 50, 350), color, -1)
    
    cv2.putText(img, "Chart with Grid", (180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def create_chart_with_legend():
    """Create a bar chart with legend box."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (60, 50), (60, 350), (0, 0, 0), 2)
    cv2.line(img, (60, 350), (460, 350), (0, 0, 0), 2)
    
    # Draw bars
    bars = [
        (100, 200, (66, 133, 244), "Product A"),
        (180, 250, (52, 168, 83), "Product B"),
        (260, 180, (251, 188, 5), "Product C"),
        (340, 300, (234, 67, 53), "Product D"),
    ]
    
    for x, height, color, _ in bars:
        cv2.rectangle(img, (x, 350 - height), (x + 50, 350), color, -1)
    
    # Draw legend box
    cv2.rectangle(img, (480, 50), (590, 180), (0, 0, 0), 1)
    
    # Legend items (small color squares)
    for i, (_, _, color, label) in enumerate(bars):
        y = 70 + i * 25
        cv2.rectangle(img, (490, y - 8), (505, y + 7), color, -1)
        cv2.putText(img, label, (515, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.putText(img, "Sales Report", (180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def run_test(name: str, img: np.ndarray, expected_bars: int, min_bar_area: int = 100) -> bool:
    """Run detection test on an image."""
    print(f"\n{'='*50}")
    print(f"Test: {name}")
    print(f"{'='*50}")
    
    # Create binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Configure detector with color segmentation
    config = ElementDetectorConfig(
        use_color_segmentation=True,
        color_saturation_threshold=30,
        min_bar_area=min_bar_area,  # Configurable min area
        max_bar_area_ratio=0.4,
    )
    detector = ElementDetector(config)
    
    # Run detection
    result = detector.detect(binary, img, f"test_{name}")
    
    print(f"Detected {len(result.bars)} bars:")
    for i, bar in enumerate(result.bars):
        w = bar.x_max - bar.x_min
        h = bar.y_max - bar.y_min
        color_str = f"RGB({bar.color.r},{bar.color.g},{bar.color.b})" if bar.color else "None"
        print(f"  Bar {i+1}: {w:.0f}x{h:.0f} @ ({bar.x_min:.0f},{bar.y_min:.0f}) color={color_str}")
    
    passed = len(result.bars) == expected_bars
    print(f"\nExpected: {expected_bars} bars, Got: {len(result.bars)}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    
    return passed


def main():
    """Run all tests."""
    print("="*60)
    print("Element Detector Test Suite - Color Segmentation Fix")
    print("="*60)
    
    tests = [
        ("Simple Bar Chart (4 vertical bars)", create_simple_bar_chart(), 4, 100),
        ("Grouped Bar Chart (6 bars)", create_grouped_bar_chart(), 6, 100),
        ("Horizontal Bar Chart (4 bars)", create_horizontal_bar_chart(), 4, 100),
        ("Thin Bars Chart (7 bars)", create_thin_bars_chart(), 7, 50),  # Lower threshold for thin bars
        ("Chart with Grid (4 bars)", create_chart_with_grid(), 4, 100),
        ("Chart with Legend (4 bars, filter legend)", create_chart_with_legend(), 4, 500),  # Higher threshold to filter legend items
    ]
    
    results = []
    for name, img, expected, min_area in tests:
        passed = run_test(name, img, expected, min_area)
        results.append((name, passed))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
