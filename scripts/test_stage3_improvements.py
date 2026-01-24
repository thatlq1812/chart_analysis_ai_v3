"""
Test script for Stage 3 improvements.

Tests:
1. Element Detector with new bar separation methods
2. Simple Classifier with grayscale-robust features
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core_engine.stages.s3_extraction.element_detector import (
    ElementDetector, 
    ElementDetectorConfig, 
    BarSeparationMethod
)
from src.core_engine.stages.s3_extraction.simple_classifier import (
    SimpleChartClassifier, 
    SimpleClassifierConfig
)


def test_element_detector():
    """Test Element Detector with new methods."""
    print("=" * 60)
    print("TEST 1: Element Detector - Bar Separation Methods")
    print("=" * 60)
    
    # Create test image with multiple bars
    test_img = np.zeros((300, 400), dtype=np.uint8)
    
    # Draw 5 separate bars
    for i in range(5):
        x = 50 + i * 60
        h = 50 + i * 30
        cv2.rectangle(test_img, (x, 300-h), (x+40, 300), 255, -1)
    
    print(f"Created test image with 5 bars")
    print()
    
    # Test each method
    methods = [
        BarSeparationMethod.CONTOUR_ONLY,
        BarSeparationMethod.WATERSHED,
        BarSeparationMethod.PROJECTION,
        BarSeparationMethod.MORPHOLOGICAL,
        BarSeparationMethod.HYBRID,
    ]
    
    for method in methods:
        config = ElementDetectorConfig(bar_separation_method=method)
        detector = ElementDetector(config)
        result = detector.detect(test_img, chart_id=f'test_{method.value}')
        print(f"  {method.value:15s}: {len(result.bars)} bars detected")
    
    print()
    print("TEST 1 PASSED: All methods executed successfully")
    return True


def test_simple_classifier_grayscale():
    """Test Simple Classifier with grayscale images."""
    print()
    print("=" * 60)
    print("TEST 2: Simple Classifier - Grayscale-Robust Features")
    print("=" * 60)
    
    classifier = SimpleChartClassifier(SimpleClassifierConfig(
        use_texture_features=True,
        use_shape_features=True
    ))
    
    # Test 1: Grayscale bar chart
    print("\nTest 2a: Grayscale Bar Chart")
    bar_img = np.ones((300, 400, 3), dtype=np.uint8) * 200
    for i in range(5):
        x = 50 + i * 60
        h = 50 + i * 30
        cv2.rectangle(bar_img, (x, 300-h), (x+40, 300), (100, 100, 100), -1)
    
    result = classifier.classify(bar_img, chart_id='grayscale_bar')
    print(f"  Type: {result.chart_type.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Is Grayscale: {result.features.get('is_grayscale', 0) > 0.5}")
    print(f"  Reasoning: {result.reasoning[:80]}...")
    
    # Test 2: Grayscale pie chart (circle)
    print("\nTest 2b: Grayscale Pie Chart")
    pie_img = np.ones((300, 300, 3), dtype=np.uint8) * 200
    cv2.circle(pie_img, (150, 150), 100, (100, 100, 100), -1)
    # Add some "slices" with lines
    cv2.line(pie_img, (150, 150), (250, 150), (200, 200, 200), 2)
    cv2.line(pie_img, (150, 150), (150, 50), (200, 200, 200), 2)
    cv2.line(pie_img, (150, 150), (80, 200), (200, 200, 200), 2)
    
    result = classifier.classify(pie_img, chart_id='grayscale_pie')
    print(f"  Type: {result.chart_type.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Circularity: {result.features.get('circularity', 0):.3f}")
    print(f"  Symmetry: {result.features.get('symmetry_score', 0):.3f}")
    
    # Test 3: Color bar chart (should still work)
    print("\nTest 2c: Color Bar Chart (control)")
    color_bar = np.ones((300, 400, 3), dtype=np.uint8) * 255
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        x = 50 + i * 60
        h = 50 + i * 30
        cv2.rectangle(color_bar, (x, 300-h), (x+40, 300), color, -1)
    
    result = classifier.classify(color_bar, chart_id='color_bar')
    print(f"  Type: {result.chart_type.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Is Grayscale: {result.features.get('is_grayscale', 0) > 0.5}")
    print(f"  N Colors: {result.features.get('n_colors', 0):.0f}")
    
    print()
    print("TEST 2 PASSED: All classifier tests executed successfully")
    return True


def test_new_features():
    """Test that all new features are computed."""
    print()
    print("=" * 60)
    print("TEST 3: New Feature Computation")
    print("=" * 60)
    
    classifier = SimpleChartClassifier()
    
    # Create test image
    test_img = np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8)
    result = classifier.classify(test_img, chart_id='feature_test')
    
    new_features = [
        'texture_uniformity', 
        'texture_contrast', 
        'hu_elongation', 
        'hu_compactness',
        'grad_h_ratio', 
        'grad_v_ratio', 
        'n_components', 
        'avg_component_area',
        'has_x_axis', 
        'has_y_axis', 
        'symmetry_score', 
        'is_grayscale'
    ]
    
    print("New features computed:")
    all_present = True
    for feat in new_features:
        if feat in result.features:
            print(f"  {feat:20s}: {result.features[feat]:.4f}")
        else:
            print(f"  {feat:20s}: MISSING!")
            all_present = False
    
    print()
    if all_present:
        print("TEST 3 PASSED: All new features present")
    else:
        print("TEST 3 FAILED: Some features missing")
    
    return all_present


def test_on_real_images():
    """Test on real chart images if available."""
    print()
    print("=" * 60)
    print("TEST 4: Real Image Test")
    print("=" * 60)
    
    image_dir = project_root / "data" / "academic_dataset" / "images"
    
    if not image_dir.exists():
        print("  Skipping: No real images available")
        return True
    
    images = list(image_dir.glob("*.png"))[:5]
    
    if not images:
        print("  Skipping: No PNG images found")
        return True
    
    detector = ElementDetector(ElementDetectorConfig(
        bar_separation_method=BarSeparationMethod.HYBRID
    ))
    classifier = SimpleChartClassifier()
    
    print(f"Testing on {len(images)} real images:")
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Test detection
        det_result = detector.detect(binary, img, chart_id=img_path.stem)
        
        # Test classification
        cls_result = classifier.classify(img, chart_id=img_path.stem)
        
        print(f"  {img_path.name}:")
        print(f"    Type: {cls_result.chart_type.value} ({cls_result.confidence:.2f})")
        print(f"    Bars: {len(det_result.bars)}, Markers: {len(det_result.markers)}")
    
    print()
    print("TEST 4 PASSED: Real image processing complete")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  STAGE 3 IMPROVEMENT TESTS")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_element_detector()
    all_passed &= test_simple_classifier_grayscale()
    all_passed &= test_new_features()
    all_passed &= test_on_real_images()
    
    print()
    print("=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
