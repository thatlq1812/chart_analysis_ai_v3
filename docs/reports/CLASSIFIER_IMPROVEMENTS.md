# Chart Classifier Improvements

| Date | Author | Status |
| --- | --- | --- |
| 2025-01-21 | That Le | Completed |

## Summary

Improved chart type classification from **0% accuracy** (all UNKNOWN) to **70% accuracy** using ML-based approach with Random Forest.

## Problem Analysis

### Original Issue

The rule-based classifier was classifying all chart types as "unknown" or "line" because:

1. **Element detection was too weak**: Bar detector found 0-4 bars even on bar charts
2. **Binary preprocessing was inverted**: White background detected as foreground
3. **Pie detection failed**: Hough circles gave false positives
4. **Feature scoring thresholds were incorrect**: All scores below `min_confidence=0.5`

### Root Cause

The original approach relied on detecting discrete elements (bars, slices, markers) from preprocessed binary images. However:

- Binary thresholding didn't work well on academic charts with white backgrounds
- Skeleton-based analysis designed for thin strokes, not filled bars
- Element detection thresholds were not tuned for real-world charts

## Solution

### Approach 1: Simple Image-based Classifier (Intermediate)

Created `simple_classifier.py` using image-level features:

- **Edge orientation** (horizontal, vertical, diagonal ratios)
- **Color distribution** (number of colors, coverage)
- **Circularity detection** (contour-based)
- **Grid pattern detection**
- **Marker count** (small blobs)
- **Rectangle detection**

**Result**: 15.6% accuracy (only Bar and Scatter improved)

### Approach 2: ML-based Classifier (Final)

Created `ml_classifier.py` with Random Forest:

1. **Feature Extraction**: Use SimpleChartClassifier to extract 11 features
2. **Training**: Random Forest with 100 trees, balanced class weights
3. **Dataset**: 872 images from ChartQA dataset (100 per type)

**Result**: **70.3% test accuracy**

## Performance Metrics

### Per-Class Accuracy (Random Forest)

| Chart Type | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| area | 0.95 | 1.00 | 0.98 |
| pie | 0.94 | 0.85 | 0.89 |
| heatmap | 0.89 | 0.80 | 0.84 |
| histogram | 0.48 | 0.70 | 0.57 |
| line | 0.65 | 0.65 | 0.65 |
| scatter | 0.59 | 0.65 | 0.62 |
| bar | 0.58 | 0.55 | 0.56 |
| box | 0.75 | 0.60 | 0.67 |
| other | 0.62 | 0.50 | 0.56 |

### Feature Importance

| Feature | Importance |
| --- | --- |
| color_coverage | 0.2058 |
| n_markers | 0.1248 |
| rect_coverage | 0.1192 |
| grid_score | 0.1130 |
| v_edge_ratio | 0.1039 |
| h_edge_ratio | 0.0950 |
| n_colors | 0.0752 |
| d_edge_ratio | 0.0583 |
| circularity | 0.0580 |

## Files Modified/Created

### New Files

- `src/core_engine/stages/s3_extraction/simple_classifier.py` - Image-based feature extraction
- `src/core_engine/stages/s3_extraction/ml_classifier.py` - Random Forest wrapper
- `scripts/train_classifier.py` - Training script
- `scripts/benchmark_simple_classifier.py` - Evaluation script
- `models/weights/chart_classifier_rf.pkl` - Trained model

### Modified Files

- `src/core_engine/stages/s3_extraction/s3_extraction.py` - Added ML classifier integration

## Usage

```python
from src.core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig

# Enable ML classifier (default)
config = ExtractionConfig(use_ml_classifier=True)
stage = Stage3Extraction(config)

# Process image
result = stage.process_image(image_bgr, chart_id="test")
print(f"Detected type: {result.chart_type.value}")
```

## Future Improvements

1. **More training data**: Current 872 images, could expand to full 2852
2. **Deep learning**: Train CNN-based classifier for potentially higher accuracy
3. **Ensemble**: Combine rule-based and ML predictions
4. **Cross-validation**: Use k-fold for more robust evaluation
5. **Feature engineering**: Add more discriminative features (texture, frequency)
