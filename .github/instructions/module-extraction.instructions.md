---
applyTo: 'src/core_engine/stages/s3_extraction/**,models/weights/efficientnet_b0*,models/weights/chart_classifier*'
---

# MODULE INSTRUCTIONS - Chart Extraction (Stage 3)

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.1.0 | 2026-03-12 | That Le | Classifier upgraded to EfficientNet-B0 3-class (97.54%), updated cascade and model details |
| 1.0.0 | 2026-02-28 | That Le | OCR, classification, and geometric extraction pipeline |

---

## 1. Overview

**Stage 3 Extraction** is the largest and most complex stage. It processes cropped chart images and extracts structured metadata: chart type, text (OCR), data elements (bars, lines, points), axis calibration, and coordinate mapping.

**Key Directory:** `src/core_engine/stages/s3_extraction/` (12 files, ~9,000 lines total)

---

## 2. Submodule Architecture

### 2.1. Processing Pipeline

```
Input:  Cropped chart image (from Stage 2)
  ↓
  ┌─────────────────────────────────────────────────────┐
  │ 1. preprocessor.py (568 lines)                       │
  │    - Negative image transform                        │
  │    - Adaptive thresholding, denoising                │
  │    - CLAHE contrast enhancement                      │
  │    - Text masking, grid removal                      │
  │    - Top-hat morphological transform                 │
  ├─────────────────────────────────────────────────────┤
  │ 2. Chart Classification (parallel to preprocessing)  │
  │    ├─ resnet_classifier.py (283L) ← LEGACY (ResNet-18 8-class)  │
  │    ├─ train_chart_classifier.py output → efficientnet_b0_3class_v1_best.pt [PRIMARY]  │
  │    ├─ classifier.py (406L) ← Rule-based fallback     │
  │    ├─ ml_classifier.py (195L) ← Random Forest        │
  │    └─ simple_classifier.py (988L) ← Feature extract  │
  ├─────────────────────────────────────────────────────┤
  │ 3. ocr_engine.py (1,011 lines)                       │
  │    - PaddleOCR (default) / EasyOCR / Tesseract       │
  │    - Confidence filtering                            │
  │    - Spatial role detection (title, axis, legend)     │
  ├─────────────────────────────────────────────────────┤
  │ 4. element_detector.py (1,738 lines)                 │
  │    - Bar detection (watershed + projection + morph)   │
  │    - Marker detection (Hough circles)                │
  │    - Pie slice detection                             │
  │    - Color extraction                                │
  ├─────────────────────────────────────────────────────┤
  │ 5. skeletonizer.py (758 lines)                       │
  │    - Lee/Zhang thinning algorithm                    │
  │    - Junction detection, spur removal                │
  │    - Topology-preserving operations                  │
  ├─────────────────────────────────────────────────────┤
  │ 6. vectorizer.py (1,124 lines)                       │
  │    - Ramer-Douglas-Peucker simplification            │
  │    - Curvature-adaptive epsilon                      │
  │    - Circle/ellipse/spline fitting                   │
  ├─────────────────────────────────────────────────────┤
  │ 7. geometric_mapper.py (999 lines)                   │
  │    - Pixel → data coordinate mapping                 │
  │    - RANSAC / Theil-Sen robust fitting               │
  │    - Hough transform axis detection                  │
  │    - Linear / logarithmic scale support              │
  └─────────────────────────────────────────────────────┘
  ↓
Output: RawMetadata (chart_type, texts, elements, axis_calibration)
```

### 2.2. Submodule Summary

| File | Lines | Class | Config Model | Purpose |
| --- | --- | --- | --- | --- |
| `s3_extraction.py` | 889 | `Stage3Extraction` | `ExtractionConfig` | Main orchestrator |
| `preprocessor.py` | 568 | `ImagePreprocessor` | `PreprocessConfig` | Image preprocessing |
| `ocr_engine.py` | 1,011 | `OCREngine` | `OCRConfig` | Multi-engine OCR |
| `element_detector.py` | 1,738 | `ElementDetector` | `ElementDetectorConfig` | Visual element detection |
| `geometric_mapper.py` | 999 | `GeometricMapper` | `MapperConfig` | Coordinate calibration |
| `skeletonizer.py` | 758 | `Skeletonizer` | `SkeletonConfig` | Line thinning |
| `vectorizer.py` | 1,124 | `Vectorizer` | `VectorizeConfig` | Curve fitting |
| `resnet_classifier.py` | 283 | `ResNet18Classifier` | - | Legacy 8-class DL classifier (use EfficientNet-B0 for new work) |
| `classifier.py` | 406 | `ChartClassifier` | `ClassifierConfig` | Rule-based classifier |
| `ml_classifier.py` | 195 | `MLChartClassifier` | - | Random Forest classifier |
| `simple_classifier.py` | 988 | `SimpleChartClassifier` | `SimpleClassifierConfig` | Feature extraction |
| `__init__.py` | 53 | - | - | Package exports |

---

## 3. Chart Classification

### 3.1. Classification Cascade

```
Image → EfficientNet-B0 (97.54% accuracy, 3 classes: bar/line/pie)
           ↓ confidence < 0.7 OR type = unknown?
         ML Classifier (Random Forest)
           ↓ confidence < 0.6?
         Rule-based Classifier (structural features)
```

### 3.2. Supported Chart Types (3 production classes)

| Class | EfficientNet-B0 F1 | Key Features |
| --- | --- | --- |
| `bar` | 97.4% | Rectangular contours, parallel alignment |
| `line` | 97.2% | Continuous curves, markers |
| `pie` | 93.5% | Circular region, angular sectors |

Note: 8-class support (scatter, area, histogram, heatmap, box) is handled by the rule-based
fallback cascade and was part of the legacy ResNet-18 model (archived in `models/weights/resnet18_chart_classifier_v2_best.pt`).

### 3.3. EfficientNet-B0 Model Details

| Property | Value |
| --- | --- |
| Architecture | EfficientNet-B0 (transfer learning, torchvision) |
| Input | 224x224, RGB |
| Training accuracy | 97.54% (val + test) |
| Macro F1 | 94.63% |
| Epochs | 21 (early stopping, patience=10) |
| Weights file | `efficientnet_b0_3class_v1_best.pt` |
| Training script | `scripts/training/train_chart_classifier.py` |

---

## 4. OCR Configuration

### 4.1. Engine Selection

Default: **PaddleOCR** (best accuracy + GPU support)

```yaml
# config/pipeline.yaml
extraction:
  ocr:
    min_confidence: 0.5
    engine: "paddleocr"       # or "easyocr", "tesseract"
```

### 4.2. Spatial Role Detection

OCR texts are classified by position in the image:

| Role | Region | Rule |
| --- | --- | --- |
| `title` | Top 15% of image | Largest text in top region |
| `xlabel` | Bottom 15% | Horizontal text below plot area |
| `ylabel` | Left 15% | Vertical/rotated text left of plot |
| `legend` | Top-right quadrant | Text near color patches |
| `value` | Inside plot area | Numeric text near data elements |

### 4.3. OCR Caching

OCR results are cached to avoid re-processing:
- Cache key: image hash + OCR engine + config hash
- Storage: `data/cache/ocr/`
- Invalidation: config change or manual clear

---

## 5. Element Detection Strategies

### 5.1. Bar Detection (Hybrid)

Three-method hybrid for robust bar separation:
1. **Watershed segmentation** - Split touching bars
2. **Projection analysis** - Horizontal/vertical intensity projections
3. **Morphological operations** - Opening/closing to separate elements

### 5.2. Marker Detection

- Hough circle detection for scatter plots
- Template matching for standard markers (star, diamond, triangle)

### 5.3. Pie Slice Detection

- Circle/ellipse fitting (Hough)
- Angular sector segmentation
- Color-based region splitting

---

## 6. Geometric Mapping

### 6.1. Axis Calibration

```
Pixel coordinates → Data coordinates
  ↓
  1. Detect axis lines (Hough transform)
  2. Find tick marks (equally-spaced line segments)
  3. Match OCR values to tick positions
  4. Fit calibration model:
     - Linear: x_data = a * x_pixel + b
     - Logarithmic: x_data = 10^(a * x_pixel + b)
  5. Robust fitting: RANSAC or Theil-Sen (outlier resistant)
```

### 6.2. Scale Support

| Scale | Detection | Fitting |
| --- | --- | --- |
| Linear | Equally-spaced ticks | Linear regression |
| Logarithmic | Geometrically-spaced ticks | Log-linear regression |

---

## 7. Rules

1. **Orchestrator** (`Stage3Extraction`) calls submodules in sequence -- individual submodules don't call each other
2. **Each submodule** has its own Pydantic config -- no shared mutable state
3. **OCR caching** is mandatory for development speed (processing 1,000+ images)
4. **Classification cascade** must be followed -- EfficientNet-B0 first, fallback only on low confidence or unknown type
5. **Never** hardcode image dimensions -- use relative coordinates
6. **All coordinates** in output are normalized [0, 1] unless explicitly labeled as pixel
7. **Color extraction** returns RGB tuples, converted to hex only at display time
8. **Preprocessing** is applied before EVERY submodule that processes images
9. **GPU** used for: EfficientNet-B0 inference, PaddleOCR. CPU for: all OpenCV operations
10. **Element detection** runs AFTER classification (chart type determines which detector to use)

---

## 8. Testing

```bash
# Full Stage 3 tests
.venv/Scripts/python.exe -m pytest tests/ -k "stage3 or extraction" -v

# Individual submodule tests
.venv/Scripts/python.exe -m pytest tests/ -k "ocr" -v
.venv/Scripts/python.exe -m pytest tests/ -k "classifier" -v
.venv/Scripts/python.exe -m pytest tests/ -k "element_detect" -v
```

Test with sample images in `data/samples/` (at least one per chart type).
