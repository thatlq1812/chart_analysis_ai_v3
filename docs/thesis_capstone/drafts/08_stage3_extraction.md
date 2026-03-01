# Stage 3: Structural Analysis (Extraction)

## 1. Architecture

### 1.1. Responsibility
The most complex stage (~5,400 lines across 7 submodules). Extracts structured metadata from cropped chart images using a **purely geometric approach** -- no LLM dependency.

### 1.2. Position in Pipeline
```
Stage2Output(List[DetectedChart]) --> [Stage 3: Extraction] --> Stage3Output(List[RawMetadata])
                                                                      |
                                                                      v
                                                                Stage 4: Reasoning
```

### 1.3. Core Design Principle
> "Treat charts as collections of geometric entities, not pixel patterns."

Charts are decomposed into measurable structures: axes, tick marks, bars, lines, data points. Each is localized, classified, and calibrated using classical computer vision and geometry.

### 1.4. Submodule Pipeline (7 modules, executed in sequence)

```
Cropped Chart Image
  |
  v
[1. Preprocessor]     -- Negative transform, adaptive threshold, denoise
  |
  v (Binary image)
[2. Skeletonizer]     -- Lee algorithm, junction detection, distance transform
  |
  v (1-pixel skeleton + keypoints)
[3. Vectorizer]       -- RDP simplification, curve fitting, sub-pixel refinement
  |
  v (Polylines + vertices)
[4. OCR Engine]       -- PaddleOCR, text role classification (title/axis/legend/value)
  |
  v (OCRText[] with roles)
[5. Element Detector] -- Bar detection (watershed+projection), markers (Hough), pie slices
  |
  v (ChartElement[])
[6. Geometric Mapper] -- Axis calibration (RANSAC), pixel-to-value mapping
  |
  v (AxisInfo with calibration)
[7. Classifier]       -- ResNet-18 (primary) -> ML Classifier -> Rule-based (cascade)
  |
  v
RawMetadata {chart_type, texts[], elements[], axis_info, confidence}
```

## 2. Configuration Parameters

### 2.1. Master Config (ExtractionConfig)
| Parameter | Default | Description |
| --- | --- | --- |
| `ocr_engine` | "easyocr" | OCR backend: easyocr, paddleocr, tesseract |
| `enable_vectorization` | True | Enable skeleton-based vectorization |
| `enable_element_detection` | True | Enable bar/marker/slice detection |
| `enable_ocr` | True | Enable text extraction |
| `enable_classification` | True | Enable chart type classification |
| `use_color_segmentation` | True | Color-based element detection |

### 2.2. Submodule Configs
Each submodule has its own Pydantic config: `PreprocessConfig`, `SkeletonConfig`, `VectorizeConfig`, `OCRConfig`, `ElementDetectorConfig`, `MapperConfig`, `ClassifierConfig`.

Source: `config/pipeline.yaml` under `extraction:`.

## 3. Algorithms (per Submodule)

### 3.1. Preprocessor (568 lines)
**Purpose**: Convert chart image to optimized binary mask.

**Negative Image Transform**:
- Invert image: $I_{neg} = 255 - I$
- Rationale: Chart strokes become bright foreground on dark background, aligning with morphological operations (erosion/dilation expect bright foreground)

**Adaptive Thresholding**:
- Gaussian method: $T(x,y) = \mu_{G}(x,y) - C$
- Block size: 11, C=2 (configurable)
- Handles non-uniform lighting in scanned documents

**Denoising**: Non-local means with edge preservation.

### 3.2. Skeletonizer (758 lines)
**Purpose**: Reduce thick strokes to 1-pixel width while preserving topology.

**Lee Algorithm** (topology-preserving thinning):
- Iteratively removes border pixels while maintaining 8-connectivity
- Homotopy-preserving: does not create or destroy holes/junctions
- Uses `skimage.morphology.skeletonize`

**Gap Filling**: Morphological closing (`kernel=3x3`) before skeletonization fills small breaks in chart lines.

**Junction Detection**: Classifies keypoints as:
- Endpoint (1 neighbor)
- Junction (3+ neighbors)
- Intermediate (2 neighbors)

**Distance Transform**: Estimates original stroke width at each skeleton pixel.

### 3.3. Vectorizer (1,124 lines)
**Purpose**: Convert pixel paths to mathematical polylines.

**Ramer-Douglas-Peucker (RDP) Algorithm**:
- Recursively simplifies polyline by removing points within tolerance $\epsilon$
- **Adaptive epsilon**: $\epsilon = \epsilon_{base} \cdot \sqrt{L / L_{ref}}$ where $L$ = polyline length, $L_{ref}$ = reference length
- Preserves data point accuracy while reducing noise

**Sub-pixel Refinement**: Uses weighted centroid of neighborhood for fractional pixel positions.

**Curve Fitting** (for line/area charts):
- Circle/arc fitting for pie charts
- Ellipse fitting
- B-spline smoothing

**Line Style Detection**: Morphological profile analysis for solid/dashed/dotted lines.

### 3.4. OCR Engine (~800 lines)
**Purpose**: Extract all text with spatial role classification.

**Engine**: PaddleOCR (primary), with EasyOCR as alternative.

**Spatial Role Detection** (region-based):
| Role | Region | Criteria |
| --- | --- | --- |
| Title | Top 15% of image | `y < 0.15 * height` |
| X-axis Label | Bottom 15% | `y > 0.85 * height` |
| Y-axis Label | Left 15% | `x < 0.15 * width` |
| Legend | Top-right corner | `x > 0.7, y < 0.3` |
| Value | Near data elements | Proximity-based |

**Content-aware Enhancement**: Keyword-based classification (e.g., text containing "Figure" assigned as title).

**OCR Post-processing**: Common error corrections:
- `loo` -> `100`, `O` -> `0`, `l` -> `1`, `S` -> `5`
- Merging nearby text fragments (threshold: 20px)

### 3.5. Element Detector (~600 lines)
**Purpose**: Detect discrete chart elements.

**Bar Detection** (3 methods combined):
1. Watershed segmentation for touching bars
2. Projection profile analysis for horizontal/vertical alignment
3. Morphological connected components

**Marker Detection**: Hough circle transform for scatter plot points.

**Pie Slice Detection**: Contour analysis with arc fitting.

**Color Clustering**: K-means (k=8) for dominant color extraction per element.

### 3.6. Geometric Mapper (999 lines)
**Purpose**: Map pixel coordinates to actual data values.

**Axis Detection**: Hough line transform to find axis lines.

**Tick Detection**: Regular spacing detection along axis lines.

**Calibration Pipeline**:
1. Match OCR tick labels to detected tick positions
2. Build (pixel, value) correspondence pairs
3. Fit calibration model using one of:
   - **RANSAC** (default): Robust to OCR outliers
   - **Theil-Sen**: Median-based, even more robust
   - **Least Squares**: When data is clean
4. Compute $R^2$ as calibration confidence

**Mapping Function** (linear scale):
$$value = \frac{pixel - b}{a}$$
where $a$ (scale factor) and $b$ (offset) come from calibration.

**Y-axis Inversion**: Image coordinates have Y=0 at top; physical values have Y=0 at bottom. The mapper handles this automatically.

**Logarithmic Scale Detection**: If tick values follow geometric progression, switch to log scale.

### 3.7. Classifier (cascade)
**Purpose**: Determine chart type.

**Cascade approach** (3 levels):

| Priority | Classifier | Accuracy | Notes |
| --- | --- | --- | --- |
| 1 | ResNet-18 | **94.14%** | CNN on 256x256 grayscale, 8 classes |
| 2 | ML Classifier | ~70% | Random Forest on structural features |
| 3 | Rule-based | ~50% | Heuristics (has bars? has slices?) |

**ResNet-18 Details**:
- Input: 256x256 grayscale (preprocessed)
- Classes: area, bar, box, heatmap, histogram, line, pie, scatter
- Training: 32,445 images, 50 epochs, RTX 3060
- ONNX export: 42.64 MB, 6.90ms mean inference (CPU)

## 4. Output Schema: RawMetadata

```python
class RawMetadata:
    chart_id: str           # From Stage 2
    chart_type: ChartType   # Classified type (8 values)
    texts: List[OCRText]    # All extracted text with roles
    elements: List[ChartElement]  # Detected visual elements
    axis_info: Optional[AxisInfo] # Calibrated axis information
    confidence: ExtractionConfidence  # 4-component weighted score
    warnings: List[str]     # Quality warnings
```

**Confidence Scoring** (weighted average):
$$C_{overall} = 0.3 \cdot C_{class} + 0.25 \cdot C_{ocr} + 0.25 \cdot C_{axis} + 0.2 \cdot C_{elem}$$

## 5. Results

| Metric | Value | Source |
| --- | --- | --- |
| Classification accuracy | **94.14%** (ResNet-18) | MASTER_CONTEXT v3.0.0 |
| ONNX inference speed | **6.90ms** mean (CPU) | MASTER_CONTEXT v3.0.0 |
| Throughput (ONNX) | **144.9 img/sec** | MASTER_CONTEXT v3.0.0 |
| OCR confidence | 91.5% mean | WEEKLY_PROGRESS_20260129 |
| Overall extraction confidence | **92.6%** | WEEKLY_PROGRESS_20260129 |
| Batch extraction | **32,364/32,364** charts, **0% error** | WEEKLY_PROGRESS_20260301 |
| Axis info coverage | **69.9%** of charts | data_pipeline_report_v1 |
| Test suite | **139/140 passing** | MASTER_CONTEXT v3.0.0 |

### Per-class Classification Accuracy

| Class | Accuracy | Samples |
| --- | --- | --- |
| pie | 98.8% | 2,421 |
| bar | 95.3% | 9,086 |
| heatmap | 94.9% | 680 |
| line | 94.2% | 10,036 |
| scatter | 93.3% | 2,802 |
| histogram | 91.2% | 2,060 |
| area | 90.5% | 493 |
| box | 89.8% | 4,867 |

## 6. Lessons Learned

1. **Negative image transform** is critical: aligns morphological operations with chart structure, improving skeleton quality by ~30%.
2. **RANSAC calibration** handles OCR errors gracefully: up to 40% outlier rejection without degrading calibration quality.
3. **Cascade classification** provides both accuracy and robustness: ResNet handles common cases, ML/rules handle edge cases.
4. **Color-based element detection** outperforms binary contour analysis for charts with multiple series.
5. **Adaptive RDP epsilon** prevents over-simplification of long curves and under-simplification of short ones.

## 7. Limitations

- Axis calibration requires at least 2 readable tick labels -- charts without visible axis numbers get `axis_info=None` (30.1% of dataset)
- OCR struggles with rotated Y-axis labels and overlapping text
- Element detector assumes standard chart layouts -- unusual designs (3D charts, multi-panel) may fail
- No semantic understanding of legend entries (deferred to Stage 4)
- Processing speed: ~7.6s per chart (optimized from initial 14.6s)
