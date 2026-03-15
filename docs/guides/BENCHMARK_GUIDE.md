# Stage 3 Benchmark - Historical Ceiling Experiment

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.1.0 | 2026-03-12 | That Le | Marked historical; result = FAIL -> VLM rewrite (v6.0.0) |
| 1.0.0 | 2026-03-04 | That Le | Initial benchmark infrastructure |

> **STATUS: COMPLETED (HISTORICAL)** — This benchmark evaluated the geometry-based Stage 3 pipeline.
> Result: **FAIL** (0/40 axis detection, 0 OCR output on 50 real-world charts).
> This result directly motivated the VLM rewrite in v6.0.0.
> The current Stage 3 uses DePlot/MatCha/Pix2Struct/SVLM VLM backends.

## 1. Purpose

This benchmark measured the **geometric ceiling** of the original Stage 3 -- the maximum
accuracy achievable by the computer vision + geometry pipeline (pre-v6.0.0) BEFORE AI reasoning
(Stage 4) corrected errors.

**Key question asked**: Can the geometric approach achieve >= 80% accuracy on core metrics?
- If YES: Focus on SLM training to fix remaining errors
- If NO: Redesign Stage 3 components before investing in SLM

**Answer**: NO (FAIL) — the benchmark revealed that geometry fundamentally fails on real-world
academic charts. OCR produced zero output (`texts: []`) for all 50 charts, and axis detection
achieved 0/40 (0%) accuracy on non-pie charts. See CHANGELOG [6.0.0] for full details.

## 2. Benchmark Composition

50 manually annotated charts, stratified by type and difficulty:

| Type | Simple | Moderate | Complex | Total |
| --- | --- | --- | --- | --- |
| bar | 5 | 3 | 2 | 10 |
| line | 5 | 3 | 2 | 10 |
| pie | 5 | 3 | 2 | 10 |
| scatter | 5 | 3 | 2 | 10 |
| area | 2 | 1 | 0 | 3 |
| histogram | 2 | 1 | 0 | 3 |
| box | 1 | 1 | 0 | 2 |
| heatmap | 1 | 1 | 0 | 2 |
| **Total** | **26** | **14** | **10** | **50** |

**Difficulty criteria** (heuristic, based on element count):
- **Simple**: Few elements (<=6 bars, <=10 line points, <=5 pie slices, <=20 scatter points)
- **Moderate**: Medium complexity (multi-series, some visual clutter)
- **Complex**: Many elements, overlapping, log-scale, stacked/grouped

## 3. Directory Structure

```
data/benchmark/
    benchmark_manifest.json     # Metadata for all 50 charts
    images/                     # Copied chart images (50 PNG files)
    annotations/                # Ground truth JSON (1 per chart)
    results/
        stage3_outputs/         # Cached Stage 3 extraction results
        evaluation_report.json  # Detailed evaluation metrics
        evaluation_report.md    # Human-readable report
```

## 4. Annotation Schema

Each chart has a JSON annotation with these sections:

### 4.1. Identity and Classification
```json
{
    "chart_id": "arxiv_2504_05445v1_page_6_img_39",
    "image_path": "data/academic_dataset/classified_charts/bar/...",
    "chart_type": "bar",
    "difficulty": "simple"
}
```

### 4.2. Complexity Traits
```json
{
    "complexity_traits": {
        "is_stacked": false,
        "is_grouped": false,
        "is_multi_series": false,
        "has_log_scale": false,
        "has_rotated_labels": false,
        "has_dense_data": false
    }
}
```

### 4.3. Text Annotations
```json
{
    "title": "Training Loss vs Epochs",
    "texts": [
        {"text": "Training Loss vs Epochs", "role": "title"},
        {"text": "Epochs", "role": "x_axis_label"},
        {"text": "Loss", "role": "y_axis_label"},
        {"text": "0", "role": "y_tick"},
        {"text": "100", "role": "y_tick"},
        {"text": "Model A", "role": "legend"}
    ]
}
```

### 4.4. Element Counts
```json
{
    "elements": {
        "primary_element_type": "bar",
        "element_count": 5,
        "series_count": 1,
        "has_grid_lines": true,
        "has_legend": false,
        "has_data_labels": false
    }
}
```

### 4.5. Axis Information
```json
{
    "axis": {
        "x_axis_type": "categorical",
        "y_axis_type": "linear",
        "x_min": null,
        "x_max": null,
        "y_min": 0.0,
        "y_max": 100.0,
        "x_categories": ["A", "B", "C", "D", "E"],
        "x_label": "Category",
        "y_label": "Value"
    }
}
```

### 4.6. Data Series (Gold Standard)
```json
{
    "data_series": [
        {
            "name": null,
            "points": [
                {"x": "A", "y": 45.0},
                {"x": "B", "y": 72.0},
                {"x": "C", "y": 63.0}
            ]
        }
    ]
}
```

## 5. Evaluation Metrics

### 5.1. Classification Accuracy
- Binary: predicted type matches ground truth?
- Normalized: stacked_bar/grouped_bar count as "bar", donut counts as "pie"

### 5.2. Element Count Accuracy
- Within 25% tolerance: |predicted - actual| / actual <= 0.25
- Relative error distribution

### 5.3. Element Type Accuracy
- Does the primary element type match? (bar, point, slice, line, area)

### 5.4. Axis Range Accuracy
- Within 15% relative error for each of x_min, x_max, y_min, y_max
- Only evaluated for numeric axes (not categorical)

### 5.5. OCR Recall (by role)
- Title recall
- Tick label recall
- Axis label recall

### 5.6. Ceiling Verdict

| Result | Classification | Elements | Axis |
| --- | --- | --- | --- |
| PASS | >= 90% | >= 70% (+-25%) | >= 60% (+-15%) |
| PARTIAL | 2/3 pass | - | - |
| FAIL | < 2/3 pass | - | - |

## 6. Workflow

### Step 1: Sample (already done)
```bash
.venv/Scripts/python.exe scripts/evaluation/benchmark/stratified_sampler.py
```

### Step 2: Run Stage 3 (generates cached outputs)
```bash
# Without OCR (fast, ~3 min):
.venv/Scripts/python.exe scripts/evaluation/benchmark/evaluate.py --ocr none

# With OCR (slow, ~20 min):
.venv/Scripts/python.exe scripts/evaluation/benchmark/evaluate.py --ocr easyocr
```

### Step 3: Pre-fill annotations
```bash
.venv/Scripts/python.exe scripts/evaluation/benchmark/prefill_annotations.py
```

### Step 4: Manually annotate
Open each chart image alongside its annotation JSON and correct:
- chart_type (verify classification)
- element_count (count bars/points/slices manually)
- axis ranges (read from chart)
- text content and roles

### Step 5: Re-evaluate with annotations
```bash
.venv/Scripts/python.exe scripts/evaluation/benchmark/evaluate.py --skip-run
```

### Step 6: With OCR (full evaluation)
```bash
.venv/Scripts/python.exe scripts/evaluation/benchmark/evaluate.py --ocr easyocr
```

## 7. Interpretation Guide

### What "PASS" means (historical)
The geometry pipeline alone could extract reasonably accurate data from charts.
SLM training would focus on error correction (OCR cleanup, value mapping).

### What "PARTIAL" means (historical)
Some chart types or components needed targeted fixes.

### What "FAIL" means (historical)
The geometry approach had fundamental limitations. This is the result that was observed:
- OCR produced 0 output texts for all 50 charts
- Axis detection achieved 0% on 40 non-pie charts
- **Action taken**: Stage 3 was rewritten using VLM extraction (v6.0.0)
