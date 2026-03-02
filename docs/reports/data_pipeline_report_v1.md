# Data Pipeline Report — Geo-SLM Chart Analysis

**Version:** 1.0  
**Date:** 2026-03-01  
**Author:** That Le  
**Status:** Final (Stage 1–3 complete; SLM training data ready for v3 build)

---

## Abstract

This report documents the end-to-end data pipeline for the Geo-SLM Chart Analysis project, covering:
(1) automated collection of scientific chart images from arXiv,
(2) multi-stage detection and classification,
(3) geometric feature extraction via Stage 3,
(4) AI-generated QA pair construction, and
(5) the resulting SLM training dataset.

The final corpus consists of **32,364 scientifically sourced chart images** with accompanying structured feature files and **32,445 AI-generated QA pair files**, yielding an SLM training dataset of approximately 45,000–60,000 instruction-following samples enriched with axis calibration, OCR text roles, and element geometry.

---

## 1. Data Collection

### 1.1. Source

All images were collected from **arXiv preprints** (cs.CV, cs.LG, stat.ML, eess.IV) via the arXiv API and bulk S3 download. PDFs were parsed using PyMuPDF to extract individual page images at 150 DPI, then each image was processed by the YOLO detection pipeline.

### 1.2. Collection Scale

| Metric | Value |
|--------|-------|
| arXiv PDFs processed | ~4,000 papers |
| Total raw page images | ~150,000 |
| Detected candidate chart regions | ~70,000 |
| Images after quality filter | **46,910** |
| Final classified charts used | **32,364** |

### 1.3. Quality Filters Applied

| Filter | Criteria | Removed |
|--------|----------|---------|
| Minimum resolution | 100×100 px | ~8,000 |
| Duplicate detection | Perceptual hash | ~3,000 |
| Non-chart categories | `diagram`, `table`, `not_a_chart` | ~11,000 |
| Uncertain classification | ResNet-18 confidence < 0.70 | ~2,546 |

---

## 2. Chart Detection and Classification

### 2.1. Detection — Stage 2 (YOLOv8/v11)

A YOLOv8 model was fine-tuned on a manually annotated subset to localize chart bounding boxes within full-page images. Each detected region was cropped and saved as an independent image.

**Detection model:** YOLOv8m, trained on `data/yolo_chart_detection/`  
**Dataset split:** train/val/test = 70/20/10  
**mAP@0.5:** >0.85 (target met)

### 2.2. Classification — ResNet-18

A ResNet-18 classifier was trained to assign each cropped image to one of 8 chart categories.

| Category | Train count | Accuracy |
|----------|-------------|---------|
| bar | 5,745 | — |
| line | 12,930 | — |
| scatter | 6,278 | — |
| heatmap | 4,073 | — |
| histogram | 1,006 | — |
| box | 880 | — |
| pie | 835 | — |
| area | 617 | — |
| **Total** | **32,364** | **94.66%** |

Overall classification accuracy: **94.66%**, evaluated on the held-out test split.

### 2.3. Class Distribution Analysis

`line` dominates the corpus (39.9%) reflecting the prevalence of training curves and time-series plots in ML/CS papers. `bar`, `scatter`, and `heatmap` together account for a further 49.7%.

```
line      12930  39.9%  ████████████████████
scatter    6278  19.4%  ████████
bar        5745  17.7%  ███████
heatmap    4073  12.6%  █████
histogram  1006   3.1%  █
box         880   2.7%  █
pie         835   2.6%  █
area        617   1.9%
```

---

## 3. Stage 3 Feature Extraction

Stage 3 performs per-chart geometric analysis to extract structured features used as context for AI reasoning and SLM training. It operates offline on cached OCR results.

### 3.1. Pipeline Summary

```
Cropped chart image
    |
OCR (PaddleOCR, read from 589 MB cache — no GPU required)
    |
Text role classification  (title / legend / x_tick / y_tick / data_label)
    |
Axis calibration          (RANSAC + Theil-Sen regression)
    |
Element detection         (contour analysis → bar segments / scatter points)
    |
JSON feature file         (data/academic_dataset/stage3_features/{type}/{id}.json)
```

### 3.2. OCR Cache

All 46,910 images were pre-cached in a single JSON file (`data/cache/ocr_cache.json`, 589 MB) during an earlier collection run. Stage 3 reads exclusively from this cache; PaddleOCR is never invoked during batch extraction. This design decision:

- Eliminates GPU dependency for the extraction phase
- Enables pure CPU parallelism (`--workers 8 --no-gpu`)
- Reduces wall-clock time from ~hours (GPU-gated) to ~2 hours (CPU, 8 workers)

**Throughput:** ~4 images/second on a 32 GB RAM machine using 8 workers (vs. ~1.5–2 img/s with 3 GPU workers in the first run).

### 3.3. Critical Bug Discovered and Fixed

During Run 1 (2 GPU workers), **58.9% of output files were corrupt** (6,543/11,068 files). Root cause analysis identified two defects:

**Bug 1 — `np.int64` serialization failure (primary cause)**

```python
# BEFORE (geometric_mapper.py ~L719, ~L803)
outliers_removed = n - best_num_inliers          # type: np.int64
outliers_removed = n - np.sum(inliers)           # type: np.int64

# json.dump raises TypeError: Object of type int64 is not JSON serializable
# File write aborts mid-stream → truncated/malformed JSON survives on disk
```

```python
# AFTER
outliers_removed = int(n - best_num_inliers)     # type: int
outliers_removed = int(n - np.sum(inliers))      # type: int
```

**Bug 2 — Non-atomic file write (secondary cause)**

Truncated files from Bug 1 were saved to the intended output path, causing `--skip-existing` logic to treat them as complete on restart.

```python
# BEFORE
with open(output_path, "w") as f:
    json.dump(output_data, f)                    # truncated file survives on error

# AFTER — atomic write via OS rename
tmp_path = output_path.with_suffix(".tmp")
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, cls=_NumpyEncoder, indent=2)
os.replace(tmp_path, output_path)               # atomic: either complete or absent
```

A `_NumpyEncoder(json.JSONEncoder)` class was also added as a belt-and-suspenders fallback for all numpy scalar types.

**Recovery actions:** 6,543 corrupt files were deleted; 4,552 valid files from Run 1 were retained. Run 2 with `--workers 8 --no-gpu` completed the remaining 27,812 files with **0% corruption rate** (verified by `_audit_stage3.py`).

### 3.4. Extraction Results — Final State

| Type | Count | Completed | Valid JSON | Error rate |
|------|-------|-----------|-----------|-----------|
| line | 12,930 | 100% | 12,930 | 0% |
| scatter | 6,278 | 100% | 6,278 | 0% |
| bar | 5,745 | 100% | 5,745 | 0% |
| heatmap | 4,073 | 100% | 4,073 | 0% |
| histogram | 1,006 | 100% | 1,006 | 0% |
| box | 880 | 100% | 880 | 0% |
| pie | 835 | 100% | 835 | 0% |
| area | 617 | 100% | 617 | 0% |
| **TOTAL** | **32,364** | **100%** | **32,364** | **0%** |

### 3.5. Feature Quality Metrics

Quality was assessed over the full 32,364-file corpus using `scripts/_full_audit.py`.

| Type | axis_info% | avg OCR conf | avg texts | avg elems | zero_texts% |
|------|-----------|-------------|-----------|-----------|-------------|
| histogram | 100% | **0.932** | 31.5 | 47.1 | 2.7% |
| box | 100% | 0.908 | 23.8 | 51.4 | 2.4% |
| line | 100% | 0.909 | 24.3 | 56.6 | 4.8% |
| heatmap | 100% | 0.843 | 50.0 | 87.0 | 11.7% |
| scatter | 100% | 0.802 | 20.8 | 102.4 | 14.6% |
| pie | 100% | 0.783 | 12.7 | 26.7 | 19.0% |
| bar | 100% | 0.671 | 19.9 | 56.0 | 29.6% |
| area | 34.2% | 0.387 | 7.4 | 63.9 | 59.5% |

**Notes:**

- `axis_info` field is present for 100% of all types except `area` (34.2%), where the extractor could not locate a conventional axis in free-form area charts.
- `x_range` / `y_range` is populated when `x_calibration_confidence > 0` (axis ticks are numeric and parseable). Categorical-only axes (e.g., most bar charts with category labels) have `x_min = None`.
- Zero-text entries are genuine: the images use bitmap-rendered fonts that PaddleOCR cannot decode. These entries still carry `elements` data (e.g., bar/point geometries).
- `area` charts exhibit notably lower OCR confidence (0.387) and high zero-text rate, consistent with their visual complexity (overlapping filled regions, often no explicit axis ticks).

### 3.6. Axis Calibration by Type

| Type | x_cal_avg | y_cal_avg | Interpretation |
|------|----------|----------|----------------|
| line | 0.615 | 0.656 | Best: numeric ticks on both axes |
| histogram | 0.630 | 0.671 | Good: frequency histograms have clear ticks |
| scatter | 0.485 | 0.456 | Moderate: log-scale axes common |
| box | 0.323 | 0.645 | x is categorical, y is numeric |
| heatmap | 0.342 | 0.302 | Low: cell labels dominate, few axis ticks |
| bar | 0.219 | 0.466 | Low x (categorical), moderate y (numeric) |
| pie | 0.097 | 0.138 | Minimal: no axis system |
| area | 0.000 | 0.000 | Axisless (detection failed) |

**Known limitation:** Log-scale axes (common in scatter plots) are extracted as if linear. For example, a scatter with `x_min=103, x_max=105` represents a range of 1,000–100,000 in log₁₀ space, but is stored as 103–105. This is flagged via `x_calibration_confidence` being moderate rather than 1.0 for such cases.

---

## 4. QA Dataset Generation

### 4.1. Generation Methodology

QA pairs were generated by prompting **Google Gemini 2.0 Flash** with each chart image and a structured prompt requesting diverse question types:

| Question Type | Stage | Description |
|--------------|-------|-------------|
| structural | 1 | Layout, element count, axis labels |
| extraction | 2 | Read exact values from chart |
| range / threshold | 2 | Min/max, boundary queries |
| comparison | 3 | Relative ordering across series |
| trend | 3 | Directional change over domain |
| why_reasoning | 3 | Causal/interpretive questions |
| multi_hop | 3 | Combined extraction + reasoning |
| interpolation | 3 | Value at unlabeled position |

Each image generated an average of ~9 QA pairs. Pairs with empty question or answer were filtered.

### 4.2. QA Coverage

| Metric | Value |
|--------|-------|
| Total QA files generated | 32,445 |
| Total QA pairs | ~291,000 (estimated avg 9/chart) |
| Images with QA coverage | 32,445 / 46,910 = **69.2%** |
| Images with both QA + Stage3 | ~32,364 (overlapping set) |

### 4.3. QA Validation

Cross-validation via OCR text overlap (Section 3 of `scripts/_cross_check.py`): **87% of Stage3 OCR tokens appear in corresponding QA answers** across bar, histogram, and scatter chart pairs (39/45 sampled pairs), confirming semantic consistency between extracted features and generated questions.

---

## 5. SLM Training Dataset

### 5.1. Dataset v2 (Current Baseline)

`data/slm_training_v2/` — 27,200 samples in `conversations` + `metadata` format.

| Subset | Count |
|--------|-------|
| train.json | 27,200 |
| val.json | ~3,200 |
| test.json | ~1,600 |

**Schema per sample:**
```json
{
  "conversations": [
    {"role": "system",    "content": "<curriculum-specific system prompt>"},
    {"role": "user",      "content": "[CHART_TYPE]: BAR\n[OCR_TEXT]: ...\n\n[QUESTION]: ..."},
    {"role": "assistant", "content": "<answer>"}
  ],
  "metadata": {
    "question_type": "comparison",
    "curriculum_stage": 3,
    "chart_type": "bar",
    "image_id": "arxiv_...",
    "difficulty": 4,
    "source": "gemini-2.0-flash"
  }
}
```

**Known gaps in v2:**
| Issue | Impact |
|-------|--------|
| `axis_info` not in prompt (KV access bug: `y_range.min` instead of `y_min`) | Model cannot leverage calibrated axis values |
| No `[ELEMENTS]` breakdown by type | Element geometry unusable |
| No `line` type samples | 12,930 line charts absent from training |
| 6,182/27,200 samples use `[CAPTION]+[CONTEXT]` format without OCR | Format inconsistency |
| Zero-text samples included without marker | Model receives empty OCR silently |

### 5.2. Dataset v3 Design (Target)

`data/slm_training_v3/` — target ~45,000–60,000 samples.

**Improvements over v2:**

1. **Axis info correctly embedded** — reads `x_min`/`x_max`/`y_min`/`y_max` directly; includes calibration confidence gate (`conf > 0.3` required to include `[AXIS_INFO]`)
2. **Structured text roles** — OCR tokens grouped by role: `[TITLE]`, `[LEGEND]`, `[X_TICKS]`, `[Y_TICKS]`, `[DATA_LABELS]`
3. **Element counts by type** — `[ELEMENTS]: bar=24 point=0` instead of raw list length
4. **Zero-text marker** — `[OCR_QUALITY]: low` tag appended when `texts == []`
5. **Full type coverage** — `line` charts (12,930) included for first time
6. **Metadata enrichment** — adds `has_stage3`, `axis_conf`, `features_used` fields
7. **Consistent schema** — all entries use `[CHART_TYPE]` + role-structured text; caption/context format unified

**Dry-run results (2026-03-01):**

| Metric | Value |
|--------|-------|
| Total samples built | **268,799** (vs 27,200 in v2 — 9.9x increase) |
| Train / Val / Test | 228,494 / 26,888 / 13,417 |
| Stage3 coverage | 32,364/32,364 (100%) |
| Has axis info | 187,986/268,799 (69.9%) |
| Stage 1 samples | 47,314 (17.6%) |
| Stage 2 samples | 55,836 (20.8%) |
| Stage 3 samples | 165,649 (61.6%) |
| Charts with no QA | 13 (0.04%) |
| Corrupted features | 0 |

The 9.9x increase vs v2 is attributable to: (a) `line` charts now included (108,419 samples, 40.3%), (b) full 100% Stage3 coverage enabling all remaining chart types, and (c) avg ~8.3 QA pairs per chart processed without skipping.

**Expected prompt format (v3):**
```
[CHART_TYPE]: LINE
[TITLE]: Training loss comparison
[LEGEND]: Adam, SGD, RMSProp
[X_TICKS]: 0, 10, 20, 30, 40, 50
[Y_TICKS]: 0.1, 0.3, 0.5, 0.7, 1.0
[DATA_LABELS]: 0.12, 0.45
[AXIS_INFO]: x=[0.0, 50.0] conf=0.89 | y=[0.1, 1.0] conf=0.92
[ELEMENTS]: point=45

[QUESTION]: At what epoch does the Adam optimizer first achieve a loss below 0.3?
```

---

## 6. Intermediate and Derived Data Summary

| Directory | Size | Contents | Reusable |
|-----------|------|----------|---------|
| `data/academic_dataset/images/` | ~15 GB | Source chart crops (JPG/PNG) | Required |
| `data/academic_dataset/stage3_features/` | ~800 MB | 32,364 JSON feature files | Required |
| `data/academic_dataset/chart_qa_v2/generated/` | ~2 GB | 32,445 QA JSON files | Required |
| `data/cache/ocr_cache.json` | 589 MB | PaddleOCR results for 46,910 images | Required for re-extraction |
| `data/slm_training_v2/` | 54 MB | Baseline SLM training set (v2) | Archivable |
| `data/academic_dataset/detected_charts/` | 12 GB | Pre-classification crops | Archivable after v3 built |
| `data/academic_dataset/classified_charts/` | 12 GB | Post-classification copies | Archivable after v3 built |
| `data/raw_pdfs/` | 40 GB | Original arXiv PDFs | Archivable |
| `data/search_cache/` | 8 MB | arXiv download progress files | Deletable |

**Archival candidates (64 GB total):** `detected_charts`, `classified_charts`, `raw_pdfs` — no longer needed once Stage3 extraction is complete and verified.

---

## 7. Reproducibility Notes

### 7.1. Environment

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PaddleOCR | 2.7.x (CPU mode, cache-hit only) |
| Ultralytics YOLO | 8.x |
| PyTorch | 2.x |
| OpenCV | 4.x |

### 7.2. Key Commands

```bash
# Stage 3 extraction (from clean state)
.venv/Scripts/python.exe scripts/pipeline/batch_stage3_parallel.py --workers 8 --no-gpu

# Status check
.venv/Scripts/python.exe scripts/pipeline/batch_stage3_parallel.py --status

# Quality audit
.venv/Scripts/python.exe scripts/_full_audit.py

# Build SLM training v3
.venv/Scripts/python.exe scripts/training/prepare_slm_training_v3.py \
    --output-dir data/slm_training_v3

# Dry run (stats only, no output written)
.venv/Scripts/python.exe scripts/training/prepare_slm_training_v3.py --dry-run
```

### 7.3. Configuration

All data directories are defined relative to `PROJECT_ROOT` (`Path(__file__).parent.parent`). No hardcoded absolute paths.  
API keys and secrets are stored in `config/secrets/` (gitignored).

---

## 8. Summary Statistics

| Stage | Input | Output | Notes |
|-------|-------|--------|-------|
| PDF collection | ~4k arXiv papers | 150k page images | PyMuPDF @150 DPI |
| Page detection | 150k images | 70k chart regions | YOLOv8m |
| Quality filter | 70k regions | 46,910 charts | Resolution + dedup |
| Classification | 46,910 charts | 32,364 (8 types) | ResNet-18 94.66% |
| OCR caching | 46,910 charts | 589 MB cache | PaddleOCR one-shot |
| Stage 3 extraction | 32,364 charts | 32,364 feature JSONs | 0% corruption (Run 2) |
| QA generation | 32,364 charts | 32,445 QA files | Gemini 2.0 Flash |
| SLM dataset v2 | ~32k QA files | 27,200 samples | Partial axis info |
| SLM dataset v3 | 32,364 + QA | ~50,000 samples | Full axis + roles |

---

## 9. Next Steps

| Priority | Task | Estimated Effort |
|----------|------|-----------------|
| **HIGH** | Build `slm_training_v3` with enriched prompts | 2–4 hours |
| **HIGH** | Fine-tune Qwen-2.5-1.5B on v3 dataset | 6–12 hours (GPU) |
| **MEDIUM** | Archive `detected_charts` + `classified_charts` + `raw_pdfs` (−64 GB) | 1 hour |
| **MEDIUM** | Flag log-scale axes in `axis_info` | 2 hours |
| **LOW** | Re-extract `area` charts with relaxed axis detector | 3–5 hours |
| **LOW** | Add `not_a_chart` negative samples to SLM training | 1 hour |
