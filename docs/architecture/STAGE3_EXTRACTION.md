# Stage 3: Extraction - VLM Chart-to-Table

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 3.1.0 | 2026-03-12 | That Le | EfficientNet-B0 3-class classifier integrated (97.54%) |
| 3.0.0 | 2026-02-28 | That Le | Full rewrite - VLM-based pluggable extractor architecture |
| 2.0.0 | 2026-02-04 | That Le | Updated with complete implementation status |

## Status: COMPLETE (v3 VLM Architecture)

Stage 3 is fully implemented as a VLM-based chart-to-table extractor with
four pluggable backends supporting ablation experiments.

## 1. Overview

Stage 3 converts cropped chart images directly to structured data tables
using vision-language models (VLMs). The legacy geometry pipeline
(OCR skeletonization, RANSAC axis calibration, pixel-to-value mapping)
has been removed in favor of end-to-end neural extraction.

### Design Rationale

| Approach | Accuracy | Robustness | Speed |
| --- | --- | --- | --- |
| Geometry (old) | Low on real data | Fails on noise/compression | Fast |
| DePlot / MatCha | High on synthetic | Good generalization | Medium |
| SVLM (Qwen2-VL) | Best zero-shot | Most robust | Slow |

The geometry approach fundamentally breaks on real-world charts due to:
- Axis label noise and font variation
- Grid line interference with skeletonization
- Domain shift between PDF rendering and training data

## 2. Architecture



## 3. Extractor Backends

All backends implement BaseChartExtractor and produce Pix2StructResult.

| Backend ID | Model | Architecture | Use Case |
| --- | --- | --- | --- |
| deplot | google/deplot | Pix2Struct fine-tuned | Default, synthetic charts |
| pix2struct | google/pix2struct-base | Pix2Struct base | Ablation baseline |
| matcha | google/matcha-base | Pix2Struct fine-tuned | Complex axes, formulas |
| svlm | Qwen/Qwen2-VL-2B-Instruct | Chat VLM | Zero-shot, real-world |

### 3.1. Pix2Struct Family (DePlot, MatCha, Pix2Struct-base)

All three use AutoProcessor + AutoModelForImageTextToText and the
DePlot linearized table parser.

Prompt: "Generate underlying data table of the figure below:"

DePlot output format (pipe-separated, newlines as special tokens):
    TITLE | chart_title
    col0  | col1 | col2
    val0  | val1 | val2
    ...

The parser strips TITLE lines and returns (headers, rows).

### 3.2. SVLM Backend (Qwen2-VL-2B-Instruct)

Uses chat-format VLM with zero-shot JSON prompting.
Parser tries JSON extraction first, then DePlot linearized as fallback.

Expected output:
    {"headers": ["Year", "Revenue"], "rows": [["2020", "100"], ...]}

### 3.3. Ablation Experiment Design

| Experiment | backend | Expected outcome |
| --- | --- | --- |
| Full fine-tuning (primary) | deplot | Best F1 on synthetic data |
| Enhanced math reasoning | matcha | Better on complex axes |
| No fine-tuning (baseline) | pix2struct | Lower accuracy (control) |
| Zero-shot VLM | svlm | Best on real-world styles |

## 4. Configuration

Controlled by ExtractionConfig (Pydantic) and config/pipeline.yaml.

Python:
    config = ExtractionConfig(
        extractor_backend="deplot",
        extractor_model=None,
        extractor_max_patches=1024,
        extractor_device="auto",
        use_efficientnet_classifier=True,
    )

YAML (config/pipeline.yaml):
    extraction:
      extractor_backend: deplot
      extractor_model: null
      extractor_max_patches: 1024
      extractor_device: auto
      use_efficientnet_classifier: true

## 5. Data Contract

Input: DetectedChart (Stage 2)
    chart.chart_id      - unique identifier
    chart.cropped_path  - Path to cropped PNG

Output: RawMetadata via Stage3Output
    chart_id, chart_type (EfficientNet-B0)
    pix2struct_table:
        headers, rows, records, extraction_confidence
        model_name (e.g. "google/deplot")
    texts=[], elements=[], axis_info=None (geometry removed)

Stage 4 reads pix2struct_table and converts to DataSeries.
When extraction_confidence == 0, Stage 4 returns empty series.

## 6. File Structure

    src/core_engine/stages/s3_extraction/
        __init__.py             - Public exports
        s3_extraction.py        - Stage3Extraction + ExtractionConfig
        extractors.py           - BaseChartExtractor + 4 backends + factory
        pix2struct_extractor.py - Backward-compat alias (kept for imports)
        resnet_classifier.py    - EfficientNet-B0 / ResNet-18 classifiers
        # Legacy files (not used by pipeline, kept for reference)
        preprocessor.py, skeletonizer.py, vectorizer.py
        element_detector.py, geometric_mapper.py, ocr_engine.py

## 7. References

| Paper | Model | Key Contribution |
| --- | --- | --- |
| Lin et al. 2022 (ACL 2023) | DePlot | Chart-to-table derendering |
| Liu et al. 2022 (ACL 2023) | MatCha | Math+chart joint pretraining |
| Lee et al. 2023 (ICML 2023) | Pix2Struct | Screenshot parsing pretraining |
| Wang et al. 2024 | Qwen2-VL | Any-resolution VLM |
