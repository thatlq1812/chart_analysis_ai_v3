---
applyTo: "src/core_engine/stages/s3_extraction/**,models/weights/efficientnet_b0*,models/weights/chart_classifier*"
---

# MODULE INSTRUCTIONS - Chart Extraction (Stage 3)

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.1.0 | 2026-03-12 | That Le | EfficientNet-B0 3-class classifier upgrade (97.54% accuracy) |
| 2.0.0 | 2026-02-28 | That Le | Full rewrite - VLM-based pluggable extraction architecture |
| 1.0.0 | 2026-01-19 | That Le | Initial geometry pipeline |}

---

## 1. Overview

Stage 3 converts cropped chart images directly to structured data tables
using a pluggable VLM extractor backend. The legacy geometry pipeline
(OCR, skeletonization, axis calibration) has been removed.

**Key files:**
- s3_extraction.py - Orchestrator + ExtractionConfig
- extractors.py - BaseChartExtractor + 4 backends + factory
- pix2struct_extractor.py - Backward-compat re-export only
- resnet_classifier.py - EfficientNet-B0 / ResNet-18 classifiers

---

## 2. Extractor Backend Architecture

All extractor backends implement BaseChartExtractor and return Pix2StructResult.

| Backend | Default Model | Config Key | Notes |
| --- | --- | --- | --- |
| DeplotExtractor | google/deplot | extractor_backend: deplot | Primary default |
| MatchaExtractor | google/matcha-base | extractor_backend: matcha | Enhanced math |
| Pix2StructBaselineExtractor | google/pix2struct-base | extractor_backend: pix2struct | Ablation baseline |
| SVLMExtractor | Qwen/Qwen2-VL-2B-Instruct | extractor_backend: svlm | Zero-shot VLM |

Use create_extractor() factory to instantiate:
    extractor = create_extractor("deplot", device="auto")
    result = extractor.extract(image_bgr, chart_id="chart_001")

---

## 3. ExtractionConfig Fields

    extractor_backend   : str = "deplot"   # backend identifier
    extractor_model     : Optional[str]   # None = backend default model ID
    extractor_max_patches: int = 1024     # Pix2Struct-family resolution
    extractor_device    : str = "auto"    # auto | cuda | cpu
    use_efficientnet_classifier: bool = True
    efficientnet_model_path: Optional[Path]
    efficientnet_confidence_threshold: float = 0.55

---

## 4. Processing Flow

Per-chart:
    1. cv2.imread(chart.cropped_path)
    2. EfficientNet-B0.predict_with_confidence() -> (chart_type, confidence)
    3. extractor.extract(image_bgr, chart_id) -> Pix2StructResult
    4. Build RawMetadata(chart_id, chart_type, pix2struct_table=result)

RawMetadata.texts = [] and .elements = [] and .axis_info = None (no geometry).

---

## 5. Output Schema (Pix2StructResult)

    headers: List[str]          # column headers, [0] = x-axis labels
    rows: List[List[str]]       # data rows as string matrices
    records: List[Dict[str,str]] # {header: value} convenience view
    raw_html: str               # raw model output for debugging
    model_name: str             # HuggingFace model ID used
    extraction_confidence: float # 1.0 success, 0.0 empty/failed

Stage 4 reads pix2struct_table and uses _pix2struct_to_series() to
produce DataSeries. Returns empty series when confidence == 0.

---

## 6. DePlot Linearized Table Parser

Shared by DeplotExtractor, MatchaExtractor, Pix2StructBaselineExtractor.

Input format from model:
    TITLE | chart_title <0x0A>
    col0  | col1 | col2 <0x0A>
    val0  | val1 | val2 <0x0A>

Parser steps:
    1. Replace <0x0A> with newlines
    2. Strip TITLE lines
    3. Detect first data row (numeric cell detection)
    4. Lines before first data row = headers
    5. Lines after = data rows

SVLMExtractor uses JSON-first parsing with linearized fallback.

---

## 7. Adding a New Backend

1. Create class inheriting BaseChartExtractor in extractors.py
2. Implement extract() and the is_available, backend_id, model_name properties
3. Add entry to BackendType enum and _DEFAULT_MODELS dict
4. Add branch to create_extractor() factory function
5. Document in STAGE3_EXTRACTION.md

---

## 8. Legacy Geometry Files

The following files exist for historical reference and are NOT used by
the current pipeline. Do NOT import them in s3_extraction.py or extractors.py:

    preprocessor.py     - Negative transform, adaptive threshold, grid removal
    skeletonizer.py     - Lee/Zhang thinning
    vectorizer.py       - RDP simplification
    element_detector.py - Bar/marker/slice detection
    geometric_mapper.py - RANSAC pixel-to-value calibration
    ocr_engine.py       - PaddleOCR/EasyOCR text extraction
    classifier.py       - Rule-based chart type classifier
    ml_classifier.py    - Random Forest chart type classifier

---

## 9. Model Weights Location

    models/weights/efficientnet_b0_3class_v1_best.pt  (97.54% acc, 3-class)
    models/weights/efficientnet_b0_4class_v3_best.pt  (4-class fallback)
    models/weights/resnet18_chart_classifier_v2_best.pt (legacy fallback)

HuggingFace cached models (loaded on first inference):
    ~/.cache/huggingface/hub/models--google--deplot     (~1.1 GB)
    ~/.cache/huggingface/hub/models--google--matcha-base (~1.1 GB)
    ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct (~6 GB)

---

## 10. Common Patterns

Standalone extraction:
    from src.core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig
    stage = Stage3Extraction(ExtractionConfig(extractor_backend="deplot"))
    result = stage.process(stage2_output)

Switch backend for ablation:
    config = ExtractionConfig(extractor_backend="matcha")
    stage = Stage3Extraction(config)

Local model path:
    config = ExtractionConfig(extractor_model="/models/my_finetuned_deplot")

Direct extractor use:
    from src.core_engine.stages.s3_extraction.extractors import create_extractor
    extractor = create_extractor("deplot")
    result = extractor.extract(image_bgr, chart_id="test")
