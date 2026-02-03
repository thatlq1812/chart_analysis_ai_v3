# MASTER CONTEXT - Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.7.0 | 2026-02-04 | That Le | Project cleanup, documentation refresh |
| 1.6.0 | 2026-01-30 | That Le | Stage 4 core implemented (ValueMapper + PromptBuilder) |
| 1.5.0 | 2026-01-29 | That Le | Stage 3 fully enhanced and validated (100% accuracy) |
| 1.4.0 | 2026-01-26 | That Le | ResNet-18 classifier integrated and validated |
| 1.3.0 | 2026-01-25 | That Le | Documentation restructured, Stage 4/5 docs added |
| 1.2.1 | 2026-01-24 | That Le | Stage 3 tested on academic dataset |
| 1.2.0 | 2026-01-24 | That Le | Stage 3 implementation complete |
| 1.1.0 | 2026-01-24 | That Le | Phase 1 complete, moving to Phase 2 |
| 1.0.0 | 2026-01-19 | That Le | Initial master context for V3 |

---

## Quick Summary

| Property | Value |
| --- | --- |
| **Project Name** | Geo-SLM Chart Analysis |
| **Project Type** | AI Research / Thesis Project |
| **Core Philosophy** | Hybrid Intelligence (Neural + Symbolic) |
| **Primary Method** | YOLO Detection + Geometric Mapping + SLM Reasoning |
| **Language** | Python 3.11+ |
| **Current Phase** | Phase 2 - Core Engine (Stage 4 In Progress) |
| **Target** | Academic Thesis + Research Paper |
| **Tests** | 176/177 passing (99.4%) |
| **OCR Cache** | 46,910 entries (~600MB) |

---

## 1. Project Identity

### 1.1. What We Are Building

A **hybrid AI system** for extracting structured data from chart images with academic-grade precision. The system combines:

| Component | Role | Technology |
| --- | --- | --- |
| **Computer Vision** | Detect and localize charts | YOLOv8/v11 |
| **Geometric Analysis** | Precise value extraction | OpenCV + NumPy |
| **Small Language Model** | OCR correction + reasoning | Qwen-2.5 / Llama-3.2 |

**Core Value Proposition:**
> "Achieve higher accuracy than pure multimodal LLMs (GPT-4V) through hybrid neuro-symbolic approach, while maintaining local inference capability."

### 1.2. What We Are NOT Building

- NOT a general-purpose document parser
- NOT a real-time video processing system
- NOT dependent on cloud LLM APIs for core inference
- NOT a production SaaS (research-first, productization later)

### 1.3. Target Users

| User | Use Case |
| --- | --- |
| Researchers | Extract data from paper charts |
| Students | Analyze charts for thesis |
| Data Analysts | Digitize legacy charts |

---

## 2. Technology Stack

### 2.1. Core Engine (LOCKED)

| Layer | Technology | Version | Rationale |
| --- | --- | --- | --- |
| Language | Python | 3.11+ | AI/ML ecosystem |
| Detection | Ultralytics YOLO | v8/v11 | SOTA object detection |
| OCR | PaddleOCR | Latest | Best multi-language |
| Image Processing | OpenCV + Pillow | Latest | Industry standard |
| Geometric | NumPy + SciPy | Latest | Precise calculations |
| SLM | Qwen-2.5-1.5B | Latest | Local inference |
| Validation | Pydantic | v2 | Schema enforcement |
| Config | OmegaConf | Latest | Hierarchical config |

### 2.2. Research & Training

| Component | Technology | Purpose |
| --- | --- | --- |
| Notebooks | Jupyter Lab | Interactive exploration |
| Experiment Tracking | MLflow / W&B | Metrics logging |
| Model Training | PyTorch + Ultralytics | Fine-tuning |
| Data Versioning | DVC (optional) | Dataset management |

### 2.3. Interface Layer

| Component | Technology | Purpose |
| --- | --- | --- |
| CLI | Typer | Developer testing |
| Demo UI | Streamlit | Quick visualization |
| API | FastAPI | Production integration |
| Docs | MkDocs | Documentation site |

---

## 3. Architecture Overview

### 3.1. System Layers

```
+------------------------------------------------------------------+
|                        INTERFACE LAYER                            |
|  [CLI Tool]       [Streamlit Demo]       [FastAPI Server]        |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        CORE ENGINE                                |
|  +------------------------------------------------------------+  |
|  |                    Pipeline Orchestrator                    |  |
|  +------------------------------------------------------------+  |
|  | Stage 1  | Stage 2  | Stage 3  | Stage 4  | Stage 5        |  |
|  | Ingest   | Detect   | Extract  | Reason   | Report         |  |
|  +------------------------------------------------------------+  |
|  |                    Pydantic Schemas                         |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        DATA LAYER                                 |
|  [Raw]     [Processed]     [Cache]     [Models]     [Outputs]    |
+------------------------------------------------------------------+
```

### 3.2. The 5-Stage Pipeline

```
Input (PDF/Image)
    |
    v
+-------------------+
| Stage 1: Ingest   | --> Load, Convert, Normalize
+-------------------+
    |
    v (Clean Images)
+-------------------+
| Stage 2: Detect   | --> YOLO Detection, Crop Charts
+-------------------+
    |
    v (Cropped Charts)
+-------------------+
| Stage 3: Extract  | --> OCR Text, Detect Elements, Geometric Analysis
+-------------------+
    |
    v (Raw Metadata)
+-------------------+
| Stage 4: Reason   | --> SLM Correction, Value Mapping, Description
+-------------------+
    |
    v (Refined Data)
+-------------------+
| Stage 5: Report   | --> Validate, Insights, Format Output
+-------------------+
    |
    v
Output (JSON + Report)
```

---

## 4. Directory Structure

```
chart_analysis_ai_v3/
|
+-- .github/
|   +-- instructions/           # AI Agent guidelines
|       +-- coding-standards.instructions.md
|       +-- docs.instructions.md
|       +-- pipeline.instructions.md
|       +-- project.instructions.md
|       +-- research.instructions.md
|       +-- system.instructions.md
|
+-- config/
|   +-- base.yaml               # Shared configuration
|   +-- models.yaml             # Model paths & thresholds
|   +-- pipeline.yaml           # Stage configuration
|   +-- secrets/                # API keys (gitignored)
|
+-- data/
|   +-- academic_dataset/       # Arxiv chart images + metadata
|   +-- raw/                    # Input files (PDF, images)
|   +-- processed/              # Pipeline outputs
|   +-- cache/                  # Intermediate results
|   +-- output/                 # Stage reports
|   +-- samples/                # Demo/test samples
|
+-- docs/                       # Documentation (see docs/README.md)
|   +-- MASTER_CONTEXT.md       # This file
|   +-- architecture/           # System design docs
|   +-- guides/                 # How-to guides
|   +-- research/               # Research methodology
|   +-- reports/                # Test reports & benchmarks
|   +-- archive/                # Historical docs
|
+-- models/
|   +-- weights/                # Trained model files
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_stage3_visualization.ipynb
|
+-- research/
|   +-- experiments/            # Experiment scripts
|   +-- papers/                 # Paper summaries
|   +-- prototypes/             # Quick tests
|
+-- src/
|   +-- core_engine/
|       +-- __init__.py
|       +-- pipeline.py         # Main orchestrator
|       +-- exceptions.py       # Custom exceptions
|       +-- schemas/            # Pydantic models
|       +-- stages/             # Pipeline stages
|           +-- s1_ingestion/   # Stage 1
|           +-- s2_detection/   # Stage 2
|           +-- s3_extraction/  # Stage 3 (OCR, geometry, elements)
|           +-- s4_reasoning/   # Stage 4 (value mapper, prompts)
|               +-- value_mapper.py    # Pixel-to-value conversion
|               +-- prompt_builder.py  # Canonical Format builder
|               +-- prompts/           # Template files
|           +-- s5_reporting/   # Stage 5 (TODO)
|       +-- validators/         # Input validators
|
+-- interface/                  # Interface layer (future)
|
+-- tests/
|   +-- conftest.py             # Shared fixtures
|   +-- test_schemas.py
|   +-- test_s3_extraction/     # Stage 3 tests (129 cases)
|   +-- test_s4_reasoning/      # Stage 4 tests (36 cases)
|       +-- test_value_mapper.py
|       +-- test_prompt_builder.py
|   +-- fixtures/               # Test data
|
+-- scripts/                    # Utility scripts
|   +-- benchmark_classifier.py
|   +-- generate_stage3_report.py
|   +-- test_stage3_academic_dataset.py
|   +-- train_yolo.py
|
+-- logs/                       # Log files
+-- pyproject.toml
+-- README.md
+-- Makefile
```

---

## 5. Current Status

### 5.1. Phase 1: Foundation [COMPLETED]

| Task | Status | Notes |
| --- | --- | --- |
| Project structure setup | [DONE] | V3 initialized |
| Documentation framework | [DONE] | Instructions created |
| Data collection pipeline | [DONE] | PDF mining + Gemini classification |
| Chart QA dataset | [DONE] | 32,445 charts, 32,445 QA pairs (v2) |
| Stage 1: Ingestion | [DONE] | PDF/Image loading implemented |
| Stage 2: Detection | [DONE] | YOLO integration complete |

### 5.2. Phase 2: Core Engine [IN PROGRESS]

| Task | Status | Notes |
| --- | --- | --- |
| Stage 3: Extraction | [DONE] | Geo-SLM hybrid approach implemented |
| Stage 3 Testing | [DONE] | Tested on 800+ images (100% accuracy) |
| Stage 4: Reasoning | [IN PROGRESS] | Core components implemented |
| Stage 5: Reporting | [TODO] | Output formatting |

**Stage 4 Implementation Details:**

| Submodule | Status | Description |
| --- | --- | --- |
| GeometricValueMapper | [DONE] | Pixel-to-value conversion using AxisInfo calibration |
| GeminiPromptBuilder | [DONE] | Canonical Format prompts with anti-hallucination |
| ReasoningEngine | [DONE] | Orchestrator integrating mapper + builder |
| Prompt Templates | [DONE] | reasoning.txt + canonical_format.md |
| Unit Tests | [DONE] | 36 test cases (16 mapper + 20 builder) |
| Gemini Integration | [PROTOTYPE] | Using API for rapid development |
| Local SLM | [TODO] | Self-trained model for production |

**Stage 4 Architecture:**

```
Stage 3 Output (RawMetadata)
         |
         v
+-----------------------------------+
|    GeometricValueMapper           |
|    - Calibrate axes from AxisInfo |
|    - Convert pixel -> value       |
|    - Handle linear/log scales     |
|    - Y-axis inversion             |
+-----------------------------------+
         |
         v (MappingResult)
+-----------------------------------+
|    GeminiPromptBuilder            |
|    - Build CanonicalContext       |
|    - Anti-hallucination rules     |
|    - Structured JSON output       |
+-----------------------------------+
         |
         v (Structured Prompt)
+-----------------------------------+
|    LLM Backend                    |
|    - Gemini API (prototype)       |
|    - Local SLM (production)       |
+-----------------------------------+
         |
         v (RefinedChartData)
+-----------------------------------+
|    Post-processing                |
|    - Merge OCR + mapped values    |
|    - Confidence scoring           |
|    - Description generation       |
+-----------------------------------+
         |
         v
Stage 5 Input (RefinedChartData)
```

**Stage 3 Implementation Details:**

| Submodule | Status | Description |
| --- | --- | --- |
| Preprocessor | [DONE] | Negative image + adaptive threshold |
| Skeletonizer | [DONE] | Lee algorithm, keypoint detection |
| Vectorizer | [DONE] | RDP algorithm, subpixel refinement |
| OCR Engine | [DONE] | PaddleOCR with role classification |
| Geometric Mapper | [DONE] | Axis calibration, pixel-to-value |
| Element Detector | [DONE] | Bars, markers, pie slices |
| Classifier | [UPGRADED] | ResNet-18 v2 (94.14% accuracy, 32,445 images) |
| Unit Tests | [DONE] | 7 test modules, 129 test cases |
| Documentation | [DONE] | STAGE3_EXTRACTION.md created |
| Academic Dataset Test | [DONE] | 15/15 images processed successfully |
| Production Integration | [DONE] | ResNet18Classifier wrapper ready |

**ResNet-18 Classifier (v2 - 2026-01-31):**

| Metric | Value |
| --- | --- |
| Test Accuracy | **94.14%** |
| Best Validation Accuracy | 94.80% |
| Training Time | 50:22 (RTX 3060) |
| Inference Speed (ONNX) | 6.90ms mean (CPU), 144.9 img/sec |
| Model Size | 42.64 MB (ONNX format) |
| Classes | 8 types (area, bar, box, heatmap, histogram, line, pie, scatter) |
| Dataset | 32,445 preprocessed images (256x256 grayscale) |

**Per-class Accuracy (v2):**

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

**Stage 3 Full Pipeline Test Results (2026-01-29):**

| Metric | Value |
| --- | --- |
| Total Images Tested | 800+ (100 per type) |
| Classification Accuracy | **100%** (all 8 types) |
| OCR Confidence | **91.5%** (EasyOCR) |
| Overall Confidence | **92.6%** |
| Avg Processing Time | ~7.6s (improved from 14.6s) |

**Stage 3 Enhancements Implemented:**

| Enhancement | Description |
| --- | --- |
| RANSAC Fitting | Robust axis calibration with outlier rejection |
| Curve Fitting | Circle/arc/ellipse fitting for line charts |
| OCR Post-processing | Error correction (O->0, l->1, etc.) |
| Content-aware Role | Keyword-based text classification |
| Confidence Scoring | Weighted overall from 4 components |
| Gap Filling | Morphological closing in skeletonizer |

Reports generated:
- [STAGE3_COMPLETION_SUMMARY.md](reports/STAGE3_COMPLETION_SUMMARY.md)
- [STAGE3_CLASSIFIED_TEST_REPORT.md](reports/STAGE3_CLASSIFIED_TEST_REPORT.md)

### 5.3. Upcoming Phases

| Phase | Focus | Timeline |
| --- | --- | --- |
| **Stage 4** | SLM fine-tuning (Qwen2.5 on 32K QA pairs) | Current |
| **Stage 5** | Parallel processing, insights generation | After Stage 4 |
| Phase 3 | Optimization & Benchmarking | Week 5-6 |
| Phase 4 | Demo & Documentation | Week 7-8 |

**SLM Training Plan:**

| Item | Description |
| --- | --- |
| Base Model | Qwen2.5-1.5B-Instruct |
| Dataset | 32,445 QA pairs (image + question-answer) |
| Training Type | LoRA fine-tuning |
| Task | Chart understanding and data extraction |
| Hardware | RTX 3060 6GB VRAM |
| Expected Output | Local SLM for Stage 4 inference |

### 5.4. OCR Cache Status (2026-02-04)

| Metric | Value |
| --- | --- |
| Total Entries | 46,910 |
| File Size | ~600MB |
| File Location | `data/cache/ocr_cache.json` |
| Key Format | `{chart_type}\{filename}` |
| OCR Engine | PaddleOCR v2.9.1 |

**Coverage by Chart Type:**

| Type | Cached | Notes |
| --- | --- | --- |
| line | ~10,000 | Complete |
| bar | ~9,000 | Complete |
| scatter | ~5,000 | Complete |
| box | ~5,000 | Complete |
| pie | ~2,400 | Complete |
| histogram | ~2,000 | Complete |
| area | ~1,200 | Complete |
| heatmap | ~700 | Complete |

### 5.5. Test Coverage (2026-02-04)

| Test Suite | Passed | Failed | Total |
| --- | --- | --- | --- |
| Schemas | 19 | 0 | 19 |
| Stage 3 Extraction | 139 | 1 | 140 |
| Stage 4 Reasoning | 18 | 0 | 18 |
| **Total** | **176** | **1** | **177** |

**Known Issue:** 1 test fails due to LINE chart classified as AREA (edge case)

### 5.6. Project Cleanup (2026-02-04)

**Deleted:**
- Root trash files: `chatlog*.md`, `log3.md`, `nul`
- `.venv_paddle/` (~2GB)
- `data/academic_dataset/classified_charts_preprocessed/` (383MB)
- 6 duplicate scripts in `scripts/`
- Old training runs in `runs/`

**Updated:**
- `scripts/README.md` - Current script list
- `notebooks/README.md` - Current notebook list
- All docs version headers updated to v2.0.0

**Stage 5 Planned Architecture:**

```
Stage 4 Output (RefinedChartData[])
         |
         v
+----------------------------------------+
|    Parallel Processing Engine          |
|    - ThreadPoolExecutor for batch      |
|    - Async insight generation          |
+----------------------------------------+
         |
         +-------+-------+-------+
         v       v       v       v
     [Trend]  [Stats] [Compare] [Summary]
         |       |       |       |
         +-------+-------+-------+
                 |
                 v
+----------------------------------------+
|    Report Generator                    |
|    - JSON schema validation            |
|    - Markdown report                   |
|    - Traceability info                 |
+----------------------------------------+
         |
         v
Final Output (PipelineResult)

---

## 6. Key Design Decisions

| Decision | Rationale | Date |
| --- | --- | --- |
| Use YOLO for detection | Fast inference, easy fine-tuning | 2026-01-19 |
| PyMuPDF for PDF processing | Fastest, preserves text coordinates | 2026-01-19 |
| PaddleOCR over Tesseract | Better accuracy for mixed languages | 2026-01-19 |
| Pydantic v2 for schemas | Type safety, validation, serialization | 2026-01-19 |
| Local SLM over API | Offline capability, cost, privacy | 2026-01-19 |
| Qwen-2.5 as default SLM | Best performance/size ratio | 2026-01-19 |
| Negative image preprocessing | Better skeleton extraction from charts | 2026-01-25 |
| RDP vectorization | Preserves data points, reduces noise | 2026-01-25 |
| Spatial OCR role classification | Context-aware text extraction | 2026-01-25 |
| Hybrid bar+line detection | Contours for bars, skeleton for lines | 2026-01-25 |
| Stage 3 = Pure Geometry | Output toán học hóa, no LLM dependency | 2026-01-29 |
| Stage 4 = Gemini prototype | Rapid iteration, then train local SLM | 2026-01-30 |
| Canonical Format prompts | Anti-hallucination, structured output | 2026-01-30 |
| Stage 5 = Parallel processing | Async insights for throughput | 2026-01-30 |

---

## 7. Academic Contributions

| Area | Contribution Type | Papers to Reference |
| --- | --- | --- |
| Hybrid Detection | Novel Pipeline Design | YOLO, Faster R-CNN |
| Geometric Mapping | Algorithm Design | Classical CV papers |
| SLM Distillation | Knowledge Transfer | DistilBERT, TinyBERT |
| Error Correction | Multi-stage Pipeline | Ensemble methods |

---

## 8. References

### Research Papers
- [YOLO: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [PaddleOCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/abs/2009.09941)
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)

### Technical Documentation
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)

### Related Projects
- chart_analysis_ai (V1) - Reference implementation
- chart_analysis_ai_v2 - Lessons learned
