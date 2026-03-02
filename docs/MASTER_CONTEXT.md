# MASTER CONTEXT - Chart Analysis AI v3

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 4.2.0 | 2026-03-02 | That Le | Training postmortem, cloud GPU strategy, project cleanup (~1.5GB freed) |
| 4.1.0 | 2026-03-02 | That Le | Stage 5 enhanced (validation, MD/CSV), SLM eval framework, 294 tests passing |
| 4.0.0 | 2026-03-02 | That Le | Full pipeline implemented, thesis complete (39 pages), 232 tests passing |
| 3.0.0 | 2026-03-01 | That Le | Data pipeline complete, SLM training dataset v3 ready (268,799 samples) |
| 2.0.0 | 2026-02-28 | That Le | Production-ready architecture upgrade (AI Routing, SLM Training, Serving) |
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
| **Project Name** | Chart Analysis AI v3 |
| **Project Type** | AI Research / Thesis Project |
| **Core Philosophy** | Hybrid Intelligence (Neural + Symbolic) |
| **Primary Method** | YOLO Detection + Geometric Mapping + Multi-Provider AI Reasoning |
| **Language** | Python 3.11+ |
| **Current Phase** | Phase 3 - SLM Training (Model Selection) |
| **Target** | Academic Thesis + Research Paper |
| **Source Files** | 48 Python modules (~19,800 LOC) |
| **Tests** | 294 tests passing (23 test files) |
| **OCR Cache** | 46,910 entries (~600MB) |
| **Instructions** | 13 files (3-tier hierarchy) |
| **Stage3 Dataset** | 32,364 charts extracted (100%, 0% error) |
| **SLM Train Dataset** | 268,799 samples (v3, all 8 types, ready) |
| **Thesis** | 39 pages, 7 chapters, 25 visual assets, 0 LaTeX errors |

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
| Validation | Pydantic | v2 | Schema enforcement |
| Config | OmegaConf | Latest | Hierarchical config |

### 2.2. AI Reasoning (Multi-Provider)

| Provider | Model | Purpose | Status |
| --- | --- | --- | --- |
| Local SLM | Qwen-2.5-1.5B + LoRA | Primary reasoning (offline) | TRAINING |
| Gemini | gemini-2.0-flash | Cloud fallback (high accuracy) | ACTIVE |
| OpenAI | gpt-4o-mini | Secondary fallback | OPTIONAL |

All providers accessed through **AIRouter** with fallback chains. See `module-reasoning.instructions.md`.

### 2.3. SLM Training Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| Base Models | Llama-3.2-1B (PRIMARY) / Qwen-2.5-1.5B | Fine-tuning candidates |
| Fine-tuning | LoRA + PEFT | Parameter-efficient training |
| Quantization | BitsAndBytes 4-bit (QLoRA) | Cloud GPU or local |
| Trainer | SFTTrainer (trl) | Supervised fine-tuning |
| Format | ChatML | Standard conversation format |
| Training Infra | Cloud GPU (RunPod/Vast.ai) | A100 40GB recommended |

### 2.4. Research & Training

| Component | Technology | Purpose |
| --- | --- | --- |
| Notebooks | Jupyter Lab | Interactive exploration |
| Experiment Tracking | MLflow / W&B | Metrics logging |
| Model Training | PyTorch + Ultralytics | Fine-tuning |
| Data Versioning | DVC (optional) | Dataset management |

### 2.5. Serving Layer (NEW)

| Component | Technology | Purpose |
| --- | --- | --- |
| API | FastAPI | REST endpoints + OpenAPI |
| Task Queue | Celery + Redis | Async pipeline jobs |
| State | SQLAlchemy + Alembic | Job tracking + persistence |
| Deployment | Docker Compose | Multi-container orchestration |

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
|                        SERVING LAYER (NEW)                        |
|  [FastAPI Routes]  →  [Celery Tasks]  →  [SQLAlchemy State]     |
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
|                     AI ROUTING LAYER (NEW)                        |
|  [AIRouter] → [LocalSLM | Gemini | OpenAI] (fallback chains)   |
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
|   +-- instructions/           # AI Agent guidelines (3-tier hierarchy)
|       +-- system.instructions.md           # Tier 1: Global agent rules
|       +-- project.instructions.md          # Tier 1: Project architecture
|       +-- workflow.instructions.md         # Tier 2: CI/CD, Git, releases
|       +-- module-detection.instructions.md # Tier 3: YOLO detection
|       +-- module-extraction.instructions.md# Tier 3: OCR, classification
|       +-- module-reasoning.instructions.md # Tier 3: AI routing, adapters
|       +-- module-training.instructions.md  # Tier 3: SLM fine-tuning
|       +-- module-serving.instructions.md   # Tier 3: API, Celery, Docker
|       +-- pipeline.instructions.md         # Tier 3: Pipeline stages
|       +-- coding-standards.instructions.md # Tier 3: Python conventions
|       +-- research.instructions.md         # Tier 3: Experiments
|       +-- docs.instructions.md             # Tier 3: Documentation
|
+-- config/
|   +-- base.yaml               # Shared configuration
|   +-- models.yaml             # Model paths & thresholds
|   +-- pipeline.yaml           # Stage configuration
|   +-- training.yaml           # SLM training config
|   +-- yolo_chart_v3.yaml      # YOLO training config
|   +-- secrets/                # API keys (gitignored)
|
+-- data/
|   +-- academic_dataset/       # Arxiv chart images + metadata
|   |   +-- chart_qa_v2/        # QA pairs per chart type
|   |   +-- classified_charts/  # Classified chart images
|   |   +-- detected_charts/    # YOLO-detected charts
|   |   +-- images/             # Raw extracted images
|   |   +-- manifests/          # Processing manifests
|   |   +-- metadata/           # Image metadata
|   |   +-- stage3_features/    # Extracted features per chart
|   +-- cache/                  # OCR cache (46,910 entries)
|   +-- output/                 # Pipeline outputs
|   +-- raw_pdfs/               # Input PDF files
|   +-- samples/                # Demo/test samples
|   +-- slm_training_v2/       # Training dataset v2 (32k)
|   +-- slm_training_v3/       # Training dataset v3 (268,799 samples)
|   +-- yolo_chart_detection/   # YOLO training data
|
+-- docs/
|   +-- MASTER_CONTEXT.md       # This file
|   +-- CHANGELOG.md            # Change log
|   +-- architecture/           # System design docs
|   |   +-- PIPELINE_FLOW.md
|   |   +-- STAGE3_EXTRACTION.md
|   |   +-- STAGE4_REASONING.md
|   |   +-- STAGE5_REPORTING.md
|   |   +-- SYSTEM_OVERVIEW.md
|   +-- guides/                 # How-to guides
|   |   +-- QUICK_START.md
|   |   +-- DEVELOPMENT.md
|   |   +-- TRAINING.md
|   |   +-- CHART_QA_GUIDE.md
|   |   +-- ARXIV_DOWNLOAD_GUIDE.md
|   +-- progress/               # Weekly progress reports
|   +-- reports/                # Technical reports
|   +-- thesis_capstone/        # LaTeX thesis (39 pages)
|   |   +-- main.tex            # Master document (XeLaTeX)
|   |   +-- refs.bib            # Bibliography (21 entries)
|   |   +-- contents/           # 7 chapter .tex files
|   |   +-- figures/            # 7 PDFs + 12 tables + 6 TikZ diagrams
|   +-- research/               # Research methodology
|   +-- archive/                # Historical docs
|
+-- models/
|   +-- weights/                # Trained model files (YOLO, ResNet)
|   +-- onnx/                   # ONNX exports (ResNet-18: 42.64 MB)
|   +-- slm/                    # SLM LoRA adapters
|   +-- evaluation/             # Model evaluation results
|   +-- explainability/         # Grad-CAM visualizations
|
+-- notebooks/                  # 12 Jupyter notebooks
|   +-- 00_full_pipeline_test.ipynb
|   +-- 00_quick_start.ipynb
|   +-- 01_stage1_ingestion.ipynb
|   +-- 01a_data_collection.ipynb
|   +-- 01b_image_extraction.ipynb
|   +-- 01c_chart_detection.ipynb
|   +-- 01d_chart_classification.ipynb
|   +-- 01e_qa_generation.ipynb
|   +-- 01f_review_uncertain.ipynb
|   +-- 02_stage2_detection.ipynb
|   +-- 03_stage3_extraction.ipynb
|   +-- 04_stage4_reasoning.ipynb
|
+-- research/
|   +-- experiments/            # Experiment scripts
|   +-- papers/                 # Paper summaries
|   +-- prototypes/             # Quick tests
|
+-- src/
|   +-- core_engine/            # Core AI engine (48 files, ~19,800 LOC)
|       +-- __init__.py
|       +-- pipeline.py         # Main orchestrator (all 5 stages wired)
|       +-- exceptions.py       # Custom exception hierarchy
|       +-- schemas/            # Pydantic models (6 files)
|       |   +-- common.py, enums.py, extraction.py
|       |   +-- stage_outputs.py, qa_schemas.py
|       +-- stages/             # Pipeline stages
|       |   +-- base.py         # BaseStage ABC
|       |   +-- s1_ingestion.py # Stage 1: PDF/image loading
|       |   +-- s2_detection.py # Stage 2: YOLO detection
|       |   +-- s3_extraction/  # Stage 3: OCR + geometry (10 submodules)
|       |   +-- s4_reasoning/   # Stage 4: AI reasoning (6 submodules)
|       |   +-- s5_reporting.py # Stage 5: Insights + reports
|       +-- ai/                 # AI Routing Layer (8 files)
|       |   +-- router.py       # Task-based routing with fallbacks
|       |   +-- task_types.py   # TaskType enum
|       |   +-- prompts.py      # Shared prompt templates
|       |   +-- exceptions.py   # AI-specific exceptions
|       |   +-- adapters/       # Provider adapters
|       |       +-- base.py         # BaseAIAdapter ABC + AIResponse
|       |       +-- gemini_adapter.py   # Google Gemini SDK
|       |       +-- openai_adapter.py   # OpenAI Chat Completions
|       |       +-- local_slm_adapter.py # HuggingFace Transformers + LoRA
|       +-- validators/         # Output validators
|       +-- data_factory/       # QA data generation
|
+-- tests/                      # 294 tests (23 files)
|   +-- conftest.py             # Shared fixtures
|   +-- test_schemas.py         # Schema validation tests
|   +-- test_s3_extraction/     # Stage 3 tests (8 files, ~140 tests)
|   +-- test_s4_reasoning/      # Stage 4 tests (3 files, ~36 tests)
|   +-- test_s5_reporting/      # Stage 5 tests (2 files, ~62 tests)
|   +-- test_ai/                # AI routing tests (5 files, ~55 tests)
|   +-- fixtures/               # Test data
|
+-- scripts/                    # Utility scripts (4 subdirs, 19 files)
|   +-- training/               # SLM/YOLO/ResNet training, data prep
|   |   +-- train_slm_lora.py
|   |   +-- train_resnet18_v2.py
|   |   +-- train_yolo_chart_detector.py
|   |   +-- prepare_slm_training_v3.py
|   |   +-- extract_mini_dataset.py
|   |   +-- run_model_selection.py
|   |   +-- setup_cloud_training.sh
|   +-- evaluation/             # Model evaluation, ONNX export
|   |   +-- evaluate_slm.py
|   |   +-- evaluate_resnet18.py
|   |   +-- export_resnet18_onnx.py
|   +-- pipeline/               # Pipeline testing, demo, batch
|   |   +-- batch_stage3_parallel.py
|   |   +-- demo_full_pipeline.py
|   |   +-- test_*.py (5 files)
|   +-- utils/                  # Download, audit, thesis generation
|       +-- download_models.py
|       +-- _full_audit.py
|       +-- generate_thesis_*.py (2 files)
|       +-- context_scanner.py
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

### 5.2. Phase 2: Core Engine [COMPLETED]

| Task | Status | Notes |
| --- | --- | --- |
| Stage 3: Extraction | [DONE] | Geo-SLM hybrid approach implemented |
| Stage 3 Testing | [DONE] | Tested on 800+ images (100% accuracy) |
| Stage 4: Reasoning | [DONE] | Core + RouterEngine + AI adapters |
| Stage 5: Reporting | [DONE] | Insights, validation, JSON + text output |
| AI Router + Adapters | [DONE] | 4 adapters, fallback chains, health checks |
| Pipeline Wiring | [DONE] | All 5 stages live-instantiated from config |

**Stage 4 Implementation Details:**

| Submodule | Status | Description |
| --- | --- | --- |
| GeometricValueMapper | [DONE] | Pixel-to-value conversion using AxisInfo calibration |
| GeminiPromptBuilder | [DONE] | Canonical Format prompts with anti-hallucination |
| ReasoningEngine | [DONE] | Orchestrator integrating mapper + builder |
| RouterEngine | [DONE] | AIRouter bridge implementing ReasoningEngine ABC |
| Prompt Templates | [DONE] | reasoning.txt + canonical_format.md |
| Unit Tests | [DONE] | 36 test cases (16 mapper + 20 builder) |
| Gemini Integration | [DONE] | Google Generative AI SDK adapter with vision |
| OpenAI Integration | [DONE] | Chat Completions adapter with vision |
| Local SLM | [DONE] | HuggingFace adapter (4-bit quantization, LoRA) |

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

### 5.3. Phase 2b: Data Pipeline & SLM Dataset [COMPLETED - v3.0.0]

Completed 2026-03-01. Stage 3 extraction complete, training dataset v3 built and validated.

| Task | Status | Notes |
| --- | --- | --- |
| Stage 3 batch extraction | [DONE] | 32,364/32,364 files, 0% corruption, atomic write + float cast fix |
| Stage 3 quality audit | [DONE] | All 8 types healthy, axis info 69.9% coverage |
| QA cross-validation | [DONE] | 87% OCR text overlap between Stage3 features and QA answers |
| SLM training v3 build | [DONE] | 268,799 samples via `prepare_slm_training_v3.py` (dry-run validated) |
| Data pipeline report | [DONE] | `docs/reports/data_pipeline_report_v1.md` (9 sections, academic grade) |
| Instructions update | [DONE] | docs.instructions.md, module-training.instructions.md updated to v1.1.0 |

**V3 Dataset Breakdown:**

| Type | Samples | Notes |
| --- | --- | --- |
| line | 108,419 | Highest volume |
| scatter | 52,163 | |
| bar | 47,330 | |
| heatmap | 33,373 | |
| box | 13,948 | |
| pie | 7,408 | |
| histogram | 4,159 | |
| area | 1,999 | Fewest (617 source charts) |
| **Total** | **268,799** | 9.9x increase over v2 |

**Build command:**
```bash
.venv/Scripts/python.exe scripts/training/prepare_slm_training_v3.py --output-dir data/slm_training_v3
```

### 5.4. Phase 3: Production Architecture [PARTIALLY COMPLETED - v2.0.0]

Added 2026-02-28. Architecture upgrade based on gap analysis with elixverse-platform.

| Task | Status | Notes |
| --- | --- | --- |
| AI Adapter Pattern design | [DONE] | BaseAIAdapter ABC, GeminiAdapter, LocalSLMAdapter, OpenAIAdapter |
| AI Router with fallback chains | [DONE] | Confidence-based routing, health checks, local-only policy |
| AI Routing implementation | [DONE] | `src/core_engine/ai/` - 8 files fully implemented |
| AI Routing tests | [DONE] | `tests/test_ai/` - 55 unit tests, all mock-based |
| SLM Training Framework design | [DONE] | Qwen-2.5-1.5B + LoRA, 4-stage curriculum |
| Training scripts | [DONE] | train_slm_lora.py (320L), prepare_slm_training_v3.py |
| Training data | [DONE] | 268,799 samples (all 8 types), saved to `data/slm_training_v3/` |
| Stage 5 implementation | [DONE] | Insights, validation, JSON + text report output |
| Pipeline wiring | [DONE] | All 5 stages live in `pipeline.py`, sequential execution |
| Serving Layer design | [DONE] | FastAPI + Celery + Redis + SQLAlchemy |
| CI/CD Pipeline design | [DONE] | workflow.instructions.md with YAML specs |
| Instruction system upgrade | [DONE] | 3-tier hierarchy, 13 instruction files |
| `src/api/` creation | [TODO] | Implement FastAPI endpoints |
| `src/worker/` creation | [TODO] | Implement Celery tasks |
| Docker Compose setup | [TODO] | Multi-container deployment |
| SLM model training | [REDO] | Llama-3.2-1B v3 had 4 critical config bugs, retrain as v4 on cloud GPU |
| SLM evaluation framework | [DONE] | `evaluate_slm.py` with EM, Contains, Numeric, BLEU-1 |
| Model comparison experiment | [IN PROGRESS] | Micro-training pipeline ready, awaiting cloud GPU execution |
| Training postmortem | [DONE] | 4 bugs found and fixed, cloud GPU strategy documented |
| Model selection pipeline | [DONE] | `extract_mini_dataset.py`, `run_model_selection.py`, `setup_cloud_training.sh` |
| Mini-dataset extraction | [DONE] | 5000 train, 500 val, stratified by chart_type x question_type |
| Training config correction | [DONE] | training.yaml updated: max_length=4096, bf16=true, grad_accum=8 |

**New Instruction Files Created:**
- `module-training.instructions.md` - SLM fine-tuning framework
- `module-reasoning.instructions.md` - AI adapter pattern, routing
- `module-serving.instructions.md` - API, task queue, deployment
- `module-detection.instructions.md` - YOLO detection stage
- `module-extraction.instructions.md` - OCR, classification stage
- `workflow.instructions.md` - CI/CD, Git, releases

**Key Architecture Documents:**
- [UPGRADE_REPORT_PRODUCTION_READY.md](reports/UPGRADE_REPORT_PRODUCTION_READY.md) - Full gap analysis
- `.github/instructions/README.md` - Instruction hierarchy overview

### 5.5. Phase 4: Academic Thesis [COMPLETED - 2026-03-02]

| Task | Status | Notes |
| --- | --- | --- |
| Thesis structure design | [DONE] | 7 chapters, XeLaTeX build, FPT University template |
| Chapter 1: Introduction | [DONE] | Problem statement, objectives, scope, Vietnamese integration |
| Chapter 2: Literature Review | [DONE] | 21 references, related work comparison table |
| Chapter 3: Methodology | [DONE] | Hybrid pipeline design, geometric analysis, AI routing |
| Chapter 4: System Design | [DONE] | Architecture, data flow, AI Router, database schema |
| Chapter 5: Results | [DONE] | ResNet-18, YOLO, dataset stats, feature quality analysis |
| Chapter 6: Project Management | [DONE] | Timeline, Git statistics, resource allocation |
| Chapter 7: Conclusion | [DONE] | Summary, contributions, future work, Vietnamese research |
| Visual assets | [DONE] | 7 PDF figures + 12 LaTeX tables + 6 TikZ diagrams |
| Bibliography | [DONE] | refs.bib with 21 entries |
| LaTeX compilation | [DONE] | 0 errors, 0 undefined references, 39 pages |
| Vietnamese content | [DONE] | Core-first Localize-second architecture documented |

**Thesis Asset Count:**

| Type | Count | Location |
| --- | --- | --- |
| Content chapters (.tex) | 7 | `docs/thesis_capstone/contents/` |
| PDF figures | 7 | `docs/thesis_capstone/figures/` |
| LaTeX tables | 12 | `docs/thesis_capstone/figures/tables/` |
| TikZ diagrams | 6 | `docs/thesis_capstone/figures/tikz/` |
| Bibliography entries | 21 | `docs/thesis_capstone/refs.bib` |
| Total pages | 39 | `docs/thesis_capstone/main.pdf` |

### 5.6. Upcoming Phases

| Phase | Focus | Timeline |
| --- | --- | --- |
| **Model Selection** | Micro-train 2-4 models on mini-dataset, compare, pick winner | CURRENT |
| **SLM Full Train** | Full 3-epoch QLoRA on 228k samples (cloud GPU) | After selection |
| **Serving Layer** | FastAPI + Celery + Docker | After SLM |
| **Benchmarking** | Model comparison experiment (thesis contrib.) | Final |
| **Optimization** | Performance tuning, demo interface (Streamlit) | Final |

**Model Selection Pipeline (Eval for Selection Strategy):**

| Step | Script | Description |
| --- | --- | --- |
| 1. Extract mini-dataset | `scripts/training/extract_mini_dataset.py` | 5000 train + 500 val, stratified |
| 2. Setup cloud instance | `scripts/training/setup_cloud_training.sh` | Automated GPU env setup |
| 3. Run model selection | `scripts/training/run_model_selection.py` | Train+eval 2-4 candidates |
| 4. Pick winner | Manual (read comparison table) | Based on EM, accuracy, latency |
| 5. Full training | `scripts/training/train_slm_lora.py` | 3 epochs on full 228k dataset |

**Mini-Dataset (data/slm_training_mini/):**

| Split | Samples | Source |
| --- | --- | --- |
| train_mini.json | 5,000 | Stratified from 228,494 |
| val_mini.json | 500 | Stratified from 26,888 |
| test.json | 13,417 | Shared with v3 (unchanged) |

**SLM Training Plan (Revised after Postmortem):**

| Item | Description |
| --- | --- |
| Base Model | Llama-3.2-1B-Instruct (PRIMARY), Qwen-2.5-1.5B (SECONDARY) |
| Current Data | 268,799 samples (all 8 types, 69.9% with axis info) |
| Data Location | `data/slm_training_v3/` (train/val/test split by chart_id) |
| Training Type | QLoRA (4-bit quantization + LoRA rank 16, alpha auto=rank*2) |
| max_length | **4096** (was 512, caused fatal bug in v3) |
| Hardware | Cloud GPU - A100 40GB (RunPod/Vast.ai), ~$1-3/hr |
| Expected Output | LoRA adapter (~60MB) saved to `llama-3.2-1b-chart-lora-v4/final/` |
| Evaluation | EM >40%, Contains >60%, Numeric >50%, latency <2s |
| Postmortem | See `docs/reports/SLM_TRAINING_POSTMORTEM_V1.md` |
| Training Guide | See `docs/guides/TRAINING.md` v1.0.0 |

### 5.6. OCR Cache Status (2026-02-04)

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

### 5.8. Test Coverage (2026-03-02)

| Test Suite | Tests | Files | Notes |
| --- | --- | --- | --- |
| Schemas | ~19 | 1 | Schema validation |
| Stage 3 Extraction | ~140 | 8 | OCR, geometry, elements, integration |
| Stage 4 Reasoning | ~36 | 3 | ValueMapper, PromptBuilder |
| Stage 5 Reporting | ~62 | 2 | Insights, validation, output formats |
| AI Routing | ~55 | 5 | Adapters, router, task types, prompts, exceptions |
| **Total** | **294** | **23** | **All passing** |

### 5.8. Project Cleanup (2026-02-04)

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

### 5.9. Project Cleanup (2026-03-02)

**Deleted (~1.5GB freed):**
- Broken LoRA training artifacts: `llama-3.2-1b-chart-lora-v3/` checkpoints + final (182MB)
- Incomplete experiments: `llama-3.2-1b-instruct-chart-lora/` (61MB), `qwen2.5-1.5b-instruct-chart-lora/` (83MB)
- Buggy Qwen LoRA: `qwen2.5-0.5b-instruct-chart-lora/` (137MB, trained with max_length=512)
- Unused base model: `qwen2.5-0.5b-instruct/` (954MB)
- Invalid eval results: `llama-1b-lora-v3*.json`
- Old dataset: `data/slm_training_v2/` (54MB, replaced by v3)
- Legacy code: `Chart_QA/` (52MB), `scripts/_archive/` (6 scripts)
- Misc: `runs/`, `logs/*.log`, `docs/archive/chatlog*.md`, `nul`, `instruction_Mar01.md`

**Reason**: First training session revealed critical config bugs. All artifacts from that run are invalid. See `docs/reports/SLM_TRAINING_POSTMORTEM_V1.md`.

**Stage 5 Architecture (Implemented):**

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
| AI Adapter Pattern | Decouple providers from pipeline (from elix) | 2026-02-28 |
| AI Router with fallbacks | Reliability: local_slm → gemini → openai | 2026-02-28 |
| QLoRA for SLM training | Fit Qwen-2.5-1.5B in RTX 3060 6GB | 2026-02-28 |
| 4-stage curriculum learning | Progressive difficulty for small dataset | 2026-02-28 |
| Celery + Redis task queue | Async processing (from elix pattern) | 2026-02-28 |
| 3-tier instruction hierarchy | Scalable AI agent guidance system | 2026-02-28 |
| Cloud GPU for SLM training | Local 6GB too constrained for max_length=4096 | 2026-03-02 |
| max_length=4096 for SFT | Ensure model sees full ground truth during training | 2026-03-02 |

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
