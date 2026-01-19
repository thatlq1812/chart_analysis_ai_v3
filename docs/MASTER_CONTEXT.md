# MASTER CONTEXT - Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
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
| **Current Phase** | Phase 1 - Foundation |
| **Target** | Academic Thesis + Research Paper |

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
|
+-- config/
|   +-- base.yaml               # Shared configuration
|   +-- models.yaml             # Model paths & thresholds
|   +-- pipeline.yaml           # Stage configuration
|   +-- secrets/                # API keys (gitignored)
|
+-- data/
|   +-- raw/                    # Input files (PDF, images)
|   +-- processed/              # Pipeline outputs
|   +-- cache/                  # Intermediate results
|   +-- training/               # Training datasets
|   +-- samples/                # Demo/test samples
|
+-- docs/
|   +-- MASTER_CONTEXT.md       # This file
|   +-- architecture/           # System design docs
|   +-- research/               # Paper notes, experiments
|   +-- guides/                 # How-to guides
|   +-- reports/                # Weekly/thesis reports
|
+-- models/
|   +-- weights/                # Trained model files
|   +-- registry/               # Model versioning
|
+-- notebooks/
|   +-- 01_exploration.ipynb    # Data exploration
|   +-- 02_training.ipynb       # Model training
|   +-- 03_evaluation.ipynb     # Result analysis
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
|       +-- utils/              # Helper functions
|
+-- interface/
|   +-- cli.py                  # Command line interface
|   +-- api/                    # FastAPI server
|   +-- demo/                   # Streamlit app
|
+-- tests/
|   +-- conftest.py             # Shared fixtures
|   +-- test_schemas/
|   +-- test_stages/
|   +-- fixtures/               # Test data
|
+-- scripts/
|   +-- setup_env.py            # Environment setup
|   +-- download_models.py      # Download weights
|
+-- .env.example
+-- .gitignore
+-- pyproject.toml
+-- README.md
+-- Makefile
```

---

## 5. Current Status

### 5.1. Phase 1: Foundation [IN PROGRESS]

| Task | Status | Notes |
| --- | --- | --- |
| Project structure setup | [DONE] | V3 initialized |
| Documentation framework | [DONE] | Instructions created |
| Stage 1: Ingestion | [TODO] | Next priority |
| Stage 2: Detection | [TODO] | After Stage 1 |

### 5.2. Upcoming Phases

| Phase | Focus | Timeline |
| --- | --- | --- |
| Phase 2 | Core Engine (Stages 3-5) | Week 2-4 |
| Phase 3 | Optimization & Benchmarking | Week 5-6 |
| Phase 4 | Demo & Documentation | Week 7-8 |

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
