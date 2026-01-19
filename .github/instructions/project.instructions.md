---
applyTo: '**'
---

# PROJECT INSTRUCTIONS - Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-19 | That Le | Project-specific rules and architecture |

## 1. Project Identity

### 1.1. What We Are Building

**Geo-SLM Chart Analysis** is a hybrid AI system for extracting structured data from chart images. It combines:

- **Computer Vision**: YOLO for detection, OpenCV for preprocessing
- **Geometric Analysis**: Classical algorithms for precise value extraction
- **Small Language Models (SLM)**: Specialized model for OCR correction and semantic reasoning

### 1.2. Core Philosophy

> "Accuracy through Hybrid Intelligence: Use Deep Learning for perception, Geometry for precision, and SLM for reasoning."

**Principles:**
1. **Neuro-Symbolic**: Combine neural networks with symbolic reasoning
2. **Local-First**: Minimize API dependencies, run offline when possible
3. **Explainable**: Every output must trace back to source evidence
4. **Academic Rigor**: Results must be reproducible and documented

### 1.3. What We Are NOT Building

- NOT a general-purpose document parser
- NOT a real-time streaming system
- NOT a cloud-first SaaS product
- NOT dependent on external LLM APIs for core inference

## 2. Technology Stack

### 2.1. Core Engine (LOCKED)

| Component | Technology | Rationale |
| --- | --- | --- |
| Language | Python 3.11+ | AI/ML ecosystem |
| Object Detection | YOLOv8/v11 | Fast, accurate, trainable |
| OCR | PaddleOCR / Tesseract | Multi-language support |
| Image Processing | OpenCV + Pillow | Industry standard |
| Geometric Calc | NumPy + Custom | Precision arithmetic |
| SLM | Qwen-2.5 / Llama-3.2 (1-3B) | Local inference |
| Data Validation | Pydantic v2 | Schema enforcement |

### 2.2. Research & Training

| Component | Technology | Purpose |
| --- | --- | --- |
| Experiment Tracking | MLflow / Weights & Biases | Log metrics, artifacts |
| Notebook | Jupyter Lab | Interactive exploration |
| Dataset Management | DVC (optional) | Version control for data |
| Model Training | PyTorch + Ultralytics | YOLO fine-tuning |

### 2.3. Interface Layer (Optional)

| Component | Technology | Purpose |
| --- | --- | --- |
| CLI | Typer / Click | Developer testing |
| Demo UI | Streamlit | Quick visualization |
| API Server | FastAPI | Production integration |
| Documentation | MkDocs | Static site |

## 3. Architecture Overview

### 3.1. Layer Diagram

```
+------------------------------------------------------------------+
|                        INTERFACE LAYER                            |
|  [CLI]        [Streamlit Demo]        [FastAPI Server]           |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        CORE ENGINE                                |
|  +------------------------------------------------------------+  |
|  |                    Pipeline Orchestrator                    |  |
|  +------------------------------------------------------------+  |
|  |  Stage 1  |  Stage 2  |  Stage 3  |  Stage 4  |  Stage 5   |  |
|  | Ingestion | Detection | Extraction| Reasoning | Reporting  |  |
|  +------------------------------------------------------------+  |
|  |                      Shared Schemas                         |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                        DATA LAYER                                 |
|  [Raw Files]    [Processed]    [Cache]    [Models]    [Outputs]  |
+------------------------------------------------------------------+
```

### 3.2. Data Flow (The Master Pipeline)

See [docs/architecture/PIPELINE_FLOW.md](../../docs/architecture/PIPELINE_FLOW.md) for detailed mermaid diagrams.

**Summary:**
```
Input (PDF/Image)
    ↓
[Stage 1] Ingestion & Sanitation
    ↓ Clean Images
[Stage 2] Detection & Localization (YOLO)
    ↓ Cropped Charts
[Stage 3] Structural Analysis (OCR + Geometry)
    ↓ Raw Metadata
[Stage 4] Semantic Reasoning (SLM)
    ↓ Refined Data
[Stage 5] Insight & Reporting
    ↓
Output (JSON + Report)
```

## 4. Directory Structure

```
chart_analysis_ai_v3/
├── .github/
│   └── instructions/           # AI Agent guidelines
├── config/
│   ├── base.yaml               # Shared config
│   ├── models.yaml             # Model paths, thresholds
│   ├── pipeline.yaml           # Stage toggles
│   └── secrets/                # API keys (gitignored)
├── data/
│   ├── raw/                    # Input files
│   ├── processed/              # Pipeline outputs
│   ├── cache/                  # Intermediate results
│   ├── training/               # Training datasets
│   └── samples/                # Demo/test samples
├── docs/
│   ├── MASTER_CONTEXT.md       # Project overview
│   ├── architecture/           # System design docs
│   ├── research/               # Paper notes, experiments
│   └── reports/                # Weekly/thesis reports
├── models/
│   ├── weights/                # Trained model files
│   └── registry/               # Model versioning
├── notebooks/
│   ├── 01_exploration.ipynb    # Data exploration
│   ├── 02_training.ipynb       # Model training
│   └── 03_evaluation.ipynb     # Result analysis
├── research/
│   ├── experiments/            # Experiment scripts
│   ├── papers/                 # Paper summaries
│   └── prototypes/             # Quick tests
├── src/
│   └── core_engine/
│       ├── __init__.py
│       ├── pipeline.py         # Main orchestrator
│       ├── schemas/            # Pydantic models
│       └── stages/             # Pipeline stages
├── interface/
│   ├── cli.py                  # Command line
│   ├── api/                    # FastAPI server
│   └── demo/                   # Streamlit app
├── tests/
│   ├── conftest.py
│   ├── test_stages/
│   └── fixtures/
├── scripts/
│   ├── setup_env.py
│   └── download_models.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── Makefile
```

## 5. Pipeline Stages (The 5-Stage Model)

### Stage 1: Ingestion & Sanitation

| Property | Value |
| --- | --- |
| Input | PDF, DOCX, PNG, JPG |
| Output | List of clean images (normalized) |
| Key Library | PyMuPDF (fitz), Pillow |
| Config | `config/pipeline.yaml#ingestion` |

**Responsibilities:**
- Convert PDF/DOCX pages to images
- Validate image quality (resolution, blur detection)
- Normalize dimensions, generate session IDs
- Preserve original color (for legend mapping later)

### Stage 2: Detection & Localization

| Property | Value |
| --- | --- |
| Input | Clean images from Stage 1 |
| Output | Cropped chart images + bounding boxes |
| Key Library | Ultralytics YOLO |
| Config | `config/models.yaml#yolo` |

**Responsibilities:**
- Detect chart regions using YOLO
- Crop each detected chart
- Store bounding box coordinates (for context extraction)
- Handle multi-chart images

### Stage 3: Structural Analysis (Hybrid)

| Property | Value |
| --- | --- |
| Input | Cropped chart image |
| Output | Raw metadata (type, text, coordinates) |
| Key Library | PaddleOCR, OpenCV, NumPy |
| Config | `config/pipeline.yaml#extraction` |

**Sub-components:**
1. **Classification**: Identify chart type (bar, line, pie)
2. **OCR Extraction**: Extract all text (title, labels, legend)
3. **Geometric Detection**: Detect bars, points, slices with coordinates

### Stage 4: Semantic Reasoning (SLM)

| Property | Value |
| --- | --- |
| Input | Raw metadata + cropped image |
| Output | Refined structured data |
| Key Library | Transformers, Local SLM |
| Config | `config/models.yaml#slm` |

**Responsibilities:**
- Correct OCR errors using context
- Map pixel coordinates to actual values
- Associate legend items with chart elements
- Generate academic-style descriptions

### Stage 5: Insight & Reporting

| Property | Value |
| --- | --- |
| Input | Refined data |
| Output | Final JSON + text report |
| Key Library | Jinja2, JSON Schema |
| Config | `config/pipeline.yaml#output` |

**Responsibilities:**
- Validate output against schema
- Generate summary insights (trends, comparisons)
- Format final JSON output
- Create human-readable report

## 6. Schema Definitions

All data flowing between stages MUST use Pydantic models defined in `src/core_engine/schemas/`.

See [pipeline.instructions.md](./pipeline.instructions.md) for detailed schema specifications.

## 7. Research Guidelines

### 7.1. Academic Value Points

| Component | Research Contribution | Papers to Cite |
| --- | --- | --- |
| Hybrid Detection | Neuro-Symbolic approach | YOLO papers, Geometric vision |
| SLM Distillation | Knowledge distillation from LLM | DistilBERT, TinyBERT papers |
| Data Augmentation | Context extraction from PDFs | Data-centric AI papers |
| Pipeline Design | Multi-stage error correction | Ensemble methods |

### 7.2. Experiment Logging

Every experiment MUST record:
- Dataset version (hash or tag)
- Model configuration (full YAML)
- Training/inference parameters
- Metrics (accuracy, precision, recall, F1)
- Failure cases analysis

## 8. Quality Gates

### 8.1. Code Quality

| Check | Tool | Threshold |
| --- | --- | --- |
| Linting | Ruff | 0 errors |
| Type Checking | MyPy | 0 errors |
| Formatting | Black | Auto-applied |
| Test Coverage | pytest-cov | >70% |

### 8.2. Model Quality

| Metric | Target | Stage |
| --- | --- | --- |
| Chart Detection mAP | >0.85 | Stage 2 |
| OCR Accuracy | >0.90 | Stage 3 |
| Value Extraction Accuracy | >0.85 | Stage 4 |
| End-to-End Accuracy | >0.80 | Full Pipeline |

## 9. Development Phases

### Phase 1: Foundation (Current)

**Data Target:** Collect minimum **1,000 chart samples** (diverse types: bar, line, pie, scatter)

- [x] Project structure setup
- [x] Documentation framework
- [ ] Data collection pipeline (web scraping, academic datasets)
- [ ] Dataset annotation (bounding boxes, chart types)
- [ ] Stage 1 implementation
- [ ] Stage 2 implementation (YOLO integration)

### Phase 2: Core Engine

- [ ] Stage 3 implementation (OCR + Geometry)
- [ ] Stage 4 implementation (SLM integration)
- [ ] Stage 5 implementation (Reporting)
- [ ] End-to-end pipeline testing

### Phase 3: Optimization

- [ ] SLM fine-tuning with distillation
- [ ] Performance benchmarking
- [ ] Error analysis and improvement
- [ ] Academic paper preparation

### Phase 4: Presentation

- [ ] Demo interface (Streamlit)
- [ ] API server (FastAPI)
- [ ] Thesis document completion
- [ ] Final defense preparation
