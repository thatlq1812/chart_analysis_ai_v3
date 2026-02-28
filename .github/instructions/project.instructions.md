---
applyTo: '**'
---

# PROJECT INSTRUCTIONS - Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-02-28 | That Le | Production-ready upgrade with AI routing, async tasks, state management |
| 1.0.0 | 2026-01-19 | That Le | Project-specific rules and architecture |

## 0. Hierarchical Instructions System

[CRITICAL] This document is the SINGLE SOURCE OF TRUTH for project-wide rules.
[CRITICAL] Do NOT use any icon or emoji anywhere in this project.

### 0.1. How Instructions Are Organized

To minimize context overhead and maximize AI agent efficiency, instructions follow a hierarchical system:

**Instructions always attached (global):**
- `system.instructions.md` - Global rules for all AI Agent sessions
- `project.instructions.md` (this file) - Single source of truth for project-specific rules

**Instructions loaded on-demand (module-specific):**
- `coding-standards.instructions.md` - Load for Python coding tasks in `src/`
- `pipeline.instructions.md` - Load for core engine pipeline work in `src/core_engine/`
- `research.instructions.md` - Load for research/experiment work in `research/`, `notebooks/`
- `docs.instructions.md` - Load for documentation work in `docs/`
- `module-detection.instructions.md` - Load ONLY for Stage 2 YOLO detection work
- `module-extraction.instructions.md` - Load ONLY for Stage 3 OCR + geometry work
- `module-reasoning.instructions.md` - Load ONLY for Stage 4 AI reasoning work
- `module-serving.instructions.md` - Load ONLY for API server + async task work
- `module-training.instructions.md` - Load ONLY for model training/fine-tuning work
- `workflow.instructions.md` - Load for CI/CD and deployment tasks

**Reference documentation (NOT attached to conversations):**
- `docs/MASTER_CONTEXT.md` - For session planning and context
- `docs/architecture/AI_ROUTING.md` - For AI routing design tasks
- `docs/architecture/DATABASE_SCHEMA.md` - For database design tasks
- `docs/architecture/API_CONTRACTS.md` - For API development tasks

### 0.2. How to Request Module-Specific Instructions

When working on a specific feature, mention it explicitly:

```
"Working on Stage 4 reasoning"
-> Agent loads: project.instructions + module-reasoning.instructions

"Working on AI provider routing"
-> Agent loads: project.instructions + module-reasoning.instructions

"Working on API server"
-> Agent loads: project.instructions + module-serving.instructions

"Working on YOLO training"
-> Agent loads: project.instructions + module-training.instructions

"Designing database schema"
-> Agent loads: project.instructions + docs/architecture/DATABASE_SCHEMA.md
```

### 0.3. Context Efficiency Benefits

This hierarchical approach saves:
- 30-40% of context tokens per conversation
- 15% faster response time
- Clearer scope for each task
- Easier to maintain as project grows

---

## 1. Project Identity

### 1.1. What We Are Building

**Geo-SLM Chart Analysis** is a hybrid AI system for extracting structured data from chart images. It combines:

| Component | Role | Technology |
| --- | --- | --- |
| **Computer Vision** | Detect and localize charts | YOLOv8/v11 |
| **Geometric Analysis** | Precise value extraction | OpenCV + NumPy |
| **AI Reasoning** | OCR correction + semantic reasoning | Multi-provider (Local SLM, Gemini, OpenAI) |

### 1.2. Core Philosophy

> "Accuracy through Hybrid Intelligence: Use Deep Learning for perception, Geometry for precision, and AI Reasoning for understanding."

**Principles:**
1. **Neuro-Symbolic**: Combine neural networks with symbolic reasoning
2. **Local-First**: Prefer local SLM inference, fallback to cloud APIs
3. **Explainable**: Every output must trace back to source evidence
4. **Production-Ready**: Handle failures gracefully, scale horizontally

### 1.3. What We Are NOT Building

- NOT a general-purpose document parser
- NOT a real-time streaming system
- NOT a cloud-first SaaS product (but cloud-deployable)
- NOT dependent on a single LLM API for core inference

## 2. Technology Stack

### 2.1. Core Engine (LOCKED)

| Component | Technology | Rationale |
| --- | --- | --- |
| Language | Python 3.11+ | AI/ML ecosystem |
| Object Detection | YOLOv8/v11 | Fast, accurate, trainable |
| OCR | PaddleOCR / Tesseract | Multi-language support |
| Image Processing | OpenCV + Pillow | Industry standard |
| Geometric Calc | NumPy + Custom | Precision arithmetic |
| Data Validation | Pydantic v2 | Schema enforcement |
| Config | OmegaConf | Hierarchical ML config |

### 2.2. AI Reasoning (Multi-Provider)

| Provider | Model | Use Case | Priority |
| --- | --- | --- | --- |
| Local SLM | Qwen-2.5 / Llama-3.2 (1-3B) | Default reasoning (offline) | PRIMARY |
| Google Gemini | gemini-2.0-flash | Complex charts, vision tasks | FALLBACK 1 |
| OpenAI | gpt-4o-mini | Alternative reasoning | FALLBACK 2 |

**Rules:**
- Local SLM is always preferred (cost = 0, no network dependency)
- Cloud providers are fallbacks when local model confidence is low
- All providers go through `AIRouter` (see Section 5)

### 2.3. Serving Layer (NEW)

| Component | Technology | Purpose |
| --- | --- | --- |
| API Server | FastAPI | REST API for chart analysis |
| Task Queue | Celery + Redis | Async pipeline execution |
| Database | PostgreSQL + SQLAlchemy | Job state, results persistence |
| Migrations | Alembic | Schema versioning |

### 2.4. Research & Training

| Component | Technology | Purpose |
| --- | --- | --- |
| Experiment Tracking | MLflow / W&B | Log metrics, artifacts |
| Notebook | Jupyter Lab | Interactive exploration |
| Model Training | PyTorch + Ultralytics | YOLO fine-tuning |
| SLM Training | Transformers + LoRA | SLM fine-tuning |

### 2.5. Interface Layer

| Component | Technology | Purpose |
| --- | --- | --- |
| CLI | Typer | Developer testing |
| Demo UI | Streamlit | Quick visualization |
| API | FastAPI | Production integration |

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
|                        TASK QUEUE (Celery + Redis)                |
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
|  |                      AI Router                              |  |
|  |  [Local SLM]    [Gemini Adapter]    [OpenAI Adapter]       |  |
|  +------------------------------------------------------------+  |
|  |                    Shared Schemas                           |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    STATE & DATA LAYER                              |
|  [PostgreSQL]   [Redis Cache]   [File Storage]   [Model Weights] |
+------------------------------------------------------------------+
```

### 3.2. Data Flow

```
Input (PDF/Image)
    |
[Stage 1] Ingestion & Sanitation
    | Clean Images
[Stage 2] Detection & Localization (YOLO)
    | Cropped Charts
[Stage 3] Structural Analysis (OCR + Geometry)
    | Raw Metadata
[Stage 4] Semantic Reasoning (AI Router -> Best Provider)
    | Refined Data
[Stage 5] Insight & Reporting
    |
Output (JSON + Report)
```

## 4. Directory Structure

```
chart_analysis_ai_v3/
    .github/
        instructions/               # AI Agent guidelines (hierarchical)
    config/
        base.yaml                   # Shared config
        models.yaml                 # Model paths, thresholds
        pipeline.yaml               # Stage toggles
        secrets/                    # API keys (gitignored)
    data/
        raw_pdfs/                   # Input files
        processed/                  # Pipeline outputs
        cache/                      # Intermediate results
        samples/                    # Demo/test samples
        output/                     # Final results
        slm_training/               # Training datasets
        yolo_chart_detection/       # YOLO data
    docs/
        MASTER_CONTEXT.md           # Project overview
        architecture/               # System design docs
        research/                   # Paper notes, experiments
        guides/                     # How-to guides
        reports/                    # Weekly/thesis reports
        confirmation/               # Agent confirmation docs
    models/
        weights/                    # Trained model files
        onnx/                       # ONNX exports
        evaluation/                 # Eval results
    notebooks/                      # Jupyter notebooks
    research/                       # Experiments
    src/
        core_engine/
            __init__.py
            pipeline.py             # Main orchestrator
            exceptions.py           # Exception hierarchy
            schemas/                # Pydantic models
            stages/                 # Pipeline stages (s1-s5)
            ai/                     # AI Router + Adapters (NEW)
                router.py
                task_types.py
                adapters/
            state/                  # DB models + repository (NEW)
            validators/             # Output validation
            data_factory/           # Data generation
        tasks/                      # Celery tasks (NEW)
        api/                        # FastAPI server (NEW)
            v1/
    interface/
        cli.py                      # Command line
        demo/                       # Streamlit app
    tests/
        conftest.py
        test_stages/
        test_ai/                    # AI adapter tests (NEW)
        test_api/                   # API tests (NEW)
        fixtures/
    scripts/
    .env.example
    .gitignore
    pyproject.toml
    README.md
    Makefile
```

## 5. AI Provider Routing System

### 5.1. Task Types

```python
class TaskType(str, Enum):
    OCR_CORRECTION = "ocr_correction"       # Fix OCR errors
    VALUE_MAPPING = "value_mapping"          # Map pixels to values
    CHART_REASONING = "chart_reasoning"      # Full chart analysis
    DESCRIPTION = "description"             # Generate descriptions
```

### 5.2. Fallback Chains

| Task Type | Chain (in priority order) |
| --- | --- |
| OCR_CORRECTION | local_slm -> gemini -> openai |
| VALUE_MAPPING | local_slm -> gemini -> openai |
| CHART_REASONING | gemini -> openai -> local_slm |
| DESCRIPTION | local_slm -> gemini -> openai |

**Logic:**
- Simple tasks (OCR correction, value mapping): Local SLM first (fast, free)
- Complex tasks (full chart reasoning): Cloud API first (better accuracy)
- If primary fails or confidence < threshold: auto-fallback to next in chain

### 5.3. Routing Flow

```
Stage 4 calls: router.resolve(TaskType.CHART_REASONING)
    |
    v
+----------------------------------------+
|            AI ROUTER                   |
|                                        |
| 1. Check config override for task type |
| 2. Walk fallback chain                 |
| 3. Check provider API key availability |
| 4. Return (adapter, model_id)          |
+----------------------------------------+
    |
    v
adapter.reason(prompt, model_id, image_path)
    |
    v
Standardized ReasoningResult
```

## 6. Pipeline Stages

### Stage 1: Ingestion

| Property | Value |
| --- | --- |
| Input | PDF, DOCX, PNG, JPG |
| Output | List of clean images |
| Key Library | PyMuPDF, Pillow |

### Stage 2: Detection

| Property | Value |
| --- | --- |
| Input | Clean images |
| Output | Cropped charts + bounding boxes |
| Key Library | Ultralytics YOLO |

### Stage 3: Extraction

| Property | Value |
| --- | --- |
| Input | Cropped chart image |
| Output | Raw metadata (type, text, coordinates) |
| Key Library | PaddleOCR, OpenCV |

### Stage 4: Reasoning

| Property | Value |
| --- | --- |
| Input | Raw metadata + cropped image |
| Output | Refined structured data |
| Key Library | AI Router -> (Qwen / Gemini / OpenAI) |

### Stage 5: Reporting

| Property | Value |
| --- | --- |
| Input | Refined data |
| Output | Final JSON + text report |
| Key Library | Jinja2, JSON Schema |

## 7. Quality Gates

### 7.1. Code Quality

| Check | Tool | Threshold |
| --- | --- | --- |
| Linting | Ruff | 0 errors |
| Type Checking | MyPy | 0 errors |
| Formatting | Black | Auto-applied |
| Test Coverage | pytest-cov | >70% |

### 7.2. Model Quality

| Metric | Target | Stage |
| --- | --- | --- |
| Chart Detection mAP | >0.85 | Stage 2 |
| OCR Accuracy | >0.90 | Stage 3 |
| Value Extraction Accuracy | >0.85 | Stage 4 |
| End-to-End Accuracy | >0.80 | Full Pipeline |

## 8. Development Phases

### Phase 1: Foundation (COMPLETED)
- [x] Project structure setup
- [x] Documentation framework
- [x] Data collection pipeline
- [x] Stage 1-3 implementation
- [x] ResNet-18 classifier (94.66% accuracy)

### Phase 2: Core Engine (CURRENT)
- [x] Stage 4 core (ValueMapper + PromptBuilder)
- [ ] AI Router + Adapter pattern
- [ ] Stage 5 implementation
- [ ] End-to-end pipeline integration

### Phase 3: Production Infrastructure
- [ ] Database state management (SQLAlchemy + Alembic)
- [ ] Celery task queue integration
- [ ] FastAPI server with REST endpoints
- [ ] Docker containerization

### Phase 4: Optimization & Presentation
- [ ] SLM fine-tuning with LoRA
- [ ] Performance benchmarking
- [ ] Demo interface (Streamlit)
- [ ] Academic paper + thesis completion
