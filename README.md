# Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 4.0.0 | 2026-03-02 | That Le | Full pipeline implemented, thesis complete (39 pages), 232 tests passing |
| 3.3.0 | 2026-01-31 | That Le | Dataset v2 (32,445 images), ResNet-18 v2 (94.14% accuracy) |
| 3.2.0 | 2026-01-26 | That Le | ResNet-18 classifier (94.66% accuracy) production-ready |
| 3.1.0 | 2026-01-25 | That Le | Documentation restructured, Stage notebooks added |
| 3.0.0 | 2026-01-19 | That Le | Chart Analysis AI V3 |

## Overview

A **hybrid AI system** for extracting structured data from chart images, combining:

| Component | Role | Technology |
| --- | --- | --- |
| **Computer Vision** | Chart detection and localization | YOLOv8/v11 |
| **Geometric Analysis** | Precise value extraction | OpenCV + NumPy |
| **AI Reasoning** | OCR correction and semantic reasoning | Multi-provider (Local SLM, Gemini, OpenAI) |

> **Core Philosophy**: Achieve higher accuracy than pure multimodal LLMs through hybrid neuro-symbolic approach, while maintaining local inference capability.

## Current Status

| Phase | Status | Description |
| --- | --- | --- |
| Phase 1: Foundation | [COMPLETED] | Dataset: 32,364 charts, 32,445 QA pairs |
| Phase 2: Core Engine | [COMPLETED] | All 5 stages implemented, AI Router + Adapters wired |
| Phase 2b: Data Pipeline | [COMPLETED] | 268,799 SLM training samples (v3), 100% extraction |
| Phase 3: SLM Training | [IN PROGRESS] | QLoRA fine-tuning on Qwen-2.5-1.5B |
| Phase 4: Thesis | [COMPLETED] | 39-page LaTeX thesis, 0 compilation errors |

**Key Metrics:**

| Metric | Value |
| --- | --- |
| Source Files | 47 Python modules (~18,900 LOC) |
| Test Suite | 232 tests (21 test files) |
| ResNet-18 Accuracy | 94.14% (8 chart classes) |
| YOLOv8m mAP@0.5 | 93.5% |
| SLM Training Dataset | 268,799 samples (v3, all 8 types) |
| Charts Extracted | 32,364 (100% success rate) |
| Thesis | 39 pages, 7 chapters, 25 visual assets |

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/thatlq1812/chart_analysis_ai_v3.git
cd chart_analysis_ai_v3

# Create virtual environment (using uv recommended)
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from pathlib import Path
from core_engine.pipeline import Pipeline

# Run full pipeline on an image
pipeline = Pipeline()
result = pipeline.run(Path("data/samples/chart.png"))

# Or use individual stages
from core_engine.stages.s3_extraction import Stage3Extraction

stage3 = Stage3Extraction()
raw_metadata = stage3.process_single_image(Path("data/samples/chart.png"))
print(f"Chart type: {raw_metadata.chart_type}")
print(f"Elements: {len(raw_metadata.elements)}")
```

## Pipeline Architecture

```
Input (PDF/Image)
    |
    v
+-------------------+
| Stage 1: Ingest   | --> Load, Convert, Normalize
+-------------------+
    |
    v
+-------------------+
| Stage 2: Detect   | --> YOLO Detection, Crop Charts
+-------------------+
    |
    v
+-------------------+
| Stage 3: Extract  | --> OCR, Element Detection, Geometric Analysis
+-------------------+
    |
    v
+-------------------+
| Stage 4: Reason   | --> AI Router (SLM/Gemini/OpenAI), Value Mapping
+-------------------+
    |
    v
+-------------------+
| Stage 5: Report   | --> Insights, Validation, JSON + Text Output
+-------------------+
    |
    v
Output (JSON + Report)
```

**AI Routing Layer** (between Stage 4 and AI providers):

```
AIRouter.resolve(task_type)
    |
    +---> LocalSLMAdapter  (Qwen-2.5 + LoRA, offline, PRIMARY)
    +---> GeminiAdapter    (gemini-2.0-flash, cloud FALLBACK 1)
    +---> OpenAIAdapter    (gpt-4o-mini, cloud FALLBACK 2)
```

## Project Structure

```
chart_analysis_ai_v3/
|
+-- .github/instructions/       # AI Agent guidelines (13 files, 3-tier hierarchy)
+-- config/                     # YAML configuration (base, models, pipeline, training)
+-- data/
|   +-- academic_dataset/       # 32,364 Arxiv chart images + metadata
|   +-- slm_training_v3/       # 268,799 SLM training samples
|   +-- cache/                  # OCR cache (46,910 entries)
|   +-- samples/                # Demo/test samples
+-- docs/
|   +-- MASTER_CONTEXT.md       # Project overview and status tracker
|   +-- architecture/           # System design (5 docs)
|   +-- thesis_capstone/        # LaTeX thesis (39 pages, 25 visual assets)
|   +-- guides/                 # How-to guides (5 docs)
|   +-- reports/                # Technical reports
+-- models/
|   +-- weights/                # YOLO, ResNet-18 weights
|   +-- onnx/                   # ONNX exports (ResNet-18: 42.64 MB)
|   +-- slm/                    # SLM LoRA adapters
+-- notebooks/                  # 12 Jupyter notebooks (00-04 + 01a-01f)
+-- src/core_engine/            # Core AI engine (47 Python files)
|   +-- pipeline.py             # Main orchestrator (all 5 stages wired)
|   +-- schemas/                # Pydantic models (6 files)
|   +-- stages/                 # Pipeline stages (s1-s5)
|   +-- ai/                     # AI Router + Adapters (8 files)
|   +-- validators/             # Output validation
|   +-- data_factory/           # QA data generation
+-- tests/                      # 232 tests (21 files)
|   +-- test_s3_extraction/     # Stage 3 tests (8 files)
|   +-- test_s4_reasoning/      # Stage 4 tests (3 files)
|   +-- test_ai/                # AI routing tests (5 files)
+-- scripts/                    # Utility scripts (18 files)
```

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `00_full_pipeline_test.ipynb` | End-to-end pipeline test |
| `00_quick_start.ipynb` | Quick introduction |
| `01_stage1_ingestion.ipynb` | PDF/Image loading and normalization |
| `01a_data_collection.ipynb` | Academic dataset collection |
| `01b_image_extraction.ipynb` | Image extraction from PDFs |
| `01c_chart_detection.ipynb` | YOLO chart detection |
| `01d_chart_classification.ipynb` | Gemini chart classification |
| `01e_qa_generation.ipynb` | QA pair generation |
| `01f_review_uncertain.ipynb` | Review uncertain classifications |
| `02_stage2_detection.ipynb` | YOLO detection stage |
| `03_stage3_extraction.ipynb` | OCR, vectorization, geometric analysis |
| `04_stage4_reasoning.ipynb` | AI reasoning with Gemini/SLM |

## Documentation

| Document | Description |
| --- | --- |
| [MASTER_CONTEXT.md](docs/MASTER_CONTEXT.md) | Project overview and status |
| [Quick Start Guide](docs/guides/QUICK_START.md) | Getting started |
| [Development Guide](docs/guides/DEVELOPMENT.md) | Dev environment setup |
| [SLM Training Guide](docs/guides/SLM_TRAINING_GUIDE.md) | Fine-tuning SLM models |
| [Pipeline Flow](docs/architecture/PIPELINE_FLOW.md) | Data flow diagrams |
| [Stage 3 Details](docs/architecture/STAGE3_EXTRACTION.md) | Extraction module deep dive |
| [Stage 4 Details](docs/architecture/STAGE4_REASONING.md) | AI reasoning module |
| [Stage 5 Details](docs/architecture/STAGE5_REPORTING.md) | Reporting and insights |
| [System Overview](docs/architecture/SYSTEM_OVERVIEW.md) | Full system design |

## Thesis

The academic thesis is located in `docs/thesis_capstone/` and compiles to a 39-page PDF.

| Component | Count |
| --- | --- |
| Content chapters (.tex) | 7 (introduction, literature review, methodology, system design, results, project management, conclusion) |
| PDF figures | 7 |
| LaTeX tables | 12 |
| TikZ diagrams | 6 |
| Bibliography entries | 21 |

Build: `cd docs/thesis_capstone && latexmk -xelatex main.tex`

## Development

```bash
# Run all tests (232 tests)
.venv/Scripts/python.exe -m pytest tests/ -v

# Run specific test suites
.venv/Scripts/python.exe -m pytest tests/test_s3_extraction/ -v
.venv/Scripts/python.exe -m pytest tests/test_ai/ -v

# Linting
ruff check src/

# Type checking
mypy src/core_engine/

# Format code
black src/ tests/
```

## Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Detection | Ultralytics YOLO v8/v11 | Chart localization |
| Classification | ResNet-18 (custom) | Chart type classification |
| OCR | PaddleOCR | Text extraction |
| Image Processing | OpenCV, scikit-image | Preprocessing, skeletonization |
| Geometric | NumPy, SciPy | Pixel-to-value calculation |
| AI Reasoning | Qwen-2.5 / Gemini / OpenAI | Multi-provider via AIRouter |
| Validation | Pydantic v2 | Schema enforcement |
| Config | OmegaConf | Hierarchical ML config |
| Training | PyTorch, PEFT, trl | LoRA fine-tuning |

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Pydantic v2](https://docs.pydantic.dev/)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [trl (SFTTrainer)](https://github.com/huggingface/trl)
