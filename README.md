# Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 3.3.0 | 2026-01-31 | That Le | Dataset v2 (32,445 images), ResNet-18 v2 (94.14% accuracy) |
| 3.2.0 | 2026-01-26 | That Le | ResNet-18 classifier (94.66% accuracy) production-ready |
| 3.1.0 | 2026-01-25 | That Le | Documentation restructured, Stage notebooks added |
| 3.0.0 | 2026-01-19 | That Le | Chart Analysis AI V3 |

## Overview

A **hybrid AI system** for extracting structured data from chart images, combining:

| Component | Role | Technology |
| --- | --- | --- |
| **Computer Vision** | Chart detection & localization | YOLOv8/v11 |
| **Geometric Analysis** | Precise value extraction | OpenCV + NumPy |
| **Small Language Model** | OCR correction & reasoning | Qwen-2.5 / Llama-3.2 |

> **Core Philosophy**: Achieve higher accuracy than pure multimodal LLMs through hybrid neuro-symbolic approach, while maintaining local inference capability.

## Current Status

| Phase | Status | Progress |
| --- | --- | --- |
| Phase 1: Foundation | [COMPLETED] | Dataset: 32,445 charts, 32,445 QA pairs |
| Phase 2: Core Engine | [IN PROGRESS] | Stage 3 done (ResNet-18 v2: 94.14%), Stage 4-5 pending |
| Phase 3: Optimization | [PLANNED] | SLM fine-tuning on QA dataset |
| Phase 4: Presentation | [PLANNED] | - |

**ResNet-18 v2 Classifier:**
- Test Accuracy: **94.14%**
- Classes: 8 (area, bar, box, heatmap, histogram, line, pie, scatter)
- Training Data: 32,445 preprocessed images (256x256 grayscale)

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
from core_engine.stages.s3_extraction import Stage3Extraction

# Initialize Stage 3
stage3 = Stage3Extraction()

# Process a chart image
result = stage3.process_single_image(Path("data/samples/chart.png"))

# Access results
print(f"Chart type: {result.chart_type}")
print(f"Elements: {len(result.elements)}")
for text in result.texts:
    print(f"  - {text.text} ({text.role})")
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
| Stage 3: Extract  | --> OCR, Geometric Analysis, Vectorization
+-------------------+
    |
    v
+-------------------+
| Stage 4: Reason   | --> SLM Correction, Value Mapping [PLANNED]
+-------------------+
    |
    v
+-------------------+
| Stage 5: Report   | --> Validate, Insights, Output [PLANNED]
+-------------------+
    |
    v
Output (JSON + Report)
```

## Project Structure

```
chart_analysis_ai_v3/
|
+-- .github/instructions/   # AI Agent guidelines
+-- config/                 # Configuration files
+-- data/
|   +-- academic_dataset/   # Arxiv chart images
|   +-- samples/            # Demo samples
+-- docs/                   # Documentation
|   +-- MASTER_CONTEXT.md   # Project overview
|   +-- architecture/       # System design
|   +-- guides/             # How-to guides
+-- models/weights/         # Model files
+-- notebooks/              # Interactive exploration
|   +-- 00_quick_start.ipynb
|   +-- 01_stage1_ingestion.ipynb
|   +-- 02_stage2_detection.ipynb
|   +-- 03_stage3_extraction.ipynb
|   +-- 04_data_exploration.ipynb
+-- src/core_engine/        # Core AI engine
+-- tests/                  # Test suites
+-- scripts/                # Utility scripts
```

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `00_quick_start.ipynb` | Quick introduction to the pipeline |
| `01_stage1_ingestion.ipynb` | Test PDF/Image loading & normalization |
| `02_stage2_detection.ipynb` | Test YOLO chart detection |
| `03_stage3_extraction.ipynb` | Test OCR, vectorization, geometric analysis |
| `04_data_exploration.ipynb` | Explore academic dataset |

## Documentation

| Document | Description |
| --- | --- |
| [MASTER_CONTEXT.md](docs/MASTER_CONTEXT.md) | Project overview & status |
| [Quick Start Guide](docs/guides/QUICK_START.md) | Getting started |
| [Development Guide](docs/guides/DEVELOPMENT.md) | Dev environment setup |
| [Pipeline Flow](docs/architecture/PIPELINE_FLOW.md) | Data flow diagrams |
| [Stage 3 Details](docs/architecture/STAGE3_EXTRACTION.md) | Extraction module |

## Development

```bash
# Run tests
pytest tests/ -v

# Run specific stage tests
pytest tests/test_s3_extraction/ -v

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
| Detection | Ultralytics YOLO | Chart localization |
| OCR | PaddleOCR | Text extraction |
| Image Processing | OpenCV, scikit-image | Preprocessing, skeletonization |
| Geometric | NumPy, SciPy | Value calculation |
| Validation | Pydantic v2 | Schema enforcement |
| Config | OmegaConf | Configuration management |

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Pydantic v2](https://docs.pydantic.dev/)
