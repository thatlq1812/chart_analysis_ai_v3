# Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 3.0.0 | 2026-01-19 | That Le | Chart Analysis AI V3 |

## Overview

A hybrid AI system for extracting structured data from chart images, combining:

- **YOLO** for chart detection
- **Geometric Analysis** for precise value extraction  
- **Small Language Model (SLM)** for OCR correction and semantic reasoning

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/chart_analysis_ai_v3.git
cd chart_analysis_ai_v3

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Download model weights
python scripts/download_models.py
```

### Basic Usage

```python
from core_engine import ChartAnalysisPipeline

# Initialize pipeline
pipeline = ChartAnalysisPipeline.from_config()

# Analyze a chart image
result = pipeline.run("path/to/chart.png")

# Get structured data
print(result.charts[0].data.series)
```

### CLI Usage

```bash
# Analyze single file
python -m interface.cli analyze chart.png

# Analyze PDF document
python -m interface.cli analyze report.pdf --output results/

# Run with custom config
python -m interface.cli analyze chart.png --config config/custom.yaml
```

## Project Structure

```
chart_analysis_ai_v3/
|-- .github/instructions/   # AI Agent guidelines
|-- config/                 # Configuration files
|-- data/                   # Data directories
|-- docs/                   # Documentation
|-- models/                 # Model weights
|-- notebooks/              # Jupyter notebooks
|-- research/               # Experiments
|-- src/core_engine/        # Core AI engine
|-- interface/              # CLI, API, Demo
|-- tests/                  # Test suites
```

## Pipeline Stages

| Stage | Purpose | Technology |
| --- | --- | --- |
| 1. Ingestion | Load and normalize inputs | PyMuPDF, Pillow |
| 2. Detection | Detect chart regions | YOLOv8 |
| 3. Extraction | OCR + element detection | PaddleOCR, OpenCV |
| 4. Reasoning | Value mapping + correction | Local SLM |
| 5. Reporting | Generate output | Pydantic, Jinja2 |

## Documentation

- [Master Context](docs/MASTER_CONTEXT.md) - Project overview
- [Architecture](docs/architecture/SYSTEM_OVERVIEW.md) - System design
- [Pipeline Flow](docs/architecture/PIPELINE_FLOW.md) - Data flow diagrams
- [Quick Start Guide](docs/guides/QUICK_START.md) - Getting started

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/
mypy src/

# Format code
black src/ tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Pydantic v2 Docs](https://docs.pydantic.dev/)
