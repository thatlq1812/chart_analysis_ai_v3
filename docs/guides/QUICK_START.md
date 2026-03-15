# Quick Start Guide

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-02-04 | That Le | Updated with current project state |

## 1. Prerequisites

### 1.1. System Requirements

| Component | Requirement |
| --- | --- |
| OS | Windows 10+, Linux, macOS |
| Python | 3.11 or higher |
| Memory | 8GB RAM minimum (16GB recommended) |
| GPU | Optional (CUDA for faster inference) |
| Disk | 5GB free space |

### 1.2. Required Tools

- Git
- Python 3.11+
- pip or uv (recommended)

## 2. Installation

### 2.1. Clone Repository

```bash
git clone https://github.com/your-org/chart_analysis_ai_v3.git
cd chart_analysis_ai_v3
```

### 2.2. Create Virtual Environment

**Using uv (recommended):**
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

**Using standard venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 2.3. Install Dependencies

**Development installation:**
```bash
pip install -e ".[dev]"
```

**Production installation:**
```bash
pip install -e .
```

### 2.4. Download Models

```bash
# Download YOLO weights
python scripts/download_models.py

# Or manually place weights in models/weights/
```

### 2.5. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "from core_engine import ChartAnalysisPipeline; print('OK')"
```

## 3. Basic Usage

### 3.1. Python API

```python
from pathlib import Path
from core_engine import ChartAnalysisPipeline
from core_engine.stages.s3_extraction import Stage3Extraction

# Initialize Stage 3 (extraction)
stage3 = Stage3Extraction()

# Process a single chart image
result = stage3.process_single_image(Path("data/samples/chart.png"))

# Access results
print(f"Chart type: {result.chart_type}")
print(f"Detected elements: {len(result.elements)}")
for text in result.texts:
    print(f"  - {text.text} ({text.role})")
```

### 3.2. Command Line (Future)

```bash
# Analyze a single image
chart-analyze image path/to/chart.png

# Analyze a PDF
chart-analyze pdf path/to/document.pdf

# With output format
chart-analyze image chart.png --output json --out-file result.json
```

## 4. Project Structure

```
chart_analysis_ai_v3/
|
+-- config/                 # Configuration files
|   +-- base.yaml           # Base settings
|   +-- models.yaml         # Model paths
|   +-- pipeline.yaml       # Pipeline settings
|
+-- data/
|   +-- samples/            # Example charts
|   +-- raw/                # Input files
|   +-- output/             # Results
|
+-- src/core_engine/        # Main library
|   +-- stages/             # Pipeline stages
|   +-- schemas/            # Data structures
|
+-- docs/                   # Documentation
+-- tests/                  # Test suite
+-- notebooks/              # Jupyter notebooks
```

## 5. Configuration

### 5.1. Pipeline Configuration

Edit `config/pipeline.yaml`:

```yaml
pipeline:
  stages:
    ingestion:
      enabled: true
      max_image_size: 4096
      
    detection:
      enabled: true
      confidence_threshold: 0.5
      
    extraction:
      enabled: true
      backend: "deplot"     # Options: deplot | matcha | pix2struct | svlm
      device: "auto"        # auto | cpu | cuda
```

### 5.2. Model Configuration

Edit `config/models.yaml`:

```yaml
models:
  yolo:
    path: "models/weights/chart_detector.pt"
    device: "auto"
    
  extraction:
    backend: "deplot"               # Default VLM backend
    model_id: "google/deplot"       # HuggingFace model ID
    device: "auto"
```

## 6. Running Examples

### 6.1. Stage 3 Test Script

```bash
# Test Stage 3 VLM extraction
.venv/Scripts/python.exe scripts/pipeline/test_stage3.py
```

### 6.2. Generate Reports

```bash
# Run full pipeline demo
.venv/Scripts/python.exe scripts/pipeline/demo.py
```

### 6.3. Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks/01_data_exploration.ipynb
```

## 7. Troubleshooting

### 7.1. Common Issues

| Issue | Solution |
| --- | --- |
| VLM model not downloaded | First run downloads from HuggingFace automatically |
| CUDA out of memory | Set `device: "cpu"` in config or use `backend: "deplot"` (smaller) |
| Model not found (YOLO) | Run `.venv/Scripts/python.exe scripts/utils/download_models.py` |
| Permission denied | Check file permissions, run as admin |

### 7.2. Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in config:

```yaml
logging:
  level: DEBUG
  file: logs/debug.log
```

## 8. Next Steps

1. Read [MASTER_CONTEXT.md](MASTER_CONTEXT.md) for project overview
2. Explore [architecture/](architecture/) for system design
3. Check [reports/](reports/) for test results
4. Run notebooks for interactive exploration

## 9. Getting Help

- **Documentation**: `docs/` folder
- **Issues**: GitHub Issues
- **Architecture**: `docs/architecture/`

---

**Happy Chart Analyzing!**
