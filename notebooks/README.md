# Notebooks Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-25 | That Le | Interactive exploration notebooks |

## Overview

This directory contains Jupyter notebooks for exploring and testing each pipeline stage.

## Notebooks

| Notebook | Purpose | Stage |
| --- | --- | --- |
| `00_quick_start.ipynb` | Quick introduction to the pipeline | All |
| `01_stage1_ingestion.ipynb` | PDF/Image loading, quality validation | Stage 1 |
| `02_stage2_detection.ipynb` | YOLO chart detection, cropping | Stage 2 |
| `03_stage3_extraction.ipynb` | OCR, vectorization, geometric analysis | Stage 3 |
| `04_data_exploration.ipynb` | Explore academic dataset | Data |

## Quick Start

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## Recommended Order

1. **Start with** `00_quick_start.ipynb` - Get a quick overview
2. **Then explore stages**:
   - `01_stage1_ingestion.ipynb` - Understand input processing
   - `02_stage2_detection.ipynb` - See YOLO in action
   - `03_stage3_extraction.ipynb` - Deep dive into extraction
3. **Finally** `04_data_exploration.ipynb` - Explore the dataset

## Prerequisites

Make sure you have installed the project dependencies:

```bash
pip install -e ".[dev]"
```

Required packages for notebooks:
- `jupyter` or `jupyterlab`
- `matplotlib`
- `opencv-python`
- `ultralytics` (for Stage 2)
- `paddleocr` (for Stage 3)

## Stage Summary

### Stage 1: Ingestion

- Load PDF/DOCX/Images
- Convert PDF pages to images (PyMuPDF)
- Quality validation (blur, resolution, contrast)
- Image normalization (CLAHE enhancement)

### Stage 2: Detection

- Load YOLO model
- Detect chart bounding boxes
- Apply confidence threshold
- Crop detected regions

### Stage 3: Extraction

- Negative image transformation
- Skeletonization (Lee algorithm)
- RDP vectorization
- OCR with role classification
- Geometric axis calibration
- Element detection (bars, lines, points)
- Chart type classification

### Stage 4: Reasoning (Planned)

- SLM integration
- OCR error correction
- Value mapping
- Description generation

### Stage 5: Reporting (Planned)

- Schema validation
- Insight generation
- Output formatting (JSON, Markdown)

## Data Paths

Notebooks expect data in these locations:

```
data/
├── academic_dataset/
│   └── images/          # Chart images from Arxiv
├── samples/             # Demo/test samples
├── raw/                 # Raw input files
└── raw_pdfs/            # PDF documents
```

## Tips

1. **Run cells sequentially** - Notebooks build on previous cells
2. **Check model paths** - Update paths if your models are in different locations
3. **GPU acceleration** - Set `device="cuda"` for faster inference
4. **Memory** - Close large visualizations if running low on memory
