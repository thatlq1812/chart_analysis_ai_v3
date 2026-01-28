# Notebooks Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-25 | That Le | Interactive exploration notebooks |
| 1.1.0 | 2026-01-26 | That Le | Added Stage 4, updated for ResNet-18 |
| 1.2.0 | 2026-01-26 | That Le | Added Data Factory notebooks (01a-01d) |

## Overview

This directory contains Jupyter notebooks for exploring and testing each pipeline stage.

## Notebooks

### Pipeline Stages

| Notebook | Purpose | Stage | Status |
| --- | --- | --- | --- |
| `00_quick_start.ipynb` | Quick introduction to the pipeline | All | STABLE |
| `01_stage1_ingestion.ipynb` | PDF/Image loading, quality validation | Stage 1 | STABLE |
| `02_stage2_detection.ipynb` | YOLO chart detection, cropping | Stage 2 | STABLE |
| `03_stage3_extraction.ipynb` | OCR, vectorization, ResNet-18 classification | Stage 3 | STABLE |
| `04_stage4_reasoning.ipynb` | SLM reasoning with Gemini API | Stage 4 | STABLE |

### Data Factory (NEW)

| Notebook | Purpose | Input | Output |
| --- | --- | --- | --- |
| `01a_data_collection.ipynb` | Collect PDFs from arXiv | arXiv API | `data/raw_pdfs/` |
| `01b_image_extraction.ipynb` | Extract images from PDFs | PDFs | `data/academic_dataset/images/` |
| `01c_chart_classification.ipynb` | Classify charts with ResNet-18 | Images | `chart_qa/classified/` |
| `01d_qa_generation.ipynb` | Generate QA pairs with Gemini | Charts | `chart_qa/dataset.json` |

### Data Exploration

| Notebook | Purpose | Status |
| --- | --- | --- |
| `04_academic_dataset_test.ipynb` | Academic dataset evaluation | STABLE |
| `04_data_exploration.ipynb` | Explore academic dataset | STABLE |

## Pipeline Status (v3.0)

| Stage | Component | Accuracy | Status |
| --- | --- | --- | --- |
| Stage 1 | Ingestion | N/A | COMPLETE |
| Stage 2 | YOLO Detection | 85%+ mAP | COMPLETE |
| Stage 3 | ResNet-18 Classifier | 94.66% | COMPLETE |
| Stage 3 | Element Detection | K-Means enhanced | COMPLETE |
| Stage 4 | Gemini Reasoning | Experimental | IN PROGRESS |
| Stage 5 | Reporting | N/A | PLANNED |

## Data Factory Pipeline

The Data Factory notebooks create training data:

```
01a_data_collection     01b_image_extraction     01c_chart_classification     01d_qa_generation
        |                        |                          |                         |
        v                        v                          v                         v
   arXiv PDFs  ------>  Extracted Images  ------>  Filtered Charts  ------>  QA Dataset
   (10,000+)               (~30,000)                 (~10,000)              (50,000+ pairs)
```

### Current Progress

| Metric | Count |
| --- | --- |
| PDFs collected | 889 |
| Images extracted | 2,852 |
| QA pairs generated | 13,297 |
| Target PDFs | 10,000 |

## Quick Start

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## Recommended Order

### For Pipeline Testing
1. **Start with** `00_quick_start.ipynb` - Get a quick overview
2. **Then explore stages**:
   - `01_stage1_ingestion.ipynb` - Understand input processing
   - `02_stage2_detection.ipynb` - See YOLO in action
   - `03_stage3_extraction.ipynb` - Deep dive into extraction with ResNet-18
   - `04_stage4_reasoning.ipynb` - SLM reasoning with Gemini API

### For Data Collection
1. `01a_data_collection.ipynb` - Download PDFs from arXiv
2. `01b_image_extraction.ipynb` - Extract images from PDFs
3. `01c_chart_classification.ipynb` - Filter charts using ResNet-18
4. `01d_qa_generation.ipynb` - Generate QA pairs using Gemini API

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
- `torch`, `torchvision` (for ResNet-18)
- `google-generativeai` (for Stage 4)

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
- **ResNet-18 Classification** (94.66% accuracy) - NEW
- Skeletonization (Lee algorithm)
- RDP vectorization
- OCR with role classification (PaddleOCR)
- **K-Means element detection** - IMPROVED
- Geometric axis calibration

### Stage 4: Reasoning

- **Gemini API integration** - NEW
- OCR error correction
- Value mapping
- Legend-color association
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
