# Notebooks Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-01-30 | That Le | Complete rewrite with accurate status |
| 1.2.0 | 2026-01-26 | That Le | Added Data Factory notebooks (01a-01f) |
| 1.1.0 | 2026-01-26 | That Le | Added Stage 4, updated for ResNet-18 |
| 1.0.0 | 2026-01-25 | That Le | Interactive exploration notebooks |

---

## Overview

This directory contains Jupyter notebooks for:
1. **Pipeline Stages** - Testing and visualizing each stage of the Geo-SLM pipeline
2. **Data Factory** - Collecting and preparing training data from academic sources

---

## Pipeline Stage Notebooks

| Notebook | Stage | Purpose | Status | Runnable |
| --- | --- | --- | --- | --- |
| [00_quick_start.ipynb](00_quick_start.ipynb) | All | Quick introduction and overview | STABLE | YES |
| [01_stage1_ingestion.ipynb](01_stage1_ingestion.ipynb) | S1 | PDF/Image loading, quality validation | STABLE | YES |
| [02_stage2_detection.ipynb](02_stage2_detection.ipynb) | S2 | YOLO chart detection, cropping | STABLE | YES |
| [03_stage3_extraction.ipynb](03_stage3_extraction.ipynb) | S3 | OCR, element detection, ResNet-18 classification | STABLE | YES |
| [04_stage4_reasoning.ipynb](04_stage4_reasoning.ipynb) | S4 | Value mapping, Gemini reasoning | IN PROGRESS | PARTIAL |

### Stage Details

#### 00_quick_start.ipynb

**Purpose:** Quick overview of the entire pipeline with sample outputs

| Section | Description |
| --- | --- |
| Environment Setup | Check dependencies, import modules |
| Pipeline Demo | Run full pipeline on sample chart |
| Output Visualization | Display extracted data and insights |

**Prerequisites:**
- All dependencies installed (`pip install -e .`)
- Sample images in `data/samples/`

**Cells:** 23 (9 markdown, 14 code)

---

#### 01_stage1_ingestion.ipynb

**Purpose:** Load and normalize input files (PDF, DOCX, images)

| Section | Description |
| --- | --- |
| File Type Detection | Identify PDF, DOCX, PNG, JPG |
| PDF Conversion | Convert PDF pages to images using PyMuPDF |
| Quality Validation | Check resolution, blur, dimensions |
| Normalization | Resize, enhance contrast, generate session ID |

**Key Code:**
```python
from core_engine.stages import Stage1Ingestion
stage1 = Stage1Ingestion(config)
result = stage1.process(input_path)  # -> Stage1Output
```

**Output Schema:** `Stage1Output` with `List[CleanImage]`

**Cells:** 17 (8 markdown, 9 code)

---

#### 02_stage2_detection.ipynb

**Purpose:** Detect chart regions using YOLO and crop them

| Section | Description |
| --- | --- |
| Model Loading | Load YOLO weights (yolo26n.pt) |
| Detection | Run inference on clean images |
| Visualization | Draw bounding boxes, show confidence |
| Cropping | Extract individual chart images |

**Key Code:**
```python
from core_engine.stages import Stage2Detection
stage2 = Stage2Detection(config)
result = stage2.process(stage1_output)  # -> Stage2Output
```

**Output Schema:** `Stage2Output` with `List[DetectedChart]`

**Model:** `yolo26n.pt` (custom trained, 85%+ mAP)

**Cells:** 20 (10 markdown, 10 code)

---

#### 03_stage3_extraction.ipynb

**Purpose:** Extract structural data from chart images

| Section | Description |
| --- | --- |
| Classification | ResNet-18 identifies chart type (8 classes) |
| OCR Extraction | EasyOCR extracts text with spatial info |
| Element Detection | Detect bars, points, slices, lines |
| Geometric Analysis | Calibrate axes, compute pixel-to-value mapping |

**Key Code:**
```python
from core_engine.stages import Stage3Extraction
stage3 = Stage3Extraction(config)
result = stage3.process(stage2_output)  # -> Stage3Output
```

**Output Schema:** `Stage3Output` with `List[RawMetadata]`

**Cells:** 18 (9 markdown, 9 code)

**Performance (tested on 800+ images):**

| Metric | Value |
| --- | --- |
| Classification Accuracy | 100% (8 types) |
| OCR Confidence | 91.5% |
| Overall Confidence | 92.6% |

---

#### 04_stage4_reasoning.ipynb

**Purpose:** Semantic reasoning and value extraction using LLM

| Section | Description |
| --- | --- |
| Value Mapping | GeometricValueMapper converts pixels to values |
| Prompt Building | GeminiPromptBuilder creates structured prompts |
| LLM Reasoning | Gemini API (prototype) / Local SLM (future) |
| Post-processing | Merge results, confidence scoring |

**Key Code:**
```python
from core_engine.stages import Stage4Reasoning
stage4 = Stage4Reasoning(config)
result = stage4.process(stage3_output)  # -> Stage4Output
```

**Output Schema:** `Stage4Output` with `List[RefinedChartData]`

**Cells:** 38 (17 markdown, 21 code)

**Status:** Core components implemented, Gemini integration testing in progress

---

## Data Factory Notebooks

| Notebook | Step | Input | Output | Status |
| --- | --- | --- | --- | --- |
| [01a_data_collection.ipynb](01a_data_collection.ipynb) | 1 | arXiv API | PDFs | STABLE |
| [01b_image_extraction.ipynb](01b_image_extraction.ipynb) | 2 | PDFs | Images | STABLE |
| [01c_chart_detection.ipynb](01c_chart_detection.ipynb) | 3 | Images | Detected charts | STABLE |
| [01d_chart_classification.ipynb](01d_chart_classification.ipynb) | 4 | Charts | Classified by type | STABLE |
| [01e_qa_generation.ipynb](01e_qa_generation.ipynb) | 5 | Charts | QA pairs | STABLE |
| [01f_review_uncertain.ipynb](01f_review_uncertain.ipynb) | 6 | Uncertain | Manual review | STABLE |

### Data Factory Pipeline

```
Step 1              Step 2              Step 3              Step 4              Step 5
01a_data_          01b_image_         01c_chart_         01d_chart_         01e_qa_
collection         extraction         detection          classification     generation
    |                  |                  |                   |                  |
    v                  v                  v                   v                  v
arXiv PDFs  --->  Raw Images  --->  Detected   --->  Classified  --->  QA Dataset
                                    Charts          Charts
```

### Individual Notebook Details

#### 01a_data_collection.ipynb

**Purpose:** Download academic PDFs from arXiv API

| Feature | Description |
| --- | --- |
| Search Categories | cs.CV, cs.LG, stat.ML (chart-rich papers) |
| Rate Limiting | Respects arXiv API limits |
| Progress Tracking | Resume from last position |
| Deduplication | Skip already downloaded papers |

**Cells:** 17 (7 markdown, 10 code)

**Output:** `data/raw_pdfs/*.pdf`

---

#### 01b_image_extraction.ipynb

**Purpose:** Extract images from PDF documents

| Feature | Description |
| --- | --- |
| Extraction Method | PyMuPDF (fitz) for embedded images |
| Quality Filter | Skip small/low-quality images |
| Format Handling | Convert CMYK to RGB if needed |
| Batch Processing | Process multiple PDFs in parallel |

**Cells:** 19 (9 markdown, 10 code)

**Output:** `data/academic_dataset/images/`

---

#### 01c_chart_detection.ipynb

**Purpose:** Detect chart regions in extracted images

| Feature | Description |
| --- | --- |
| Model | YOLO (yolo26n.pt) |
| Multi-chart | Handle images with multiple charts |
| Cropping | Save individual chart images |
| Metadata | Store bounding box info |

**Cells:** 10 (2 markdown, 8 code)

**Output:** `data/academic_dataset/detected_charts/`

---

#### 01d_chart_classification.ipynb

**Purpose:** Classify detected charts by type

| Feature | Description |
| --- | --- |
| Model | ResNet-18 (94.66% accuracy) |
| Classes | 8 types: area, bar, box, heatmap, histogram, line, pie, scatter |
| Confidence | Track low-confidence for review |
| Organization | Sort into type-specific folders |

**Cells:** 11 (2 markdown, 9 code)

**Output:** `data/academic_dataset/classified_charts/{type}/`

---

#### 01e_qa_generation.ipynb

**Purpose:** Generate question-answer pairs using Gemini API

| Feature | Description |
| --- | --- |
| Question Types | Value extraction, comparison, trend analysis |
| Validation | JSON schema validation |
| Batching | Process in batches with rate limiting |
| Error Handling | Retry failed generations |

**Cells:** 13 (2 markdown, 11 code)

**Output:** `data/academic_dataset/chart_qa_v2/`

---

#### 01f_review_uncertain.ipynb

**Purpose:** Manual review of uncertain classifications

| Feature | Description |
| --- | --- |
| Display | Show image with predicted class |
| Action | Confirm, reclassify, or reject |
| Tracking | Update metadata after review |

**Cells:** 8 (2 markdown, 6 code)

---

## Experimental/Archive Notebooks

| Notebook | Description | Status |
| --- | --- | --- |
| [01e_qa_generation_new.ipynb](01e_qa_generation_new.ipynb) | Alternative QA generation approach | EXPERIMENTAL |
| [03x_qa_generation_v2.ipynb](03x_qa_generation_v2.ipynb) | V2 QA format with Canonical Format | EXPERIMENTAL |

**Note:** These are experimental notebooks used during development. Use the main notebooks for production workflows.

---

## Project Status Summary

### Pipeline Stages

| Stage | Module | Files | Tests | Accuracy | Status |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | s1_ingestion.py | 1 | - | N/A | COMPLETE |
| Stage 2 | s2_detection.py | 1 | - | 85%+ mAP | COMPLETE |
| Stage 3 | s3_extraction/ | 12 | 129 | 100% class / 91.5% OCR | COMPLETE |
| Stage 4 | s4_reasoning/ | 6 | 36 | - | IN PROGRESS |
| Stage 5 | s5_reporting/ | 0 | 0 | - | PLANNED |

**Total Tests:** 177 passing

### Data Factory Status

| Metric | Current | Target |
| --- | --- | --- |
| PDFs Collected | (cleared) | 10,000 |
| Images Extracted | (cleared) | ~30,000 |
| Charts Detected | (cleared) | ~10,000 |
| QA Pairs | (cleared) | 50,000+ |

**Note:** Data directories have been cleared. Run Data Factory notebooks to repopulate.

---

## Quick Start

### For Pipeline Testing

```bash
# 1. Start Jupyter
jupyter lab

# 2. Open notebooks in order:
# - 00_quick_start.ipynb (overview)
# - 01_stage1_ingestion.ipynb
# - 02_stage2_detection.ipynb
# - 03_stage3_extraction.ipynb
# - 04_stage4_reasoning.ipynb
```

### For Data Collection

```bash
# Run in order:
# 1. 01a_data_collection.ipynb
# 2. 01b_image_extraction.ipynb
# 3. 01c_chart_detection.ipynb
# 4. 01d_chart_classification.ipynb
# 5. 01e_qa_generation.ipynb
```

---

## Prerequisites

### Required Packages

```bash
# Core
pip install -e .  # Install project package

# Notebooks
pip install jupyterlab ipywidgets matplotlib

# Stage-specific
pip install torch torchvision  # ResNet-18
pip install ultralytics        # YOLO
pip install easyocr            # OCR
pip install google-generativeai  # Gemini API (Stage 4)
```

### Required Models

| Model | Path | Size | Purpose |
| --- | --- | --- | --- |
| YOLO | `yolo26n.pt` | ~6MB | Chart detection |
| ResNet-18 | `models/weights/resnet18_classifier.pth` | ~43MB | Chart classification |
| ONNX | `models/onnx/resnet18_classifier.onnx` | ~43MB | Fast inference |

### Environment Variables

```bash
# Required for Stage 4 (Gemini API)
GOOGLE_API_KEY=your_api_key_here

# Or create config/secrets/gemini.yaml
api_key: "your_api_key_here"
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
| --- | --- |
| Module not found | Run `pip install -e .` from project root |
| YOLO model not found | Check `yolo26n.pt` exists in project root |
| ResNet model not found | Run `scripts/train_resnet18_classifier.py` |
| Gemini API error | Check `GOOGLE_API_KEY` is set |
| CUDA out of memory | Set `device: cpu` in config |

### Checking Environment

```python
# Run in notebook to verify setup
import sys
print(f"Python: {sys.version}")

from core_engine import __version__
print(f"Core Engine: {__version__}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## Contributing

When creating new notebooks:

1. Follow naming convention: `{NN}_{stage_or_purpose}.ipynb`
2. Include markdown header with purpose and prerequisites
3. Add entry to this README
4. Clear outputs before committing (optional)

---

## References

- [MASTER_CONTEXT.md](../docs/MASTER_CONTEXT.md) - Project overview
- [PIPELINE_FLOW.md](../docs/architecture/PIPELINE_FLOW.md) - Pipeline architecture
- [research.instructions.md](../.github/instructions/research.instructions.md) - Research guidelines
