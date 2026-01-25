# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Stage 3: Complete OCR + Geometric Analysis
- Stage 4: Reasoning (SLM Integration)
- Stage 5: Reporting (Output Formatting)

---

## [0.3.0] - 2026-01-25

### Week 1: ResNet-18 Classifier [COMPLETED]

#### Added
- **ResNet-18 Chart Classifier**
  - Model: ResNet-18 with 8-class output (area, bar, box, heatmap, histogram, line, pie, scatter)
  - Accuracy: 94.66% test accuracy (vs 37.5% baseline SimpleChartClassifier)
  - Training: 27 minutes on NVIDIA GPU, early stopping at epoch 15/100
  - Dataset: Academic dataset with stratified train/test split
  
- **Grad-CAM Explainability** (`scripts/generate_gradcam.py`)
  - Visual explanation of model attention regions
  - Target layer: ResNet-18 layer4[-1].conv2 (last convolutional layer)
  - Generated: 8 per-class visualizations + 1 summary (9 files total)
  - Output: `models/explainability/gradcam_*.png`
  
- **ONNX Model Export** (`scripts/export_resnet18_onnx.py`)
  - Cross-platform deployment format (42.64 MB)
  - Inference speed: 6.90ms mean (CPU), 144.9 images/sec throughput
  - Validation: PyTorch vs ONNX predictions match (max diff 0.000982)
  - Output: `models/onnx/resnet18_chart_classifier.onnx` + metadata JSON
  
- **Pipeline Integration** (`src/core_engine/stages/s3_extraction/resnet_classifier.py`)
  - Production wrapper: `ResNet18Classifier` class
  - API methods: `predict()`, `predict_with_confidence()`, `predict_batch()`, `get_class_probabilities()`
  - Device support: Auto-detection (CUDA > MPS > CPU)
  - Configuration: `config/models.yaml` updated with ResNet-18 settings

#### Changed
- **Stage 3 Classifier**: Replaced SimpleChartClassifier (37.5%) with ResNet-18 (94.66%)
- **Configuration**: Updated `config/models.yaml` with ResNet-18 paths and 8 chart classes

#### Fixed
- ONNX export device mismatch: Model on GPU, input on CPU
- Model loading structure: Checkpoint uses ResNetWrapper with "resnet." prefix
- Integration test validation: 93.75% accuracy (15/16 correct)

#### Tested
- Integration test: `scripts/test_resnet_integration.py`
  - Overall: 93.75% accuracy (15/16 samples)
  - Per-class: 7/8 types at 100%, area at 50% (1 misclassification)
  - Pass threshold: 90% (PASSED)
  - Output: Visualization grid + JSON results

#### Documentation
- `docs/reports/WEEK1_COMPLETION_SUMMARY.md`: Comprehensive completion report
- Evaluation results: Confusion matrix, per-class metrics, training curves
- Explainability: Grad-CAM visualizations showing model attention

---

## [0.2.0] - 2026-01-24

### Phase 1: Foundation [COMPLETED]

#### Added
- **Chart QA Dataset**: 2,852 classified charts with 13,297 QA pairs
  - Source: Arxiv academic papers (800+ PDFs)
  - Classification via Google Gemini API
  - 5 QA pairs per chart (structural, counting, comparison, reasoning, extraction)
  
- **Data Factory Tools** (`tools/data_factory/`)
  - PDF Miner: Extract images from PDF documents
  - Gemini Classifier: Chart detection and type classification
  - QA Generator: Automated QA pair generation
  
- **Core Engine Stages**
  - Stage 1: Ingestion (`src/core_engine/stages/s1_ingestion.py`)
  - Stage 2: Detection (`src/core_engine/stages/s2_detection.py`)
  
- **Documentation**
  - MASTER_CONTEXT.md: Project overview
  - PIPELINE_FLOW.md: 5-stage pipeline architecture
  - SYSTEM_OVERVIEW.md: System design
  - CHART_QA_GUIDE.md: Chart QA generation guide
  - ARXIV_DOWNLOAD_GUIDE.md: PDF download instructions

#### Data Statistics
| Metric | Value |
| --- | --- |
| Total Images Processed | 2,852 |
| Total QA Pairs Generated | 13,297 |
| Source PDFs | 800+ |
| Chart Types | bar, line, pie, scatter, area, other |

---

## [0.1.0] - 2026-01-19

### Project Initialization

#### Added
- Initial project structure (V3)
- Configuration files (`config/base.yaml`, `config/models.yaml`, `config/pipeline.yaml`)
- GitHub instructions for AI agents
- Basic schema definitions (`src/core_engine/schemas/`)
- Test fixtures and configuration

#### Structure
```
chart_analysis_ai_v3/
├── .github/instructions/    # AI agent guidelines
├── config/                  # YAML configurations
├── data/                    # Data directories
├── docs/                    # Documentation
├── src/core_engine/         # Main engine code
├── tests/                   # Test suite
└── tools/                   # Utility tools
```

---

## Versioning

- **Major version (X.0.0)**: Breaking changes, architecture redesign
- **Minor version (0.X.0)**: New features, phase completion
- **Patch version (0.0.X)**: Bug fixes, minor improvements
