# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Stage 4: Local SLM Integration (Qwen/Llama)
- Stage 5: Reporting (Output Formatting)
- PaddleOCR compatibility fix for Windows

---

## [0.4.0] - 2026-01-26

### Week 2: Stage 4 Reasoning with Gemini API [PARTIAL]

#### Added
- **Stage 4 Reasoning Module** (`src/core_engine/stages/s4_reasoning/`)
  - `s4_reasoning.py`: Main Stage4Reasoning orchestrator
  - `gemini_engine.py`: Google Gemini API integration
  - `base_engine.py`: Abstract base class for reasoning engines
  - Features:
    - OCR error correction (loo→100, O→0, 2O25→2025)
    - Academic-style description generation
    - Value mapping from pixel coordinates
    - Legend-color association
    - Rule-based fallback when API unavailable

- **Gemini API Configuration**
  - Model: `gemini-2.0-flash-exp`
  - Temperature: 0.3 (deterministic)
  - Max tokens: 2048
  - Vision support: Prepared (not yet enabled)

- **Notebook: Stage 4 Demo** (`notebooks/04_stage4_reasoning.ipynb`)
  - 18 cells demonstrating Stage 4 capabilities
  - Tests: Mock data, OCR correction, real chart processing
  - Full pipeline test (Stage 3 → Stage 4)

#### Changed
- **Default OCR Engine**: Changed from `paddleocr` to `easyocr`
  - Reason: PaddleOCR 3.3.x has oneDNN compatibility issues on Windows
  - EasyOCR works reliably on Windows Python 3.12
  - File: `src/core_engine/stages/s3_extraction/s3_extraction.py`

#### Fixed
- **Gemini Engine NoneType Error**: Safe null checks for `corrections` and `color_rgb`
- **OCR Engine Fallback**: Auto-fallback to EasyOCR when PaddleOCR fails

#### Known Issues
- PaddleOCR 3.3.x crashes on Windows with `NotImplementedError: ConvertPirAttribute2RuntimeAttribute`
- Gemini API occasionally returns unparseable JSON (fallback works)

#### Next Steps
- [ ] Implement local SLM (Qwen-2.5 / Llama-3.2)
- [ ] Add vision model support to Gemini engine
- [ ] Improve value mapping with geometric calibration
- [ ] Build Stage 5: Reporting

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
