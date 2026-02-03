# Scripts Directory

Utility scripts for training, evaluation, and data processing.

## Available Scripts

### Model Training

| Script | Purpose | Status |
|--------|---------|--------|
| `train_resnet18_v2.py` | Train ResNet-18 chart classifier | Production |
| `train_yolo_chart_detector.py` | YOLO chart detection training | Production |
| `train_slm_lora.py` | Fine-tune Qwen SLM with LoRA | Ready |

### Model Evaluation & Export

| Script | Purpose | Status |
|--------|---------|--------|
| `evaluate_resnet18.py` | Evaluate ResNet-18 on test set | Ready |
| `export_resnet18_onnx.py` | Export to ONNX format | Ready |
| `generate_gradcam.py` | Generate Grad-CAM visualizations | Ready |
| `test_resnet_integration.py` | Integration test for classifier | Ready |

### Data Processing

| Script | Purpose | Status |
|--------|---------|--------|
| `download_arxiv_batch.py` | Download papers from arXiv | Ready |
| `batch_stage3_parallel.py` | Batch Stage 3 extraction (parallel) | Production |
| `prepare_slm_training_data.py` | Merge QA + Stage3 for SLM training | Ready |
| `extract_backgrounds.py` | Extract text-only pages for YOLO training | Ready |
| `generate_synthetic_dataset.py` | Generate synthetic YOLO training data | Ready |
| `verify_qa_dataset.py` | Verify QA dataset integrity | Ready |

### Testing & Demo

| Script | Purpose | Status |
|--------|---------|--------|
| `demo_full_pipeline.py` | Demo full pipeline end-to-end | Ready |
| `test_element_detector.py` | Test element detection module | Ready |
| `test_stage4.py` | Test Stage 4 reasoning | Ready |
| `test_qwen_slm.py` | Test Qwen SLM integration | Ready |

## Usage Examples

### Batch Stage 3 Extraction

```bash
# Run parallel extraction on all classified charts
.venv/Scripts/python.exe scripts/batch_stage3_parallel.py --workers 8

# Limit to specific chart type
.venv/Scripts/python.exe scripts/batch_stage3_parallel.py --workers 8 --chart-type bar
```

### Train ResNet-18 Classifier

```bash
.venv/Scripts/python.exe scripts/train_resnet18_v2.py \
    --epochs 50 \
    --batch-size 128 \
    --preprocess
```

### Prepare SLM Training Data

```bash
.venv/Scripts/python.exe scripts/prepare_slm_training_data.py

# With curriculum stage
.venv/Scripts/python.exe scripts/prepare_slm_training_data.py --curriculum stage2
```

### Train YOLO Chart Detector

```bash
.venv/Scripts/python.exe scripts/train_yolo_chart_detector.py \
    --epochs 100 \
    --batch 16
```

### Export to ONNX

```bash
.venv/Scripts/python.exe scripts/export_resnet18_onnx.py \
    --model models/weights/resnet18_chart_classifier_v2_best.pt \
    --output models/onnx/resnet18_chart_classifier_v2.onnx
```
