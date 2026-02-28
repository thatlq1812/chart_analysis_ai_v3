# Scripts Directory

Utility scripts for training, evaluation, and data processing.

Last updated: 2026-03-01 (v3.0.0 — post data-pipeline housekeeping)

## Active Scripts

### Model Training

| Script | Purpose | Status |
|--------|---------|--------|
| `train_resnet18_v2.py` | Train ResNet-18 chart classifier | Production |
| `train_yolo_chart_detector.py` | YOLO chart detection training | Production |
| `train_slm_lora.py` | Fine-tune Qwen2.5-1.5B with QLoRA | Ready (next step) |

### Model Evaluation and Export

| Script | Purpose | Status |
|--------|---------|--------|
| `evaluate_resnet18.py` | Evaluate ResNet-18 on test set | Ready |
| `export_resnet18_onnx.py` | Export ResNet-18 to ONNX format | Ready |
| `test_resnet_integration.py` | Integration test for ResNet classifier | Ready |

### Data Processing

| Script | Purpose | Status |
|--------|---------|--------|
| `batch_stage3_parallel.py` | Batch Stage 3 extraction (parallel, 8 workers) | Production |
| `prepare_slm_training_v3.py` | Build SLM training dataset v3 (268,799 samples) | PRIMARY |
| `_full_audit.py` | Quality audit of Stage3 features (all 8 types) | Ready |
| `download_models.py` | Download model weights | Ready |

### Testing and Demo

| Script | Purpose | Status |
|--------|---------|--------|
| `demo_full_pipeline.py` | Demo full pipeline end-to-end | Ready |
| `test_element_detector.py` | Test element detection module | Ready |
| `test_full_pipeline.py` | Full pipeline integration test | Ready |
| `test_stage4.py` | Test Stage 4 reasoning | Ready |
| `test_qwen_slm.py` | Test Qwen SLM integration | Ready |

## Usage Examples

### Build SLM Training Dataset v3

```bash
# Dry-run (no disk write)
.venv/Scripts/python.exe scripts/prepare_slm_training_v3.py --dry-run

# Full build to disk
.venv/Scripts/python.exe scripts/prepare_slm_training_v3.py --output-dir data/slm_training_v3
```

### Train SLM (Qwen2.5-1.5B + QLoRA)

```bash
.venv/Scripts/python.exe scripts/train_slm_lora.py \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/qwen2.5-1.5b-chart-lora-v3 \
    --epochs 3 --batch-size 4 --lora-rank 16
```

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

### Run Full Quality Audit

```bash
.venv/Scripts/python.exe scripts/_full_audit.py
```

### Train YOLO Chart Detector

```bash
.venv/Scripts/python.exe scripts/train_yolo_chart_detector.py \
    --epochs 100 \
    --batch 16
```

## Archive

One-time and superseded scripts are in `scripts/_archive/`. Do not run these in production.

| Script | Reason Archived |
|--------|----------------|
| `prepare_slm_training_data.py` | Replaced by v3 script (axis key bug, bar-only) |
| `download_arxiv_batch.py` | Data collection complete (32,364 charts) |
| `extract_backgrounds.py` | One-time YOLO background extraction, done |
| `generate_gradcam.py` | One-time research visualization, done |
| `generate_synthetic_dataset.py` | One-time synthetic data generation, done |
| `verify_qa_dataset.py` | One-time QA dataset verification, done |
| `_audit_data.py` | Absorbed by `_full_audit.py` |
| `_audit_stage3.py` | Absorbed by `_full_audit.py` |
| `_cross_check.py` | One-time QA cross-validation, done |

### Export to ONNX

```bash
.venv/Scripts/python.exe scripts/export_resnet18_onnx.py \
    --model models/weights/resnet18_chart_classifier_v2_best.pt \
    --output models/onnx/resnet18_chart_classifier_v2.onnx
```
