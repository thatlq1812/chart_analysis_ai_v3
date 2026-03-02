# Scripts Directory

Utility scripts for training, evaluation, data processing, and testing.

Last updated: 2026-03-02 (v4.3.0 -- reorganized into subdirectories)

## Directory Structure

```
scripts/
    training/           # Model training, data prep, cloud setup
    evaluation/         # Model evaluation and export
    pipeline/           # Pipeline testing and demo scripts
    utils/              # Download, audit, thesis generation
```

## training/

| Script | Purpose | Status |
|--------|---------|--------|
| `train_slm_lora.py` | Fine-tune SLM with QLoRA (4 models supported) | Production |
| `train_resnet18_v2.py` | Train ResNet-18 chart classifier | Production |
| `train_yolo_chart_detector.py` | YOLO chart detection training | Production |
| `prepare_slm_training_v3.py` | Build SLM training dataset v3 (268,799 samples) | Production |
| `extract_mini_dataset.py` | Stratified mini-dataset for model selection (5k train) | Production |
| `run_model_selection.py` | Multi-model micro-training + comparison | Production |
| `setup_cloud_training.sh` | Automated cloud GPU instance setup | Production |

## evaluation/

| Script | Purpose | Status |
|--------|---------|--------|
| `evaluate_slm.py` | SLM evaluation (EM, Contains, Numeric, BLEU-1) | Production |
| `evaluate_resnet18.py` | Evaluate ResNet-18 on test set | Ready |
| `export_resnet18_onnx.py` | Export ResNet-18 to ONNX format | Ready |

## pipeline/

| Script | Purpose | Status |
|--------|---------|--------|
| `batch_stage3_parallel.py` | Batch Stage 3 extraction (parallel, 8 workers) | Production |
| `demo_full_pipeline.py` | Demo full pipeline end-to-end | Ready |
| `test_full_pipeline.py` | Full pipeline integration test | Ready |
| `test_stage4.py` | Test Stage 4 reasoning | Ready |
| `test_element_detector.py` | Test element detection module | Ready |
| `test_resnet_integration.py` | Integration test for ResNet classifier | Ready |
| `test_qwen_slm.py` | Test Qwen SLM integration | Ready |

## utils/

| Script | Purpose | Status |
|--------|---------|--------|
| `download_models.py` | Download model weights | Ready |
| `context_scanner.py` | Scan project context for documentation | Ready |
| `_full_audit.py` | Quality audit of Stage3 features (all 8 types) | Ready |
| `generate_thesis_figures.py` | Generate thesis PDF figures | Ready |
| `generate_thesis_tables.py` | Generate thesis LaTeX tables | Ready |

## Usage Examples

### Model Selection (Micro-Training)

```bash
# 1. Extract stratified mini-dataset
.venv/Scripts/python.exe scripts/training/extract_mini_dataset.py

# 2. Dry-run to see plan
.venv/Scripts/python.exe scripts/training/run_model_selection.py \
    --dry-run --models llama-1b qwen-1.5b

# 3. Run model selection on cloud GPU
python scripts/training/run_model_selection.py \
    --models llama-1b qwen-1.5b \
    --data-dir data/slm_training_mini --epochs 3
```

### Full SLM Training

```bash
python scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --epochs 3 --max-length 4096
```

### SLM Evaluation

```bash
.venv/Scripts/python.exe scripts/evaluation/evaluate_slm.py \
    --base-model models/slm/llama-3.2-1b-instruct \
    --lora-path models/slm/llama-3.2-1b-instruct-chart-lora-v4/final \
    --test-data data/slm_training_v3/test.json \
    --max-samples 200 --stratified
```

### Batch Stage 3 Extraction

```bash
.venv/Scripts/python.exe scripts/pipeline/batch_stage3_parallel.py --workers 8
```

### Cloud GPU Setup

```bash
bash scripts/training/setup_cloud_training.sh
```
