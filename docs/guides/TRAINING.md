# Training Guide -- Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-02 | That Le | Comprehensive training guide for all model types |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Training Architecture](#2-training-architecture)
3. [Environment Setup](#3-environment-setup)
4. [Datasets](#4-datasets)
5. [SLM Fine-tuning (QLoRA)](#5-slm-fine-tuning-qlora)
6. [ResNet-18 Chart Classifier](#6-resnet-18-chart-classifier)
7. [YOLO Chart Detector](#7-yolo-chart-detector)
8. [Run Management System](#8-run-management-system)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Evaluation](#10-evaluation)
11. [Model Comparison (Thesis)](#11-model-comparison-thesis)
12. [Cloud GPU Training](#12-cloud-gpu-training)
13. [Troubleshooting](#13-troubleshooting)
14. [Quick Reference](#14-quick-reference)

---

## 1. Overview

### 1.1. Models in the System

The Geo-SLM system uses 3 model types, each serving a different pipeline stage:

| Model | Role | Stage | Size | Training Method |
| --- | --- | --- | --- | --- |
| **YOLOv8/v11** | Detect chart regions in document images | Stage 2 | ~6MB | Transfer learning (Ultralytics) |
| **ResNet-18** | Classify chart type (bar, line, pie, ...) | Stage 3 | ~45MB | Fine-tune (PyTorch) |
| **SLM (Llama/Qwen)** | Reasoning + OCR correction + value extraction | Stage 4 | ~2-6GB base + ~60MB LoRA | QLoRA fine-tuning (PEFT + trl) |

### 1.2. Training Principles

1. **Reproducibility**: Every run must have a frozen config, git commit hash, and metrics
2. **Isolation**: Every run is saved in its own directory (`runs/<name>/`)
3. **Fallback**: Local SLM is primary; cloud APIs are fallbacks
4. **Strict Data Split**: Train/val/test fully separated (no leakage)
5. **Config-Driven**: Use `--config config/training.yaml` instead of hardcoded values

### 1.3. Current Status (2026-03-02)

| Component | Status | Result |
| --- | --- | --- |
| ResNet-18 classifier | **DONE** | 94.14% accuracy (8 chart types) |
| Training data v3 | **DONE** | 268,799 ChatML samples |
| SLM training script | **DONE** | `train_slm_lora.py` (4 bugs fixed) |
| SLM training session 1 | **DONE** | EM=4% (failed -- see postmortem) |
| SLM training session 2 | **PENDING** | Waiting for cloud GPU (fixed config) |
| YOLO detector | **DONE** | mAP=0.92 |
| Model comparison | **PENDING** | Requires SLM training completion first |

---

## 2. Training Architecture

### 2.1. Relevant Directories

```
chart_analysis_ai_v3/
    config/
        training.yaml           # Main training config (SLM + run management)
        models.yaml             # Model paths, thresholds
        base.yaml               # Shared project config
    scripts/
        training/
            train_slm_lora.py           # SLM QLoRA fine-tuning
            train_resnet18_v2.py        # ResNet-18 chart classifier
            train_yolo_chart_detector.py # YOLO detection training
            prepare_slm_training_v3.py  # Build SLM training dataset
            run_model_selection.py      # Multi-model comparison runner
            extract_mini_dataset.py     # Create mini dataset for quick testing
            setup_cloud_training.sh     # Cloud GPU environment setup
        evaluation/
            evaluate_slm.py             # SLM benchmark (EM, BLEU, numeric)
            evaluate_resnet18.py        # ResNet-18 accuracy + confusion matrix
            export_resnet18_onnx.py     # ONNX export for inference
    src/
        training/
            run_manager.py              # Run isolation + config freezing
            experiment_tracker.py       # WandB/TensorBoard/JSON tracking
    data/
        slm_training_v3/               # Primary SLM dataset (268k samples)
        slm_training_mini/             # Mini dataset (model selection)
        academic_dataset/
            classified_charts/         # ResNet-18 training data
        yolo_chart_detection/          # YOLO training data
    models/
        slm/                           # Trained LoRA adapters
        weights/                       # ResNet-18, YOLO weights
    runs/                              # Isolated run directories (gitignored)
```

### 2.2. Overall Training Flow

```
[1] Prepare data
    |
    v
[2] Model selection (compare candidates on mini dataset)
    |
    v
[3] Full training (config-driven, isolated run)
    |
    v
[4] Evaluate on test set
    |
    v
[5] Cross-model comparison (thesis contribution)
    |
    v
[6] Integrate into pipeline (AIRouter + Adapters)
```

---

## 3. Environment Setup

### 3.1. Local (Windows, RTX 3060 6GB)

```bash
# Python virtual environment (MUST use .venv)
# See system.instructions.md Section 3

# Core training dependencies (already in pyproject.toml)
.venv/Scripts/pip.exe install torch transformers peft trl datasets accelerate bitsandbytes

# Optional: experiment tracking
.venv/Scripts/pip.exe install wandb tensorboard

# Verify GPU
.venv/Scripts/python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3.2. Cloud GPU

```bash
# Use the automated setup script:
bash scripts/training/setup_cloud_training.sh

# The script will:
# 1. Check GPU, Python, disk space
# 2. Create virtual environment
# 3. Install dependencies
# 4. Verify data availability
# 5. Test GPU compatibility (bf16, CUDA)
# 6. Print sample training commands
```

See [Section 12](#12-cloud-gpu-training) for detailed cloud setup instructions.

### 3.3. Verify Configuration

```bash
# Verify training script runs (no GPU required)
.venv/Scripts/python.exe scripts/training/train_slm_lora.py --help

# Smoke test (2 steps, verify end-to-end pipeline)
.venv/Scripts/python.exe scripts/training/train_slm_lora.py --smoke-test

# Verify run management imports
.venv/Scripts/python.exe -c "from src.training import RunManager, ExperimentTracker; print('OK')"
```

---

## 4. Datasets

### 4.1. SLM Training Data (v3)

| Property | Value |
| --- | --- |
| Total samples | 268,799 |
| Format | ChatML (conversations JSON) |
| Split | Train: 228,494 / Val: 26,888 / Test: 13,417 |
| Chart types | 8 types (line, scatter, bar, heatmap, histogram, box, pie, area) |
| Axis info coverage | 69.9% of samples include axis information |
| Location | `data/slm_training_v3/` |

**Distribution by chart type:**

| Chart Type | Samples | Percentage |
| --- | --- | --- |
| line | 108,419 | 40.3% |
| scatter | 52,163 | 19.4% |
| bar | 47,330 | 17.6% |
| heatmap | 33,373 | 12.4% |
| histogram | 8,438 | 3.1% |
| box | 7,362 | 2.7% |
| pie | 6,607 | 2.5% |
| area | 5,107 | 1.9% |

### 4.2. Building the Dataset

```bash
# Preview statistics (dry-run, no files written)
.venv/Scripts/python.exe scripts/training/prepare_slm_training_v3.py --dry-run

# Build the full dataset
.venv/Scripts/python.exe scripts/training/prepare_slm_training_v3.py \
    --output-dir data/slm_training_v3

# Cap samples per type (when low on RAM)
.venv/Scripts/python.exe scripts/training/prepare_slm_training_v3.py \
    --output-dir data/slm_training_v3 --max-per-type 3000
```

### 4.3. Mini Dataset (for Model Selection)

```bash
# Extract a mini dataset (~5-10k samples) from the full dataset
.venv/Scripts/python.exe scripts/training/extract_mini_dataset.py
```

The mini dataset is saved to `data/slm_training_mini/` and used for:
- Quick smoke tests
- Model selection experiments (comparing multiple models on the same data)
- Debugging the training pipeline

### 4.4. Data Format (ChatML)

Each sample follows this structure:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a chart analysis expert..."
    },
    {
      "role": "user",
      "content": "Chart Type: bar\n[TITLE]: Revenue\n[X_TICKS]: 2021, 2022, 2023\n[ELEMENTS]: bar=3\n..."
    },
    {
      "role": "assistant",
      "content": "{\"title\": \"Revenue\", \"chart_type\": \"bar\", ...}"
    }
  ],
  "metadata": {
    "chart_type": "bar",
    "question_type": "extraction",
    "curriculum_stage": 2
  }
}
```

### 4.5. ResNet-18 Training Data

| Property | Value |
| --- | --- |
| Total images | ~32,364 |
| Classes | 8 chart types |
| Label source | Gemini-corrected classification |
| Location | `data/academic_dataset/classified_charts/` |
| Split | 80/10/10 (stratified, performed in-script) |

### 4.6. YOLO Training Data

| Property | Value |
| --- | --- |
| Total images | Varies by version |
| Format | YOLO (images + labels .txt) |
| Config | `config/yolo_chart_v3.yaml` |
| Location | `data/yolo_chart_detection/` |

---

## 5. SLM Fine-tuning (QLoRA)

### 5.1. Overview

The `train_slm_lora.py` script uses:
- **PEFT** (QLoRA): Trains only ~1% of parameters, reduces VRAM from ~12GB to ~4GB
- **trl 0.29** (SFTTrainer + SFTConfig): HuggingFace supervised fine-tuning framework
- **BitsAndBytes**: 4-bit NF4 quantization
- **Cosine LR scheduler**: Warmup followed by cosine decay

### 5.2. Supported Models

| Key | Model | Params | VRAM (4-bit) | Notes |
| --- | --- | --- | --- | --- |
| `llama-1b` | Llama-3.2-1B-Instruct | 1.24B | ~2GB | PRIMARY |
| `qwen-1.5b` | Qwen2.5-1.5B-Instruct | 1.54B | ~2.5GB | CANDIDATE |
| `qwen-0.5b` | Qwen2.5-0.5B-Instruct | 0.49B | ~1.5GB | LIGHTWEIGHT |
| `llama-3b` | Llama-3.2-3B-Instruct | 3.21B | ~3.5GB | Requires >24GB VRAM |

**LoRA efficiency**: With rank=16, only ~11.27M parameters are trainable out of ~1.24B total (0.9%).

### 5.3. Default Configuration

The file `config/training.yaml` contains all hyperparameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `lora.rank` | 16 | LoRA rank (8-64). Higher = more capacity, more VRAM |
| `lora.alpha` | 32 (auto = rank x 2) | LoRA scaling factor |
| `lora.dropout` | 0.05 | Regularization |
| `training.num_train_epochs` | 3 | Number of epochs |
| `training.per_device_train_batch_size` | 2 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 8 | Effective batch = 2 x 8 = 16 |
| `training.learning_rate` | 2e-4 | Peak learning rate |
| `training.max_seq_length` | 4096 | **CRITICAL**: Must be >= p95 of sequence lengths |
| `training.lr_scheduler_type` | cosine | LR schedule type |
| `training.bf16` | true | BFloat16 precision (A100/RTX 3090+) |
| `quantization.use_4bit` | true | NF4 4-bit quantization |
| `training.eval_steps` | 100 | Evaluate every N steps |
| `training.save_steps` | 200 | Checkpoint every N steps |
| `training.save_total_limit` | 3 | Keep at most N checkpoints |

### 5.4. Training Commands

#### A. Config-driven (RECOMMENDED)

```bash
# Full training with YAML config (creates an isolated run directory)
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --config config/training.yaml

# Ablation study (override a single parameter)
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --config config/training.yaml \
    --override slm_training.training.learning_rate=1e-5

# Multiple overrides at once
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --config config/training.yaml \
    --override slm_training.lora.rank=32 \
    --override slm_training.training.num_train_epochs=5

# With WandB tracking
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --config config/training.yaml \
    --tracker wandb
```

#### B. CLI arguments (legacy, still compatible)

```bash
# Select model + hyperparameters directly
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 16 \
    --max-length 4096 \
    --gradient-accumulation-steps 4 \
    --eval-steps 500 \
    --save-steps 1000
```

#### C. Smoke test (verify pipeline)

```bash
# 2 training steps, no large data required
.venv/Scripts/python.exe scripts/training/train_slm_lora.py --smoke-test

# Smoke test with a specific model
.venv/Scripts/python.exe scripts/training/train_slm_lora.py --smoke-test --model qwen-0.5b
```

#### D. Resume training (incremental)

```bash
# Session 1: Train epoch 1
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b --epochs 1

# Session 2: Resume, train up to epoch 2 (auto-detects checkpoint)
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b --epochs 2 --resume

# Session 3: Resume, train up to epoch 3
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b --epochs 3 --resume

# Resume from a specific checkpoint
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --epochs 2 --resume-from-checkpoint models/slm/llama-3.2-1b-instruct-chart-lora/checkpoint-28500
```

### 5.5. Training Checklists

#### Before Training

- [ ] Verify `max_length >= 4096` (or >= p95 of sequence lengths)
- [ ] Verify pad_token is NOT eos_token (Llama-3 specific -- already fixed in script)
- [ ] Check VRAM: A100 supports batch=4, RTX 3060 supports batch=1-2
- [ ] Close heavy applications (Chrome, VS Code debugger) when training locally
- [ ] Check disk space (each checkpoint ~60MB, need ~500MB per run)
- [ ] Ensure data exists at `data/slm_training_v3/`
- [ ] Run smoke test first: `--smoke-test`
- [ ] For cloud: upload data + scripts and verify GPU >= 24GB VRAM

#### During Training

- [ ] Monitor loss via terminal output (`logging_steps=10`)
- [ ] Watch disk space (checkpoints accumulate up to `save_total_limit`)
- [ ] On cloud: keep training inside tmux to survive SSH disconnects
- [ ] Periodically check `nvidia-smi` for GPU utilization (should be >90%)

#### After Training

- [ ] Read final `train_loss` and `eval_loss` from trainer_state.json
- [ ] Run quick inference check (Section 10.3) to verify JSON output format
- [ ] Run formal evaluation: `evaluate_slm.py`
- [ ] Download LoRA adapter to local machine (~60MB) if trained on cloud
- [ ] Generate comparison table (Section 11)
- [ ] Update `docs/MASTER_CONTEXT.md` and `docs/CHANGELOG.md`

### 5.6. Output Structure

```
# With --config (RunManager):
runs/slm_lora_llama-1b_20260302_153022/
    resolved_config.yaml        # Frozen config (YAML + CLI overrides)
    run_metadata.json           # Run ID, git commit, timestamps
    training_info.json          # Model params, dataset stats, sessions
    checkpoints/                # HuggingFace checkpoints
        checkpoint-1000/
        checkpoint-2000/
    logs/                       # TensorBoard / JSON metrics
    artifacts/                  # Custom outputs
    final/                      # Final LoRA adapter weights
        adapter_config.json
        adapter_model.safetensors   (~60MB)
        tokenizer.json
        tokenizer_config.json

# Without --config (legacy):
models/slm/llama-3.2-1b-instruct-chart-lora/
    checkpoint-1000/
    checkpoint-2000/
    final/
    training_info.json
    trainer_state.json
```

### 5.7. Monitoring Loss

#### Reading from trainer_state.json

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path

# Adjust path accordingly
state_path = Path('runs/<RUN_NAME>/checkpoints/trainer_state.json')
if state_path.exists():
    state = json.loads(state_path.read_text())
    print('Best metric:', state.get('best_metric'))
    print('Best checkpoint:', state.get('best_model_checkpoint'))
    print()
    evals = [e for e in state.get('log_history', []) if 'eval_loss' in e]
    for e in evals[-10:]:
        print(f'  step={e[\"step\"]:>6}  epoch={e.get(\"epoch\", \"?\"):.2f}  eval_loss={e[\"eval_loss\"]:.4f}')
"
```

#### Convergence Indicators

| Situation | train_loss | eval_loss | Action |
| --- | --- | --- | --- |
| Normal | Decreasing steadily | Decreasing in parallel | Continue training |
| Overfitting | Still decreasing | Starts increasing | Stop, use best checkpoint |
| Underfitting | Decreasing slowly | Decreasing slowly | Add epochs or increase learning rate |
| Divergence | Increasing or NaN | NaN | Reduce learning rate, check data |

#### Expected Targets by Epoch

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| train_loss | ~2.0 | ~1.5 | ~1.2 |
| eval_loss | ~2.2 | ~1.7 | ~1.4 |

### 5.8. How `--resume` Works Internally

The `--resume` flag uses `find_latest_checkpoint()` to scan the output directory:

```
models/slm/llama-3.2-1b-chart-lora-v4/
    checkpoint-14250/    <- mid epoch 1
    checkpoint-28500/    <- end of epoch 1  <-- SELECTED (highest step)
    checkpoint-2/        <- smoke test (ignored, too small)
```

The selected checkpoint path is passed to `trainer.train(resume_from_checkpoint=...)`. HuggingFace Trainer then:
1. Reloads LoRA weights from the checkpoint
2. Reloads optimizer state (no warmup restart needed)
3. Calculates steps already completed and skips finished epochs
4. Continues training from the next epoch

To specify a checkpoint manually instead of auto-detection:

```bash
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --epochs 3 \
    --resume-from-checkpoint models/slm/llama-3.2-1b-chart-lora-v4/checkpoint-28500
```

### 5.9. Lessons from Session 1 (Postmortem)

> Full details: `docs/reports/SLM_TRAINING_POSTMORTEM_V1.md`

Session 1 (local RTX 3060, 1 epoch) produced EM=4% due to 4 configuration bugs:

| Bug | Severity | Fix Applied |
| --- | --- | --- |
| `max_length=512` | **FATAL** | Changed to 4096 (model never saw ground truth) |
| `lora_alpha=32` hardcoded | Medium | Auto-computed as rank x 2 |
| `pad_token = eos_token` | Medium | Use `<\|finetune_right_pad_id\|>` for Llama-3 |
| `gradient_accumulation=4` | Low | Changed to 8 (effective batch=16) |

**All 4 bugs have been fixed** in the current version of `train_slm_lora.py`.

---

## 6. ResNet-18 Chart Classifier

### 6.1. Overview

ResNet-18 is fine-tuned to classify 8 chart types. Current result: **94.14% accuracy**.

### 6.2. Training

#### Config-driven (recommended)

```bash
.venv/Scripts/python.exe scripts/training/train_resnet18_v2.py \
    --config config/training.yaml

# Ablation override
.venv/Scripts/python.exe scripts/training/train_resnet18_v2.py \
    --config config/training.yaml \
    --override resnet.lr=5e-4 \
    --override resnet.epochs=80
```

#### Legacy CLI

```bash
.venv/Scripts/python.exe scripts/training/train_resnet18_v2.py \
    --epochs 60 --batch-size 64 --lr 1e-4
```

### 6.3. Key Features

- **Data augmentation**: Grayscale-first (suited for academic chart images), random flip, rotation
- **Class-balanced**: `WeightedRandomSampler` to handle underrepresented chart types
- **Mixed precision**: fp16 training for faster throughput
- **Cosine annealing**: LR schedule with warmup
- **Best model saving**: Only saves the model with the highest val_accuracy

### 6.4. Evaluation

```bash
.venv/Scripts/python.exe scripts/evaluation/evaluate_resnet18.py

# Output:
# - Classification report (precision, recall, F1 per class)
# - Confusion matrix heatmap
# - Misclassified examples
```

### 6.5. ONNX Export (for Production Inference)

```bash
.venv/Scripts/python.exe scripts/evaluation/export_resnet18_onnx.py
# Output: models/onnx/resnet18_chart_classifier.onnx
```

---

## 7. YOLO Chart Detector

### 7.1. Overview

YOLOv8/v11 is trained to detect chart regions in document page images. Current result: **mAP=0.92**.

### 7.2. Training

```bash
# Using the project training script
.venv/Scripts/python.exe scripts/training/train_yolo_chart_detector.py

# Or directly with the Ultralytics CLI:
yolo detect train data=config/yolo_chart_v3.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 7.3. Configuration

File `config/yolo_chart_v3.yaml` contains:
- Dataset paths (train/val/test)
- Class definitions
- Image size and augmentation settings

---

## 8. Run Management System

### 8.1. Architecture

When `--config` is provided, `RunManager` automatically:

1. **Merges config**: `base.yaml` + `training.yaml` + `--override` values (via OmegaConf)
2. **Creates an isolated directory**: `runs/<prefix>_<timestamp>/`
3. **Freezes config**: Saves `resolved_config.yaml` (immutable snapshot)
4. **Computes config hash**: SHA-256 for reproducibility comparison
5. **Saves metadata**: Git commit hash, timestamps, environment info
6. **Finalizes**: Updates the run registry upon completion

### 8.2. Config Resolution Priority

```
LOW                                                 HIGH
  |                                                   |
  v                                                   v
base.yaml  ->  training.yaml  ->  --override KEY=VAL  ->  --cli-args
  (global)    (training-specific)  (dynamic override)     (direct CLI)
```

Example: If `training.yaml` has `learning_rate=2e-4` and the CLI has `--learning-rate 1e-5`, the value **1e-5** is used.

### 8.3. Run Directory Structure

```
runs/slm_lora_llama-1b_20260302_153022/
    resolved_config.yaml    # Frozen config snapshot
    run_metadata.json       # {"run_id": "...", "git_commit": "...", "started_at": "..."}
    training_info.json      # Model info, dataset stats, session history
    checkpoints/            # HuggingFace trainer checkpoints
    logs/                   # Metrics logs (JSON/TensorBoard)
    artifacts/              # Evaluation results, custom outputs
    final/                  # Final LoRA adapter weights
```

### 8.4. Run Registry

All runs are indexed in `runs/run_registry.json`:

```python
from src.training.run_manager import RunManager

# List all completed runs
completed = RunManager.list_runs(status="completed")
for run in completed:
    print(f"{run['run_name']} | config_hash={run['config_hash'][:12]} | metrics={run['metrics']}")

# Load config from a previous run
old_config = RunManager.load_run_config("runs/slm_lora_llama-1b_20260302_153022")
```

### 8.5. Run Management Rules

1. **ALWAYS** use `--config` for production training runs
2. **NEVER** delete run directories before explicitly archiving
3. Config is frozen automatically before training starts
4. Registry is updated automatically upon finalization
5. `runs/` is gitignored -- only commit configs and code

---

## 9. Experiment Tracking

### 9.1. Supported Backends

| Backend | Setup Required | When to Use |
| --- | --- | --- |
| `json` (default) | None | Development, quick tests |
| `tensorboard` | `pip install tensorboard` | Local visualization |
| `wandb` | `pip install wandb && wandb login` | Thesis figures, ablation studies |
| `none` | -- | When tracking is not needed |

### 9.2. Selecting a Backend

```bash
# Via CLI flag
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --config config/training.yaml --tracker wandb

# Via config file (config/training.yaml)
run_management:
  tracking_backend: "wandb"     # Default for all runs
```

### 9.3. Fallback Chain

If the selected backend is unavailable (e.g., wandb not installed), the system automatically falls back:

```
wandb (not installed) -> tensorboard (not installed) -> json (always works) -> none
```

### 9.4. Viewing Metrics

```bash
# JSON logs: read directly
cat runs/<RUN_NAME>/logs/metrics.jsonl

# TensorBoard
.venv/Scripts/python.exe -m tensorboard.main --logdir runs/<RUN_NAME>/logs

# WandB: view on web dashboard (synced automatically)
```

### 9.5. HuggingFace Trainer Integration

`ExperimentTracker.get_report_to()` returns a list compatible with SFTConfig:

```python
tracker = ExperimentTracker(backend="tensorboard", ...)
sft_config = SFTConfig(
    report_to=tracker.get_report_to(),  # ["tensorboard"]
    logging_dir=str(run_manager.logs_dir),
    ...
)
```

---

## 10. Evaluation

### 10.1. SLM Evaluation

```bash
# Evaluate a LoRA fine-tuned model
.venv/Scripts/python.exe scripts/evaluation/evaluate_slm.py \
    --base-model models/slm/llama-3.2-1b-instruct \
    --lora-path models/slm/llama-3.2-1b-chart-lora-v4/final \
    --test-data data/slm_training_v3/test.json \
    --output models/evaluation/llama-1b-lora-v4.json \
    --max-samples 500 --stratified

# Evaluate base model (zero-shot, for comparison)
.venv/Scripts/python.exe scripts/evaluation/evaluate_slm.py \
    --base-model models/slm/llama-3.2-1b-instruct \
    --test-data data/slm_training_v3/test.json \
    --output models/evaluation/llama-1b-base.json \
    --max-samples 500

# Smoke test (10 samples)
.venv/Scripts/python.exe scripts/evaluation/evaluate_slm.py \
    --base-model models/slm/llama-3.2-1b-instruct \
    --lora-path models/slm/llama-3.2-1b-chart-lora-v4/final \
    --test-data data/slm_training_v3/test.json \
    --output models/evaluation/smoke.json \
    --max-samples 10
```

### 10.2. SLM Metrics

| Metric | Description | Target |
| --- | --- | --- |
| Exact Match (EM) | Normalized string match | >50% |
| Contains Match | Reference answer found in prediction | >70% |
| Numeric Accuracy | Numbers correct within 5% tolerance | >85% |
| BLEU-1 | Unigram overlap for longer answers | >0.5 |
| JSON Valid Rate | % of outputs that are valid JSON | >95% |
| Latency | Inference time per sample | <2s |
| VRAM Peak | Peak GPU memory during inference | <4GB |

### 10.3. Quick Inference Check

After each epoch, quickly verify that the model has learned something:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE = "models/slm/llama-3.2-1b-instruct"
LORA = "runs/<RUN_NAME>/final"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, LORA)

messages = [
    {"role": "system", "content": "You are a chart analysis expert."},
    {"role": "user", "content": "Chart Type: bar\nOCR Texts: ['Reverue', '2021', '2022', '2023']\nDetected Elements: 3 bars\nAxis Info: x=['2021','2022','2023'], y_range=[0,25]\n\nExtract as JSON."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
print(tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

**Expected output after epoch 1:**
```json
{"chart_type": "bar", "x_axis": "Year", "series": [...]}
```
Output may be incomplete -- the primary check is whether the model has learned the JSON format.

### 10.4. ResNet-18 Evaluation

```bash
.venv/Scripts/python.exe scripts/evaluation/evaluate_resnet18.py
# Output: classification report, confusion matrix, misclassified examples
```

---

## 11. Model Comparison (Thesis)

### 11.1. Experiment Design

This is the **primary academic contribution** of the thesis: demonstrating that a fine-tuned 1-3B SLM can approach cloud LLM quality for chart analysis tasks.

| Model | Size | Method | Test Set |
| --- | --- | --- | --- |
| Llama-3.2-1B (base) | 1B | Zero-shot | test.json |
| Llama-3.2-1B (LoRA v4) | 1B + 60MB | Fine-tuned | test.json |
| Qwen2.5-1.5B (base) | 1.5B | Zero-shot | test.json |
| Qwen2.5-1.5B (LoRA) | 1.5B + 60MB | Fine-tuned | test.json |
| Gemini 2.0 Flash | Cloud | Zero-shot API | test.json |
| GPT-4o-mini | Cloud | Zero-shot API | test.json |

### 11.2. Model Selection (Before Full Training)

Run model selection on the mini dataset to pick the best candidate before investing compute:

```bash
# Compare Llama-1B vs Qwen-1.5B on mini dataset
.venv/Scripts/python.exe scripts/training/run_model_selection.py \
    --data-dir data/slm_training_mini \
    --models llama-1b qwen-1.5b \
    --epochs 3

# All models
.venv/Scripts/python.exe scripts/training/run_model_selection.py \
    --data-dir data/slm_training_mini \
    --models llama-1b qwen-1.5b qwen-0.5b \
    --epochs 3 --eval-samples 200

# Evaluate only (skip training)
.venv/Scripts/python.exe scripts/training/run_model_selection.py \
    --eval-only \
    --eval-dirs models/slm/llama-3.2-1b-instruct-chart-lora-micro/final \
               models/slm/qwen2.5-1.5b-instruct-chart-lora-micro/final
```

### 11.3. Comparison Table (Thesis Output)

| Model | EM% | Contains% | Numeric% | BLEU-1 | Latency | Cost/1K |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B (base) | 0.0% | 9.0% | 36.2% | 0.063 | 7.04s | $0 |
| Llama-3.2-1B (LoRA v3, broken) | 4.0% | 8.0% | 17.5% | 0.281 | 1.34s | $0 |
| Llama-3.2-1B (LoRA v4, fixed) | ? | ? | ? | ? | ? | $0 |
| Qwen2.5-1.5B (LoRA) | ? | ? | ? | ? | ? | $0 |
| Gemini 2.0 Flash | - | - | ~99%* | - | API | ~$X |

*This table will be completed after SLM training is done.*

---

## 12. Cloud GPU Training

### 12.1. Why Use a Cloud GPU

| Comparison | Local (RTX 3060 6GB) | Cloud (A100 40GB) |
| --- | --- | --- |
| VRAM | 6 GB | 40-80 GB |
| Time per epoch | ~13-15 hours | ~1-3 hours |
| Max feasible max_length | 1024-2048 (OOM risk) | 4096+ |
| batch_size | 1-2 | 4-16 |
| Cost for 3 epochs | $0 | ~$3-9 |
| Risks | Thermal throttling, OOM | None |

### 12.2. Recommended Strategy

Training runs **3 epochs continuously** on cloud GPU (no need to split sessions like on local):

```
Cloud GPU Session (~3-9h total)
    |
  Epoch 1 -> Epoch 2 -> Epoch 3
  ~1-3h      ~1-3h      ~1-3h
    |          |          |
  checkpoint  checkpoint  final/
    |          |          |
  [Auto eval] [Auto eval] [Full eval]
```

If the rental time is limited, use `--resume` to split across sessions (see Section 5.8).

### 12.3. Recommended Providers

| Provider | GPU | Estimated Cost | Notes |
| --- | --- | --- | --- |
| RunPod | A100 40GB | ~$1.5-2/h | Terraform support |
| Vast.ai | A100/L4 | ~$1-2/h | Auction pricing |
| Lambda | A100 | ~$1.5/h | Bare metal |
| Google Colab Pro | T4/A100 | ~$10/month | Simplest option |


### 12.4. Workflow

```
[1] Prepare a bundle on local machine
    |-- tar data + scripts + config
    |
[2] Rent a cloud GPU instance (A100 40GB)
    |
[3] Upload + set up environment
    |-- bash scripts/training/setup_cloud_training.sh
    |
[4] Smoke test inside tmux
    |-- python train_slm_lora.py --smoke-test
    |
[5] Full training inside tmux
    |-- python train_slm_lora.py --config config/training.yaml
    |
[6] Download results to local machine
    |-- scp -r final/ + training_info.json + trainer_state.json
    |
[7] Evaluate locally
    |-- python evaluate_slm.py --lora-path ...
```

### 12.5. Preparing Data for Upload

```bash
# Pack required files
tar -czf slm_training_bundle.tar.gz \
    scripts/training/train_slm_lora.py \
    scripts/training/setup_cloud_training.sh \
    scripts/evaluation/evaluate_slm.py \
    src/training/ \
    data/slm_training_v3/ \
    config/training.yaml \
    config/base.yaml \
    pyproject.toml
```

**Base model options:**
- Upload separately (~2.4GB): `tar -czf base_model.tar.gz models/slm/llama-3.2-1b-instruct/`
- Or download directly on cloud from HuggingFace: use `--model meta-llama/Llama-3.2-1B-Instruct` (requires `huggingface-cli login`)

### 12.6. Setting Up the Server

```bash
# 1. Upload and extract
scp slm_training_bundle.tar.gz user@server:/workspace/
ssh user@server
cd /workspace && tar -xzf slm_training_bundle.tar.gz

# 2. Run the setup script
bash scripts/training/setup_cloud_training.sh

# 3. Start tmux (keeps training alive if SSH disconnects)
tmux new -s training

# 4. Smoke test
python scripts/training/train_slm_lora.py --smoke-test --model llama-1b

# 5. Full training
python scripts/training/train_slm_lora.py \
    --config config/training.yaml \
    --model llama-1b \
    --batch-size 4 \
    --max-length 4096

# 6. Detach tmux: Ctrl+B then D
# 7. Reattach: tmux attach -t training
```

### 12.7. Downloading Results

```bash
# Only the LoRA adapter (~60MB) + metadata are needed
scp -r user@server:/workspace/runs/<RUN_NAME>/final/ \
    runs/<RUN_NAME>/final/

scp user@server:/workspace/runs/<RUN_NAME>/training_info.json \
    user@server:/workspace/runs/<RUN_NAME>/resolved_config.yaml \
    runs/<RUN_NAME>/
```

### 12.8. Cloud Tips

- **ALWAYS use tmux** to keep training alive when SSH disconnects
- **Monitor GPU**: `watch -n 2 nvidia-smi` in a separate tmux pane
- **Check disk**: Each checkpoint is ~60MB; ensure enough disk for `save_total_limit` checkpoints
- **Save money**: Shut down the instance immediately after training completes

---

## 13. Troubleshooting

### 13.1. CUDA Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

Solutions in order of priority:
1. Reduce `--batch-size 1`
2. Increase `--gradient-accumulation-steps 16` (maintains effective batch size)
3. Reduce `--max-length 2048` (**NEVER go below 1024**)
4. Verify 4-bit quantization is enabled (default is on; check `--no-4bit`)
5. Use a larger GPU

### 13.2. eval_loss Not Decreasing

- Learning rate may be too high -- try `--learning-rate 1e-4`
- max_length may be too small -- check p95 of sequence lengths
- Data quality issue -- run `prepare_slm_training_v3.py --dry-run` to verify

### 13.3. Model Outputs Nonsense After Training

- Check max_length (see postmortem Bug 1)
- Check pad_token (see postmortem Bug 3)
- Run the quick inference check (Section 10.3)
- Compare against base model zero-shot output

### 13.4. Training Too Slow

- Check `nvidia-smi` -- GPU utilization should be > 90%
- Increase batch_size if VRAM allows
- Reduce eval_steps (fewer evaluations = faster)
- Disable gradient_checkpointing (uses more VRAM but increases speed)

### 13.5. WandB/TensorBoard Not Working

```bash
# Verify installation
.venv/Scripts/pip.exe install wandb tensorboard

# Verify WandB login
wandb login

# Fallback: use JSON tracking (always works)
--tracker json
```

### 13.6. Resume Cannot Find Checkpoint

```bash
# Check whether checkpoints exist
ls runs/<RUN_NAME>/checkpoints/

# Use an explicit checkpoint path
--resume-from-checkpoint runs/<RUN_NAME>/checkpoints/checkpoint-1000
```

### 13.7. Machine Crashed Mid-Training

No data is lost. The most recent checkpoint (within the last `save_steps` steps) still exists. Use `--resume` to continue:

```bash
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b --epochs 3 --resume
```

---

## 14. Quick Reference

### 14.1. Common Commands

| Purpose | Command |
| --- | --- |
| Smoke test | `train_slm_lora.py --smoke-test` |
| Full SLM training | `train_slm_lora.py --config config/training.yaml` |
| ResNet training | `train_resnet18_v2.py --config config/training.yaml` |
| Model selection | `run_model_selection.py --data-dir data/slm_training_mini --models llama-1b qwen-1.5b` |
| Evaluate SLM | `evaluate_slm.py --base-model ... --lora-path ... --test-data ...` |
| Evaluate ResNet | `evaluate_resnet18.py` |
| Build dataset | `prepare_slm_training_v3.py` |
| Cloud setup | `bash setup_cloud_training.sh` |
| View run history | `RunManager.list_runs()` |

### 14.2. Key Files

| File | Purpose |
| --- | --- |
| `config/training.yaml` | All hyperparameters + run management config |
| `src/training/run_manager.py` | Run isolation, config freezing, registry |
| `src/training/experiment_tracker.py` | Tracking abstraction (wandb/tb/json) |
| `runs/run_registry.json` | Index of all training runs |
| `docs/reports/SLM_TRAINING_POSTMORTEM_V1.md` | Lessons from training session 1 |

### 14.3. Environment Variables

| Variable | Purpose | Example |
| --- | --- | --- |
| `HF_HOME` | HuggingFace model cache directory | `export HF_HOME=~/.cache/hf` |
| `WANDB_API_KEY` | WandB API key | `export WANDB_API_KEY=...` |
| `CUDA_VISIBLE_DEVICES` | Select GPU device | `export CUDA_VISIBLE_DEVICES=0` |

### 14.4. Dependencies

```
# Core training
torch >= 2.0
transformers >= 4.40
peft >= 0.10
trl >= 0.29
datasets >= 2.19
accelerate >= 0.30
bitsandbytes >= 0.43

# Optional tracking
wandb >= 0.16
tensorboard >= 2.15

# ResNet training
torchvision >= 0.15
scikit-learn
matplotlib
seaborn
tqdm
```
