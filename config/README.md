# Config Directory

This document describes every configuration file and when to load it.

## File Overview

```
config/
    base.yaml          Shared constants (logging, directory paths, session format)
    models.yaml        AI model weights (YOLO, classifier, OCR, SLM) + inference settings
    pipeline.yaml      Stage toggles + adapter selection (which model backend per stage)
    training.yaml      All training hyperparameters (classifier + SLM fine-tuning)
    yolo_chart_v3.yaml YOLO dataset split paths (consumed directly by ultralytics CLI)
    secrets/           API keys, never committed (see .env.example at project root)
```

---

## `base.yaml`

**Purpose**: Project-wide constants shared by every component.

**Loaded by**: `ChartAnalysisPipeline.from_config()` — always merged first.

**Key sections**:

| Section | Content |
|---|---|
| `logging` | Log level, format string, output directory |
| `data` | Canonical paths for raw/processed/cache/samples |
| `output` | Reports and export directories |
| `session` | ID prefix and timestamp format for pipeline runs |

**When to edit**: Rarely. Only when changing top-level directory layout.

---

## `models.yaml`

**Purpose**: Paths and inference settings for every AI model used at runtime.

**Loaded by**: `ChartAnalysisPipeline.from_config()` — merged after `base.yaml`.

**Key sections**:

| Section | Content |
|---|---|
| `yolo` | Detection model path, confidence/IoU thresholds, input size |
| `classifier` | Chart type classifier path, class list, confidence threshold |
| `ocr` | Engine choice (`paddleocr`/`tesseract`) and per-engine settings |
| `slm` | Local SLM name, generation parameters, LoRA adapter path |

**When to edit**:
- After training a new classifier → update `classifier.path` and `classifier.classes`.
- After training a new YOLO model → update `yolo.path`.
- After SLM fine-tuning → uncomment `slm.lora_path`.

---

## `pipeline.yaml`

**Purpose**: Controls which stages are active and which adapter each stage uses.

**Loaded by**: `ChartAnalysisPipeline.from_config()` — merged last (highest priority).

**Key sections**:

| Section | Content |
|---|---|
| `pipeline.stages.ingestion` | Enable/disable Stage 1 |
| `pipeline.stages.detection` | Enable/disable Stage 2 + `adapter: yolov8\|yolov11\|mock` |
| `pipeline.stages.extraction` | Enable/disable Stage 3 |
| `pipeline.stages.reasoning` | Enable/disable Stage 4 |
| `pipeline.stages.reporting` | Enable/disable Stage 5 |
| `ingestion.*` | PDF DPI, image size constraints, blur threshold |
| `detection.*` | Confidence threshold override for Stage 2 |
| `extraction.*` | OCR merge threshold, geometric analysis settings |
| `reasoning.*` | Prompt template paths |
| `reporting.*` | Insight generation toggle |

**Adapter swap (Stage 2)**:
```yaml
pipeline:
  stages:
    detection:
      adapter: yolov11   # yolov8 | yolov11 | mock
```

**When to edit**: When toggling stages for a run, or switching YOLO adapter.

---

## `training.yaml`

**Purpose**: All training hyperparameters. NOT merged into the pipeline config.
Read exclusively by training scripts via `RunManager`.

**Key sections**:

| Section | Script | Purpose |
|---|---|---|
| `run_management` | All training scripts | Tracking backend (wandb/tensorboard/json), runs directory |
| `classifier` | `train_chart_classifier.py` | Chart type classifier training (ResNet, EfficientNet, etc.) |
| `slm_training` | `train_slm_lora.py` | SLM QLoRA fine-tuning (Qwen/Llama) |

### `classifier` section

Controls all chart classifier experiments via a single unified script.

**Mode selection** (`classifier.mode`):

| Mode | Classes | Use case |
|---|---|---|
| `4class` | bar / line / pie / others | Thesis (default) |
| `8class` | area / bar / box / heatmap / histogram / line / pie / scatter | Research baseline |

**Backbone selection** (`classifier.model.backbone`):

| Backbone | Params | GPU speed | Notes |
|---|---|---|---|
| `resnet18` | 11.7M | fast | Default. Strong baseline, well studied |
| `resnet34` | 21.8M | medium | Marginal gain, not recommended |
| `resnet50` | 25.6M | slow | Overkill for 4-class |
| `efficientnet_b0` | 5.3M | medium | Best accuracy/size ratio. Use for ablation |
| `efficientnet_b1` | 7.8M | medium | Marginal gain over b0 |
| `mobilenet_v3_small` | 2.5M | fastest | Edge deployment only |
| `mobilenet_v3_large` | 5.5M | fast | Good production trade-off |

**Typical ablation runs for thesis**:
```bash
# Run A: ResNet-18 baseline (4-class)
python scripts/training/train_chart_classifier.py

# Run B: EfficientNet-B0 ablation
python scripts/training/train_chart_classifier.py \
    --override classifier.model.backbone=efficientnet_b0 \
    --override classifier.output.model_name=efficientnet_b0_4class_v3

# Run C: ResNet-18 on 8-class (research baseline)
python scripts/training/train_chart_classifier.py \
    --override classifier.mode=8class \
    --override classifier.output.model_name=resnet18_8class_v3
```

---

## `yolo_chart_v3.yaml`

**Purpose**: YOLO dataset configuration consumed directly by the Ultralytics CLI.
This file is NOT part of the OmegaConf merge chain.

**Used by**:
```bash
yolo train model=yolov8n.pt data=config/yolo_chart_v3.yaml epochs=100
```

**Key fields**: `path`, `train`, `val`, `test` (relative to `path`), `names` (class list).

**When to edit**: When dataset split paths change (e.g. after running `prepare_dataset_splits.py`).

---

## `secrets/`

API keys and credentials. Never committed to git (`.gitignore`d).

See `.env.example` at project root for all required variables.

**Typical contents** (create manually):
```
secrets/
    gemini_api_key.txt
    openai_api_key.txt
```

Or use environment variables directly — the pipeline reads from `.env` via `python-dotenv`.

---

## Config Merge Order (Runtime Pipeline)

```
base.yaml           (lowest priority — shared defaults)
    +
models.yaml         (model paths override base paths if needed)
    +
pipeline.yaml       (highest priority — stage toggles and adapter selection)
    =
Resolved DictConfig (passed to ChartAnalysisPipeline)
```

CLI overrides (via `PipelineBuilder`) are applied after the merge and have the highest priority.
