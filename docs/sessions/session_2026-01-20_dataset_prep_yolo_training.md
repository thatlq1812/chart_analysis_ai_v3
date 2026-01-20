# Session: Dataset Preparation & YOLO Training

| Field | Value |
|-------|-------|
| Date | 2026-01-20 |
| Duration | ~4 hours |
| Author | That Le + AI Agent |
| Status | Training In Progress |

## 1. Session Objectives

1. Complete ArXiv PDF mining (885 PDFs)
2. Prepare combined dataset for YOLO training
3. Create data splits (train/val/test)
4. Auto-label all images for YOLO format
5. Start YOLO training

## 2. Tasks Completed

### 2.1. ArXiv Mining (Completed)

**Command:**
```bash
.venv/Scripts/python -m tools.data_factory mine --source arxiv --limit 1000
```

**Results:**
- Total PDFs processed: 885
- Images extracted: 12,839
- Success rate: ~14.5 images/PDF average
- Output: `data/academic_dataset/arxiv/`

**Bug Fixed:** PIL encoding error in `miner.py` - resolved using BytesIO buffer approach.

### 2.2. Dataset Composition

| Source | Images | Description |
|--------|--------|-------------|
| ChartQA (HuggingFace) | 20,000 | Pre-labeled chart images |
| ArXiv Mining | 12,839 | Extracted from academic PDFs |
| Synthetic | 447 | Generated charts |
| **Total** | **33,286** | Combined dataset |

### 2.3. Dataset Split

**Command:**
```bash
.venv/Scripts/python scripts/prepare_dataset_splits.py --force
```

**Results:**

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 26,271 | 80% |
| Val | 3,283 | 10% |
| Test | 3,285 | 10% |
| **Total** | **32,839** | 100% |

**Output Structure:**
```
data/training/
├── images/
│   ├── train/    # 26,271 images
│   ├── val/      # 3,283 images
│   └── test/     # 3,285 images
├── labels/
│   ├── train/    # 26,271 .txt files
│   ├── val/      # 3,283 .txt files
│   └── test/     # 3,285 .txt files
└── dataset.yaml  # YOLO config
```

### 2.4. Auto-Labeling

**Strategy:** Full-image bounding box with random margin (1-5%)

Since all images are already cropped charts, we label each image as containing one chart covering most of the image area.

**YOLO Label Format:**
```
class_id x_center y_center width height
0 0.500000 0.500000 0.928846 0.928846
```

- `class_id`: 0 (chart)
- `x_center, y_center`: 0.5, 0.5 (centered)
- `width, height`: 0.92-0.98 (full image minus random margin)

**Script:** `scripts/auto_label_dataset.py`

### 2.5. YOLO Training (In Progress)

**Command:**
```bash
yolo train model=yolov8n.pt data=D:/elix/chart_analysis_ai_v3/data/training/dataset.yaml epochs=100 imgsz=640 batch=16 project=D:/elix/chart_analysis_ai_v3/results/training_runs name=chart_detector_v3
```

**Configuration:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | YOLOv8n | Nano - fastest |
| Epochs | 100 | Standard training |
| Image Size | 640x640 | YOLO default |
| Batch Size | 16 | CPU-friendly |
| Classes | 1 | "chart" only |

**Output Directory:** `results/training_runs/chart_detector_v3/`

## 3. Files Created/Modified

### Created:
- `config/yolo_chart_v3.yaml` - YOLO dataset configuration
- `notebooks/01_data_exploration.ipynb` - EDA notebook
- `scripts/auto_label_dataset.py` - Auto-labeling script
- `scripts/fix_duplicate_labels.py` - Label cleanup utility
- `data/training/dataset.yaml` - Training dataset config
- `data/training/images/{train,val,test}/` - Image splits
- `data/training/labels/{train,val,test}/` - Label files

### Modified:
- `tools/data_factory/services/miner.py` - BytesIO fix for PIL encoding
- `tools/data_factory/services/generator.py` - Added YOLO label output

## 4. Key Decisions

### Decision 1: Single-Class Detection
- **Choice:** Train with only "chart" class (not bar/line/pie separately)
- **Rationale:** Phase 1 focus on detection accuracy; classification in later stages

### Decision 2: Auto-Labeling Approach
- **Choice:** Full-image bbox with random margin
- **Rationale:** Images already cropped = each image IS a chart
- **Trade-off:** Less precise than manual annotation, but enables training with 32k+ images

### Decision 3: YOLOv8 Nano
- **Choice:** Smallest model for fastest iteration
- **Rationale:** CPU training, quick validation of pipeline
- **Future:** Scale to YOLOv8s/m after validation

## 5. Metrics to Monitor

During training, watch for:
- **mAP@0.5**: Target > 0.85
- **mAP@0.5:0.95**: Target > 0.70
- **Loss convergence**: Should decrease steadily
- **Overfitting**: Val loss should track train loss

## 6. Next Steps

1. [ ] Monitor training progress (check `results/training_runs/chart_detector_v3/`)
2. [ ] Evaluate best.pt on test set
3. [ ] Integrate trained model into pipeline Stage 2
4. [ ] Consider fine-tuning with multi-class labels (bar/line/pie)
5. [ ] Test on real-world PDF documents

## 7. Commands Reference

```bash
# Check training progress
ls results/training_runs/chart_detector_v3/

# View training metrics
cat results/training_runs/chart_detector_v3/results.csv

# Test trained model
yolo predict model=results/training_runs/chart_detector_v3/weights/best.pt source=data/samples/

# Validate on test set
yolo val model=results/training_runs/chart_detector_v3/weights/best.pt data=data/training/dataset.yaml split=test
```

## 8. Session Notes

- Training started at ~2026-01-20
- **CUDA Enabled:** PyTorch 2.6.0+cu124 with RTX 3060 Laptop GPU (6GB)
- Estimated training time: **~30-60 minutes** (GPU vs 4-8 hours CPU)
- Model weights will be saved to `results/training_runs/chart_detector_v3_gpu/weights/`

### CUDA Setup Steps:
1. Uninstalled CPU-only PyTorch: `pip uninstall torch torchvision -y`
2. Installed CUDA version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
3. Fixed dataset.yaml: Changed `nc: 6` to `nc: 1` (single-class detection)
4. Cleared label cache files
5. Started training with `device=0` (GPU)
