# Early Stopping Strategy for YOLO Training

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2025-01-20 | That Le | Early stopping configuration guide |

## 1. Overview

Early Stopping is a regularization technique that stops training when the model performance stops improving on a validation set. This prevents overfitting and saves training time.

## 2. How Early Stopping Works in YOLO

```
Epoch 1:  mAP50 = 0.65   [best = 0.65] patience_counter = 0
Epoch 2:  mAP50 = 0.72   [best = 0.72] patience_counter = 0  (improved)
Epoch 3:  mAP50 = 0.78   [best = 0.78] patience_counter = 0  (improved)
Epoch 4:  mAP50 = 0.77   [best = 0.78] patience_counter = 1  (no improvement)
Epoch 5:  mAP50 = 0.76   [best = 0.78] patience_counter = 2  (no improvement)
...
Epoch 18: mAP50 = 0.75   [best = 0.78] patience_counter = 15 (no improvement)
                                                               
STOP! No improvement for 15 epochs (patience exceeded)
```

## 3. Key Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `patience` | 100 | Epochs to wait without improvement before stopping |
| `epochs` | 100 | Maximum number of training epochs |

**Recommendation for Chart Detection:**

| GPU VRAM | Dataset Size | Patience | Max Epochs | Rationale |
| --- | --- | --- | --- | --- |
| 6GB (RTX 3060) | 10,000 | 15 | 100 | Conservative to prevent OOM during long runs |
| 8GB (RTX 3070) | 15,000 | 20 | 150 | More patience for larger datasets |
| 16GB+ | 20,000+ | 25 | 200 | Can afford longer training |

## 4. Usage

### 4.1. Command Line

```bash
# Basic usage with early stopping
yolo train model=yolov8n.pt data=dataset.yaml epochs=100 patience=15 device=0

# Full command with recommended settings
yolo train \
    model=yolov8n.pt \
    data=data/training_synthetic/dataset.yaml \
    epochs=100 \
    patience=15 \
    batch=16 \
    imgsz=640 \
    device=0 \
    project=results/training_runs \
    name=chart_detector_v1
```

### 4.2. Using Training Script

```bash
# Use the provided training script
python scripts/train_detector_with_early_stopping.py \
    --epochs 100 \
    --patience 15 \
    --batch 16 \
    --device 0

# Resume from checkpoint
python scripts/train_detector_with_early_stopping.py --resume
```

### 4.3. Python API

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data/training_synthetic/dataset.yaml",
    epochs=100,
    patience=15,      # Early stopping patience
    batch=16,
    imgsz=640,
    device=0,
    project="results/training_runs",
    name="chart_detector_v1",
)
```

## 5. What Gets Saved

When training completes (either by early stopping or reaching max epochs):

```
results/training_runs/chart_detector_v1/
├── weights/
│   ├── best.pt          # Best model (highest mAP50)
│   └── last.pt          # Final model (for resume)
├── results.png          # Training metrics plot
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
└── args.yaml            # Training arguments (reproducibility)
```

## 6. Monitoring Metrics

Early stopping monitors `fitness`, which is a combination of metrics:

```
fitness = 0.1 * mAP50 + 0.9 * mAP50-95
```

For single-class chart detection, focus on:

| Metric | Target | Meaning |
| --- | --- | --- |
| mAP50 | > 0.85 | Detection accuracy at 50% IoU |
| mAP50-95 | > 0.60 | Stricter detection accuracy |
| Precision | > 0.85 | Fewer false positives |
| Recall | > 0.85 | Fewer missed charts |

## 7. Troubleshooting

### 7.1. Training Stops Too Early

**Symptom:** Training stops at epoch 20-30 with low mAP

**Solutions:**
1. Increase patience: `patience=25`
2. Check if dataset is too small
3. Verify labels are correct

### 7.2. Training Never Stops (Overfitting)

**Symptom:** Training loss decreases but validation mAP stops improving

**Solutions:**
1. Decrease patience: `patience=10`
2. Add more augmentation
3. Use smaller model (yolov8n instead of yolov8s)

### 7.3. Oscillating mAP

**Symptom:** mAP jumps up and down between epochs

**Solutions:**
1. Reduce learning rate: `lr0=0.005`
2. Increase batch size if VRAM allows
3. Enable EMA (default in YOLO)

## 8. Best Practices

1. **Always use early stopping** - Set `patience=15` minimum
2. **Monitor TensorBoard** - Watch for validation plateau
3. **Save both weights** - `best.pt` for deployment, `last.pt` for resume
4. **Document experiments** - Keep `args.yaml` for reproducibility
5. **Validate before deploy** - Run `yolo val` on held-out test set

## 9. Example Training Session

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/100      4.32G     1.4521     0.9876     1.1234         45        640
          mAP50: 0.4523  mAP50-95: 0.2341
2/100      4.32G     1.2345     0.8765     1.0234         48        640
          mAP50: 0.5678  mAP50-95: 0.3456  [IMPROVED]
...
35/100     4.32G     0.5678     0.3456     0.6789         51        640
          mAP50: 0.8923  mAP50-95: 0.6234  [IMPROVED]
36/100     4.32G     0.5567     0.3345     0.6678         49        640
          mAP50: 0.8901  mAP50-95: 0.6198  [no improvement, patience: 1/15]
...
50/100     4.32G     0.5234     0.3123     0.6456         50        640
          mAP50: 0.8845  mAP50-95: 0.6123  [no improvement, patience: 15/15]

EarlyStopping: Training stopped early as no improvement observed in last 15 epochs.
Best results observed at epoch 35.

Results saved to results/training_runs/chart_detector_v1
```

## 10. Related Files

- [train_detector_with_early_stopping.py](../../scripts/train_detector_with_early_stopping.py) - Training script
- [generate_synthetic_dataset.py](../../scripts/generate_synthetic_dataset.py) - Dataset generation
- [page_synthesizer.py](../../tools/data_factory/services/page_synthesizer.py) - Copy-paste augmentation
