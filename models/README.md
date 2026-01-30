# Models Directory

## Directory Structure

```
models/
├── weights/           # Trained model weights
│   ├── yolov8n.pt     # YOLOv8 nano base model (pretrained)
│   ├── yolo_chart_detector.pt  # Fine-tuned YOLO for chart detection [NEEDS RETRAIN]
│   ├── resnet18_chart_classifier_v2_best.pt  # ResNet-18 v2 (94.14% accuracy)
│   └── resnet18_v2_results.json  # Training results
├── onnx/              # ONNX exported models (for production)
├── evaluation/        # Evaluation results and visualizations
└── explainability/    # Grad-CAM and other explainability outputs
```

## Current Models

### ResNet-18 Chart Classifier v2

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.14%** |
| Best Val Accuracy | 94.80% |
| Training Time | 50:22 |
| Input Size | 224x224 grayscale (replicated to 3 channels) |
| Classes | 8 (area, bar, box, heatmap, histogram, line, pie, scatter) |
| Dataset | 32,445 preprocessed images |

**Per-class Accuracy:**
- pie: 98.8%
- bar: 95.3%
- heatmap: 94.9%
- line: 94.2%
- scatter: 93.3%
- histogram: 91.2%
- area: 90.5%
- box: 89.8%

### YOLO Chart Detector

| Metric | Value |
|--------|-------|
| Status | **NEEDS RETRAIN** |
| Base Model | YOLOv8 nano |
| Task | Binary detection (chart vs non-chart) |
| Current Dataset | ~3,800 images (old) |
| Target Dataset | 32,445 images (new) |

**TODO:**
1. Regenerate YOLO dataset from `classified_charts/`
2. Retrain with new data
3. Update `yolo_chart_detector.pt`

## Usage

### ResNet-18 Classifier

```python
from core_engine.stages.s3_extraction.classifier import ResNet18Classifier

classifier = ResNet18Classifier()
chart_type, confidence = classifier.predict("path/to/chart.png")
print(f"Type: {chart_type}, Confidence: {confidence:.2%}")
```

### YOLO Detector

```python
from ultralytics import YOLO

model = YOLO("models/weights/yolo_chart_detector.pt")
results = model.predict("path/to/document.png")
```

## Regenerating Models

### Export ResNet-18 to ONNX

```bash
.venv/Scripts/python.exe scripts/export_resnet18_onnx.py
```

### Retrain YOLO

```bash
# 1. Generate synthetic dataset
.venv/Scripts/python.exe scripts/generate_synthetic_dataset.py --num-samples 10000

# 2. Train YOLO
.venv/Scripts/python.exe scripts/train_yolo_chart_detector.py --epochs 100
```
