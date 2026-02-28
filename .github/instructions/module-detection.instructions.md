---
applyTo: 'src/core_engine/stages/s2_detection*,config/yolo_chart_v3.yaml,models/weights/chart_detector*,data/yolo_chart_detection/**'
---

# MODULE INSTRUCTIONS - Chart Detection (Stage 2)

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-02-28 | That Le | YOLO-based chart detection pipeline |

---

## 1. Overview

**Stage 2 Detection** locates chart regions within document pages using YOLOv8. Takes clean page images from Stage 1, outputs cropped chart images with bounding box coordinates.

**Key Files:**
- `src/core_engine/stages/s2_detection.py` (336 lines) - Single-file stage implementation
- `config/yolo_chart_v3.yaml` - YOLO dataset configuration
- `config/pipeline.yaml` - Stage-level settings (confidence, padding, max detections)
- `models/weights/chart_detector.pt` - Trained YOLO weights

---

## 2. Detection Model

### 2.1. Architecture

| Property | Value |
| --- | --- |
| Model | YOLOv8 nano (`yolov8n.pt` base) |
| Task | Binary detection (chart vs non-chart) |
| Input size | 640x640 |
| Classes | 1 (`chart`) |
| Conf threshold | 0.5 |
| IoU threshold | 0.45 |
| Max detections/page | 10 |

### 2.2. Dataset

| Split | Count | Source |
| --- | --- | --- |
| Train | ~26,000 (target) | ChartQA + ArXiv papers + synthetic |
| Val | ~3,200 | Same distribution |
| Test | ~3,200 | Held-out papers |

**Current status:** Dataset has ~3,800 images (old). Needs expansion to 32,445 for full training.

### 2.3. Training

```bash
# Train YOLO from config
yolo detect train model=yolov8n.pt data=config/yolo_chart_v3.yaml \
    epochs=100 imgsz=640 batch=16 device=0
```

---

## 3. Stage Implementation

### 3.1. Key Classes

| Class | Purpose |
| --- | --- |
| `DetectionConfig(BaseModel)` | Pydantic config for model path, thresholds, output settings |
| `Stage2Detection(BaseStage)` | Orchestrator: load model → run inference → filter → crop → save |

### 3.2. Processing Flow

```
Input:  List[CleanPageImage] from Stage 1
  ↓
  1. Load YOLO model (lazy, cached)
  ↓
  2. Run inference on each page image
  ↓
  3. Filter detections by confidence > threshold
  ↓
  4. Filter by min_area_pixels (remove tiny detections)
  ↓
  5. Apply NMS (non-max suppression) with IoU threshold
  ↓
  6. Crop detected regions (with padding)
  ↓
  7. Save cropped images + bounding box metadata
  ↓
Output: List[DetectedChart] with bounding boxes + cropped images
```

### 3.3. Configuration

From `config/pipeline.yaml`:
```yaml
detection:
  confidence_threshold: 0.5
  multi_chart: true
  max_charts_per_page: 10
  crop_padding: 10        # pixels around detection box
```

---

## 4. Rules

1. **Model loading** is lazy -- YOLO model loads on first `process()` call
2. **GPU** auto-detection: use CUDA if available, else CPU
3. **Detection caching** is enabled -- cached results invalidated when model changes
4. **Multi-chart** support: a single page may contain multiple charts
5. **Bounding boxes** stored in YOLO format (x_center, y_center, width, height, normalized)
6. **Crop padding** adds pixels around detection to avoid cutting chart edges
7. **Never** modify the YOLO config file (`yolo_chart_v3.yaml`) for runtime changes -- use `pipeline.yaml` overrides
8. **Model weights** are NOT in git -- download from model registry or train locally

---

## 5. Testing

```bash
# Run Stage 2 tests
.venv/Scripts/python.exe -m pytest tests/ -k "stage2 or detection" -v
```

Test fixtures should use small synthetic images (no real model loading for unit tests). Integration tests with real YOLO model use `@pytest.mark.integration`.
