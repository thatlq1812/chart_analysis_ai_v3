# Stage 2: Detection & Localization

## 1. Architecture

### 1.1. Responsibility
Detect chart regions in document images using a trained YOLO model, crop individual charts, and record bounding box metadata for traceability.

### 1.2. Position in Pipeline
```
Stage1Output(List[CleanImage]) --> [Stage 2: Detection] --> Stage2Output(List[DetectedChart])
                                                                 |
                                                                 v
                                                           Stage 3: Extraction
```

### 1.3. Class Hierarchy
```
BaseStage[Stage1Output, Stage2Output]
  +-- Stage2Detection (336 lines)
      Config: DetectionConfig (Pydantic BaseModel)
      Model: Ultralytics YOLO (lazy-loaded)
```

## 2. Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `model_path` | `models/weights/chart_detector.pt` | Path to trained YOLO weights |
| `device` | `auto` | Inference device (auto/cpu/cuda/mps) |
| `conf_threshold` | 0.5 | Minimum detection confidence |
| `iou_threshold` | 0.45 | Non-Maximum Suppression IoU threshold |
| `imgsz` | 640 | Input image size for YOLO |
| `min_area_pixels` | 100 | Minimum detection area filter |
| `max_detections_per_image` | 50 | Max charts per page |
| `save_cropped_images` | True | Save cropped chart images |

Source: `config/models.yaml` under `yolo:` + `config/pipeline.yaml` under `detection:`.

## 3. Model: YOLOv8

### 3.1. Architecture
- **Model variant**: YOLOv8m (medium) -- balance of speed and accuracy
- **Task**: Binary object detection (single class: "chart")
- **Input**: 640x640 RGB, auto-padded and scaled
- **Output**: Bounding boxes with confidence scores

### 3.2. Training Configuration
From `config/yolo_chart_v3.yaml`:
- **Classes**: 1 (`chart`)
- **Strategy**: Detect first, classify separately (in Stage 3)
- **Rationale**: Single-class detection is simpler, faster, and more accurate; type classification uses specialized ResNet-18

### 3.3. Dataset
| Split | Images | Ratio |
| --- | --- | --- |
| Train | ~22,700 | 80% |
| Validation | ~4,870 | ~17% |
| Test | ~4,870 | ~17% |
| **Total** | **~32,440** | 100% |

Source: `data/yolo_chart_detection/` (6.53 GB, 64,893 files including labels)

### 3.4. Augmentation
Charts require **conservative augmentation** -- no flips, no rotation:
- HSV: `h=0.015, s=0.4, v=0.4`
- Translate: 0.1
- Scale: 0.3
- Mosaic: 0.5
- Flip: disabled (charts are orientation-sensitive)

## 4. Algorithm: Detection Pipeline

```
For each CleanImage in Stage1Output:
  1. Load image (PIL -> numpy)
  2. Run YOLO inference (conf=0.5, iou=0.45, imgsz=640)
  3. Extract detections from YOLO results
  4. Filter by min_area_pixels
  5. For each valid detection:
     a. Generate chart_id: {session}_p{page}_c{idx}
     b. Crop image region (bbox coordinates in original space)
     c. Save cropped PNG to {session}/cropped_charts/
     d. Create DetectedChart record
  6. Return Stage2Output with all detected charts
```

### 4.1. Non-Maximum Suppression
YOLO's built-in NMS with IoU threshold 0.45 removes duplicate detections of the same chart.

### 4.2. Multi-Chart Handling
A single page may contain multiple charts. Each detection gets a unique `chart_id` and is processed independently through Stages 3-5.

## 5. Results

| Metric | Value | Source |
| --- | --- | --- |
| mAP@0.5 | **93.5%** (>0.85 target) | WEEKLY_PROGRESS_20260204 |
| Inference speed | Real-time on GPU, ~100ms/image CPU | Ultralytics benchmark |
| Detection class | 1 (binary: chart vs non-chart) | yolo_chart_v3.yaml |

## 6. Lessons Learned

- **Single-class detection + separate classification** outperforms multi-class YOLO for charts. YOLO excels at localization; classification benefits from dedicated architecture (ResNet-18 achieves 94.14%).
- **Conservative augmentation** is critical: rotation/flip corrupts chart semantics.
- **Padding** (10px around bbox) prevents text clipping at chart edges.

## 7. Limitations

- Model trained on academic paper charts (arXiv) -- may underperform on business/presentation charts
- No chart segmentation (only bounding box) -- overlapping charts are not separated
- Single-scale inference (640px) -- very small charts in high-resolution documents may be missed
- YOLO model file required at inference time (~25 MB)
