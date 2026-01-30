# Scripts Directory

Utility scripts for training, evaluation, and data processing.

## Available Scripts

### Model Training

| Script | Purpose | Status |
|--------|---------|--------|
| `train_resnet18_v2.py` | Train ResNet-18 chart classifier | Production |
| `train_yolo.py` | Generic YOLO training script | Ready |
| `train_yolo_chart_detector.py` | YOLO chart detection training | Needs update |

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
| `extract_backgrounds.py` | Extract text-only pages for YOLO training | Ready |
| `generate_synthetic_dataset.py` | Generate synthetic YOLO training data | Ready |
| `verify_qa_dataset.py` | Verify QA dataset integrity | Ready |

### Testing

| Script | Purpose | Status |
|--------|---------|--------|
| `test_element_detector.py` | Test element detection module | Ready |
| `test_stage4.py` | Test Stage 4 reasoning | Ready |

## Usage Examples

### Train ResNet-18 v2

```bash
# Full training with preprocessing
.venv/Scripts/python.exe scripts/train_resnet18_v2.py \
    --epochs 50 \
    --batch-size 128 \
    --preprocess
```

### Generate YOLO Dataset

```bash
# Extract backgrounds from PDFs
.venv/Scripts/python.exe scripts/extract_backgrounds.py \
    --pdf-dir data/raw_pdfs \
    --output-dir data/backgrounds \
    --max-pages 5000

# Generate synthetic dataset
.venv/Scripts/python.exe scripts/generate_synthetic_dataset.py \
    --num-samples 10000 \
    --output-dir data/yolo_chart_detection
```

### Train YOLO

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

## Deleted Scripts (Obsolete)

- `merge_chartqa_labels.py` - Replaced by Gemini batch processing
- `prepare_training_data.py` - Replaced by `train_resnet18_v2.py` preprocessing
- `train_classifier.py` - Replaced by `train_resnet18_v2.py`
- `train_resnet18_classifier.py` - Replaced by v2 version
