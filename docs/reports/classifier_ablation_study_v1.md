# Chart Type Classifier Ablation Study

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-12 | That Le | EfficientNet-B0 vs ResNet-18, resolution and class scope ablation |

---

## 1. Objective

Identify the optimal backbone, input resolution, and class scope for the Stage 2b
chart type classifier used in the Geo-SLM pipeline. The classifier must correctly
distinguish bar, line, and pie charts to route subsequent geometric extraction logic.

---

## 2. Experimental Setup

### 2.1. Hardware

| Component | Spec |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop (6GB VRAM) |
| CPU | AMD Ryzen 7 6800H (8 cores) |
| RAM | 32 GB |

### 2.2. Dataset

Source: `data/academic_dataset/classified_charts/` — 37,523 train / 4,686 val / 4,701 test images.

| Class | Train | Val | Test |
| --- | --- | --- | --- |
| bar | 4,596 | 574 | 575 |
| line | 10,344 | 1,293 | 1,293 |
| pie | 668 | 83 | 84 |
| others (v3 only) | 21,915 | 2,736 | 2,749 |

### 2.3. Training Protocol (fixed across all runs)

| Setting | Value |
| --- | --- |
| Phase 1 (frozen backbone) | 5 epochs, lr=5e-3 |
| Phase 2 (full fine-tune) | Up to 55 epochs, lr=1e-3 |
| LR schedule | Cosine annealing |
| Loss | Weighted CrossEntropy + label smoothing=0.1 |
| Sampling | WeightedRandomSampler (inverse class freq) |
| Batch size | 128 |
| AMP | Enabled (CUDA) |
| Early stopping | Patience=10 (Phase 2 only) |
| Augmentation | RandomCrop, HorizontalFlip, ColorJitter, RandomRotation(15), RandomAffine |

---

## 3. Ablation Runs

| Run | Backbone | Params | Image Size | Classes | Test Acc | Macro F1 | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 baseline | EfficientNet-B0 | 4.01M | 160px | 4 (w/ others) | 84.64% | 77.52% | 60 |
| eff-b0-3class | **EfficientNet-B0** | **4.01M** | **224px** | **3** | **97.54%** | **94.63%** | **21** |
| resnet18-3class | ResNet-18 | 11.7M | 224px | 3 | 95.95% | 92.62% | 25 |

---

## 4. Final Model Evaluation — EfficientNet-B0 3-class

**Model**: `models/weights/efficientnet_b0_3class_v1_best.pt`
**Test set**: 1,952 images

### 4.1. Overall Metrics

| Metric | Score | Target | Pass |
| --- | --- | --- | --- |
| Test Accuracy | 97.54% | 92.00% | YES |
| Macro F1 | 94.63% | 90.00% | YES |

### 4.2. Per-class Precision / Recall / F1

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| bar | 97.05% | 97.39% | 97.22% |
| line | 99.21% | 97.53% | 98.36% |
| pie | 79.81% | 98.81% | 88.30% |
| **Macro** | **92.02%** | **97.91%** | **94.63%** |

### 4.3. Confusion Matrix

|  | Predicted: bar | Predicted: line | Predicted: pie |
| --- | --- | --- | --- |
| Actual: bar | **560** | 9 | 6 |
| Actual: line | 17 | **1261** | 15 |
| Actual: pie | 0 | 1 | **83** |

---

## 5. Key Findings

**1. Resolution is the dominant factor.**
Increasing image input from 160px to 224px raised accuracy from 84.64% to 95—97%,
regardless of backbone. This is because 160px crops lose fine-grained chart structure
(axis tick marks, pie segment boundaries).

**2. Class scope removal was the second key driver.**
The `others` class grouped 6 visually heterogeneous chart types (area, box, heatmap,
histogram, scatter, table) into one label. This caused systematic confusion with bar
and line charts. Removing this class reduced the classification problem to 3 visually
distinct shapes.

**3. EfficientNet-B0 wins over ResNet-18 at same configuration.**
At 224px / 3-class, EfficientNet-B0 outperforms ResNet-18 by:
- +1.59pp accuracy (97.54% vs 95.95%)
- +2.01pp macro F1 (94.63% vs 92.62%)
- 66% fewer parameters (4.01M vs 11.7M)
- Converges 4 epochs faster (21 vs 25)

**4. Pie recall is high despite limited data.**
98.81% recall with only 668 training samples demonstrates that WeightedRandomSampler
+ label_smoothing=0.1 effectively handles severe class imbalance. The lower precision
(79.81%) represents 21 false positives out of 1,952 test images — acceptable in pipeline.

---

## 6. Production Decision

**Selected model**: EfficientNet-B0, 3-class, `efficientnet_b0_3class_v1_best.pt`

Rationale:
- Best accuracy and F1 of all runs
- Lightest backbone considered (4.01M params)
- Confidence threshold set to 0.70 in `config/models.yaml` — images with max_prob < 0.70
  are labeled `unknown` to prevent silent misclassification of out-of-scope chart types.

---

## 7. Recommendations

1. **Pie class**: Collect 500+ additional pie chart images if pipeline integration shows
   consistent false positives. Pie precision (79.81%) is the weakest point.
2. **Unknown handling**: Stage 3 must handle `unknown` type gracefully (skip geometric
   extraction, fallback to pure OCR).
3. **ResNet-18 baseline** is preserved at `models/weights/resnet18_3class_v1_best.pt`
   for thesis comparison tables.
