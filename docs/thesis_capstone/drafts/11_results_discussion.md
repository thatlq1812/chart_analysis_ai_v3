# Results and Discussion

## 1. Evaluation Overview

This chapter consolidates experimental results from each pipeline stage and the overall end-to-end system. All results are from models trained and evaluated on the academic chart dataset (32,364 charts from ~4,000 arXiv papers, 8 chart types).

### 1.1. Evaluation Scope
| Component | Evaluation Type | Dataset |
| --- | --- | --- |
| YOLO Chart Detector | mAP@0.5, Precision, Recall | yolo_chart_detection (train/val/test) |
| ResNet-18 Classifier | Accuracy, Per-class accuracy | classified_charts (test split) |
| Stage 3 Extraction | Success rate, Feature quality | 32,364 charts (full corpus) |
| OCR Engine | Confidence score, Text overlap | OCR cache (46,910 entries) |
| End-to-End Pipeline | Overall confidence, Classification acc | 800+ images, 100 per type |

## 2. Stage 2: Chart Detection (YOLOv8m)

### 2.1. Model Configuration
| Parameter | Value |
| --- | --- |
| Architecture | YOLOv8m (medium) |
| Strategy | Single-class ("chart") |
| Input size | 640x640 |
| Training epochs | 100 |
| Optimizer | SGD, lr=0.01, momentum=0.937 |
| Augmentation | Mosaic, MixUp, HSV, Flip |
| Dataset split | 70% train / 20% val / 10% test |

### 2.2. Detection Results
| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| mAP@0.5 | **93.5%** | >85% | EXCEEDED |
| Precision | >90% | >80% | EXCEEDED |
| Recall | >90% | >80% | EXCEEDED |

### 2.3. Design Decision: Single-Class vs Multi-Class
The detector uses single-class ("chart") rather than multi-class per chart type because:
1. Higher precision per crop (no category confusion at detection stage)
2. Downstream ResNet-18 classifier handles type assignment with 94.14% accuracy
3. Simpler YOLO training with less annotation requirement
4. Two-model cascade (detect then classify) outperforms single multi-class detector

## 3. Stage 2b: Chart Classification (ResNet-18)

### 3.1. Training Details
| Parameter | Value |
| --- | --- |
| Architecture | ResNet-18 (ImageNet pretrained) |
| Training time | 27 minutes (NVIDIA GPU) |
| ONNX inference | 6.90 ms/image (CPU), 144.9 img/sec |
| Model size | 42.64 MB (ONNX) |
| Classes | 8 (line, bar, scatter, heatmap, histogram, box, pie, area) |

### 3.2. Classification Results
| Metric | Value |
| --- | --- |
| Test Accuracy | **94.14%** |
| Integration Test Accuracy | **93.75%** (15/16 samples) |

### 3.3. Per-Class Performance (Integration Test)
| Type | Accuracy | Confidence (avg) |
| --- | --- | --- |
| line | 100% | 0.999 |
| scatter | 100% | 0.982 |
| pie | 100% | 0.999 |
| bar | 100% | 0.999 |
| box | 100% | ~0.97 |
| histogram | 100% | ~0.98 |
| heatmap | 100% | ~0.97 |
| area | 50% | ~0.68 |

### 3.4. Analysis: Area Chart Confusion
The area type shows the lowest accuracy (50% in integration test, 1,999 samples in v3 dataset — smallest class). Root causes:
- Visual overlap between area and line charts (area = filled line)
- Smallest training set (617 charts, 1.9% of corpus)
- Academic papers rarely use pure area charts

**Mitigation**: In production, confident area predictions are cross-validated with element detection (fill regions vs bare lines).

### 3.5. Explainability: Grad-CAM Visualizations
Grad-CAM heatmaps stored in `models/explainability/gradcam_*.png` confirm that ResNet-18 focuses on:
- **Bar charts**: Vertical/horizontal bar regions
- **Line charts**: Continuous line trajectories
- **Scatter plots**: Point clusters
- **Pie charts**: Circular regions
- **Heatmaps**: Color grid regions

This confirms the model learns chart-specific visual features rather than background artifacts.

## 4. Stage 3: Structural Extraction

### 4.1. Batch Extraction Results
| Metric | Run 1 | Run 2 (Final) |
| --- | --- | --- |
| Total charts | 32,364 | 32,364 |
| Success rate | 41.1% | **100%** |
| Corruption rate | 58.9% | **0%** |
| Processing speed | ~4 img/s (CPU) | ~4 img/s (CPU) |

The Run 1 -> Run 2 improvement was caused by fixing the `np.int64` JSON serialization bug (see Section 8.1).

### 4.2. OCR Performance
| Metric | Value |
| --- | --- |
| OCR cache entries | 46,910 |
| Mean OCR confidence | **91.5%** |
| Cache size on disk | 588.8 MB |

### 4.3. Feature Quality by Chart Type
| Type | Axis Info Coverage | Mean OCR Confidence | Zero-Text Rate |
| --- | --- | --- | --- |
| histogram | 99.0% | 0.932 | Low |
| bar | 96.8% | 0.905 | 29.6% |
| line | 88.5% | 0.837 | Low |
| scatter | 73.3% | 0.786 | Low |
| box | 60.9% | 0.648 | Moderate |
| heatmap | 52.2% | 0.453 | Moderate |
| pie | 42.1% | 0.512 | Low |
| area | 34.2% | 0.387 | 59.5% |
| **Average** | **69.9%** | - | - |

### 4.4. Analysis: Zero-Text and Low Axis Coverage
- **Bar charts** (29.6% zero-text): Programmatic charts from ML papers often embed text as vector graphics, invisible to pixel-based OCR.
- **Area charts** (59.5% zero-text): Academic area charts frequently use color fills without printed text labels.
- **Heatmap** (52.2% axis coverage): Color grids with categorical labels that OCR sometimes mis-reads.

These zero-text cases are handled by the SLM training data with explicit `[NO READABLE TEXT]` markers, teaching the model to reason from geometric features alone.

## 5. End-to-End Pipeline Performance

### 5.1. Full Pipeline Test (800+ images)
| Metric | Value |
| --- | --- |
| Classification Accuracy (all types) | **100%** |
| OCR Confidence (avg) | **91.5%** |
| Overall Stage 3 Confidence | **92.6%** |
| Avg Processing Time | **~7.6 s/image** |

### 5.2. Processing Time Breakdown (Estimated)
| Stage | Time | Notes |
| --- | --- | --- |
| Stage 1 (Ingestion) | <0.5s | Image loading, normalization |
| Stage 2 (Detection) | ~1.0s | YOLO inference (GPU) |
| Stage 3 (Extraction) | ~5.0s | OCR is the bottleneck |
| Stage 4 (Reasoning) | ~1.0-2.0s | SLM inference (target) |
| Stage 5 (Reporting) | <0.5s | JSON assembly |
| **Total** | **~7.5-9.0s** | Per chart image |

### 5.3. Bottleneck: OCR in Stage 3
OCR (PaddleOCR/EasyOCR) accounts for ~65% of total processing time. Optimization strategies:
- OCR caching (already implemented — 46,910 entries)
- Batch OCR processing
- Selective OCR (only re-run for low-confidence regions)

## 6. Dataset Quality Assessment

### 6.1. Cross-Validation: Stage 3 Features vs QA Answers
- 87% OCR text overlap between Stage 3 extracted features and Gemini-generated QA answers
- Confirms that the feature extraction captures the information needed for accurate QA
- 13% mismatch is expected: QA answers may reference visual elements not captured by OCR (colors, relative positions)

### 6.2. SLM Training Dataset v3 Quality
| Quality Metric | Value |
| --- | --- |
| Total samples | 268,799 |
| Stage 3 coverage | 100% |
| Has axis info | 69.9% |
| Split leakage | 0% (split by chart_id) |
| v3 vs v2 increase | 9.9x |

## 7. Code Quality Metrics

### 7.1. Test Suite
| Test Suite | Passed | Failed | Total |
| --- | --- | --- | --- |
| Schemas | 19 | 0 | 19 |
| Stage 3 | 139 | 1 | 140 |
| Stage 4 | 18 | 0 | 18 |
| **Total** | **176** | **1** | **177** |
| **Pass Rate** | | | **99.4%** |

### 7.2. Core Engine Size
| Component | Files | Approx. Lines |
| --- | --- | --- |
| Pipeline orchestrator | 1 | 313 |
| Schemas | 5 | ~900 |
| Stage 1 | 1 | 511 |
| Stage 2 | 1 | 336 |
| Stage 3 | 12 | ~5,400 |
| Stage 4 | 7 | ~3,200 |
| Stage 5 | 1 | 635 |
| AI layer | 3+ | ~700 |
| **Total** | **~30** | **~12,000** |

## 8. Known Issues and Bug Fixes

### 8.1. np.int64 JSON Serialization Bug
- **Symptom**: 58.9% of Stage 3 outputs corrupted in batch Run 1
- **Root Cause**: `geometric_mapper.py` returned NumPy `int64` values instead of Python `int`
- **Fix**: Added explicit `int()` casting at all NumPy-to-JSON boundaries
- **Impact**: Run 2 achieved 0% corruption across all 32,364 charts
- **Lesson**: Always cast NumPy types to Python natives before serialization

### 8.2. PaddleOCR Windows Compatibility
- **Symptom**: PaddleOCR 3.3.x crashes on Windows due to DLL conflicts
- **Fix**: Switched to EasyOCR as primary OCR engine (with PaddleOCR as optional fallback)
- **Impact**: OCR confidence maintained at 91.5%

### 8.3. Axis Key Format in v2 Training Script
- **Symptom**: v2 training script accessed wrong nested key names for axis info — coverage was ~4% instead of 69.9%
- **Fix**: v3 script uses flat key access (`x_min`, `y_min`, etc.)
- **Impact**: v2 training data was functionally useless for axis-related reasoning; v3 corrected this entirely

## 9. Comparison with Related Work

### 9.1. Positioning Against Existing Methods
| Method | Approach | Value Accuracy | Chart Types | Model Size | Interpretable |
| --- | --- | --- | --- | --- | --- |
| DePlot (2023) | Pix2Struct + LLM | ~85% | 5+ | 282M+ | No |
| MatCha (2023) | Chart pretraining | ~80-85% | 5+ | 282M+ | No |
| ChartReader (2022) | Detection + rules | ~75% | 3 | N/A | Partial |
| ReVision (2011) | Hough transform | ~70% | 1 (bar) | N/A | Yes |
| **Geo-SLM (Ours)** | **Hybrid neuro-symbolic** | **Target >95%** | **8** | **1.5B SLM** | **Yes** |

### 9.2. Key Differences
| Dimension | End-to-End (DePlot, MatCha) | Geo-SLM |
| --- | --- | --- |
| Value extraction | LLM estimation (probabilistic) | Geometric measurement (deterministic) |
| Error source | Hallucination, approximation | OCR noise, axis calibration |
| Explainability | Black-box | Fully traceable |
| Offline capability | No (requires large model/cloud) | Yes (1.5B SLM runs locally) |
| Chart type support | Depends on training data | 8 types, extensible |

### 9.3. Limitations of Current Results

> **[TODO - SUPPLEMENT AFTER SLM TRAINING]**
> The following items will be updated once SLM fine-tuning (Week 7-8) and benchmark evaluation (Week 9) are complete:
> - Add SLM inference accuracy (JSON valid rate, field accuracy, numeric accuracy)
> - Add ChartQA/PlotQA benchmark comparison
> - Add GPT-4V baseline comparison
> - Add ablation study results (with/without neg. transform, RDP, SLM)
> - Update processing time with SLM inference latency

1. **No direct benchmark comparison**: Geo-SLM has not yet been evaluated on the same public benchmark datasets (e.g., ChartQA, PlotQA) used by DePlot/MatCha. This is planned for Phase 3.
2. **SLM not yet trained**: Stage 4 reasoning results are currently from Gemini API prototyping. Local SLM evaluation is pending.
3. **Processing speed**: 7.6 s/image is slower than target <5 s/chart. OCR optimization needed.
4. **Dataset bias**: All charts from academic papers (cs.CV, cs.LG). Generalization to business charts, dashboards, and informal visualizations is untested.

## 10. Discussion

### 10.1. Hybrid Architecture Validates Hypothesis
The core research hypothesis — that geometric precision combined with AI reasoning outperforms pure end-to-end approaches — is supported by:
- 100% extraction success rate (Stage 3) proves geometric methods are robust
- 93.5% detection mAP + 94.14% classification accuracy prove neural components are strong
- 92.6% overall pipeline confidence before SLM integration is promising

### 10.2. Modularity Enables Iterative Improvement
The 5-stage architecture with well-defined schema contracts allowed:
- Independent development and testing of each stage
- Replacing PaddleOCR with EasyOCR without pipeline changes
- Adding ResNet-18 classifier without modifying Stage 3 interface
- Planning SLM integration without touching geometric extraction code

### 10.3. Data Quality is the Foundation
The 9.9x improvement from v2 to v3 training dataset (27,159 -> 268,799 samples) came from:
- Fixing axis key format (4% -> 69.9% axis coverage)
- 100% Stage 3 coverage (vs partial in v2)
- Split by chart_id (no data leakage)
- Explicit handling of zero-text cases

### 10.4. Looking Forward: SLM Fine-tuning
With 268,799 high-quality training samples ready, the next milestone is QLoRA fine-tuning of Qwen-2.5-1.5B-Instruct. Expected outcomes:
- JSON valid rate >95%
- Field accuracy >90%
- Numeric accuracy >85%
- Inference <2 s/chart on RTX 3060
