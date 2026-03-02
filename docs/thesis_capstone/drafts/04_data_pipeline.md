# Data Pipeline & SLM Training Dataset

## 1. Overview

The data pipeline transforms raw academic papers into a structured SLM training dataset. This is a **separate batch workflow** from the real-time 5-stage pipeline -- it uses Stages 1-3 for feature extraction, then external tools for QA generation and dataset assembly.

```
ArXiv Papers (~4,000)
  |
  v
[PDF Mining] -> 150,000 page images
  |
  v
[YOLO Detection] -> 70,000 candidate charts
  |
  v
[Classification + Filtering] -> 46,910 verified charts
  |
  v
[Stage 3 Extraction] -> 32,364 feature JSONs
  |
  v
[QA Generation (Gemini)] -> 291,000 QA pairs
  |
  v
[Dataset Assembly] -> 268,799 SLM training samples
```

## 2. Data Collection

### 2.1. Source: ArXiv Academic Papers
- ~4,000 papers from cs.CV, cs.LG, cs.AI categories
- Focus on papers containing charts/visualizations
- Downloaded via ArXiv API with rate limiting (~3s between requests)
- Resume capability via `data/search_cache/arxiv_progress.json`

### 2.2. Page Image Extraction
- PyMuPDF renders each PDF page at 150 DPI
- ~150,000 page images generated
- Stored in `data/academic_dataset/images/`

### 2.3. Chart Detection (YOLO)
- YOLOv8m with confidence threshold 0.5
- ~70,000 candidate chart regions detected
- Cropped and saved to `data/academic_dataset/detected_charts/`

### 2.4. Chart Classification & Filtering
| Type | Count | Share |
| --- | --- | --- |
| line | 10,036 | 31.0% |
| bar | 9,086 | 28.1% |
| box | 4,867 | 15.0% |
| scatter | 2,802 | 8.7% |
| pie | 2,421 | 7.5% |
| histogram | 2,060 | 6.4% |
| heatmap | 680 | 2.1% |
| area | 412 | 1.3% |
| **Total** | **32,364** | **100%** |

Stored in `data/academic_dataset/classified_charts/` organized by type.

## 3. Stage 3 Batch Extraction

### 3.1. Process
- `scripts/pipeline/batch_stage3_parallel.py` processes all 32,364 charts
- Parallel execution with worker pool
- Atomic file writes (write to temp, then rename)
- Pure CPU extraction (no GPU required for this stage)
- OCR cache: 46,910 entries (~589 MB) in `data/cache/ocr_cache.json`

### 3.2. Results
| Metric | Run 1 | Run 2 (Final) |
| --- | --- | --- |
| Total charts | 32,364 | 32,364 |
| Success rate | 41.1% | **100%** |
| Corruption rate | 58.9% | **0%** |
| Processing speed | ~4 img/s (CPU) | ~4 img/s (CPU) |

### 3.3. Critical Bug Fixed
**Run 1 failure**: `np.int64` values from geometric_mapper.py caused JSON serialization errors in 58.9% of outputs.

**Fix**: Added explicit `float()` casting in geometric_mapper.py before JSON serialization. Run 2 achieved 0% corruption.

### 3.4. Feature Quality
| Type | Axis Info Coverage | Mean OCR Confidence |
| --- | --- | --- |
| histogram | 99.0% | 0.932 |
| bar | 96.8% | 0.905 |
| line | 88.5% | 0.837 |
| scatter | 73.3% | 0.786 |
| box | 60.9% | 0.648 |
| heatmap | 52.2% | 0.453 |
| pie | 42.1% | 0.512 |
| area | 34.2% | 0.387 |
| **Average** | **69.9%** | - |

## 4. QA Generation

### 4.1. Tool
- `tools/data_factory/` with Gemini 2.0 Flash API
- 10 questions per chart across multiple depth levels

### 4.2. Question Distribution
| Depth | Types | Target % |
| --- | --- | --- |
| Shallow | structural, extraction, counting | 25% |
| Intermediate | comparison, trend, range | 35% |
| Deep | interpolation, threshold, multi_hop | 30% |
| Conceptual | why_reasoning, caption_aware | 10% |

### 4.3. Output
- 32,445 QA files generated
- ~291,000 individual QA pairs
- 87% OCR text overlap confirmed between Stage 3 features and QA answers

## 5. SLM Training Dataset v3

### 5.1. Assembly
Built by `scripts/training/prepare_slm_training_v3.py`:
- Merges Stage 3 features + QA pairs
- Formats as ChatML conversations
- Splits by `chart_id` (no data leakage between train/val/test)

### 5.2. Dataset Statistics
| Split | Samples | Ratio |
| --- | --- | --- |
| Train | 228,494 | 85% |
| Validation | 26,888 | 10% |
| Test | 13,417 | 5% |
| **Total** | **268,799** | 100% |

### 5.3. Per-Type Distribution
| Type | Samples | Source Charts |
| --- | --- | --- |
| line | 108,419 | 10,036 |
| scatter | 52,163 | 2,802 |
| bar | 47,330 | 9,086 |
| heatmap | 33,373 | 680 |
| box | 13,948 | 4,867 |
| pie | 7,408 | 2,421 |
| histogram | 4,159 | 2,060 |
| area | 1,999 | 412 |

### 5.4. v3 Improvements Over v2
| Issue | v2 | v3 |
| --- | --- | --- |
| Axis key names | Wrong (`x_min` as string) | Fixed (proper float values) |
| OCR text format | Raw list | Role-grouped (title, xlabel, ylabel, etc.) |
| Element data | Raw count | Type breakdown (bars: N, markers: M) |
| Sample count | 27,159 | **268,799** (9.9x increase) |
| Split method | Random | By chart_id (no leakage) |

### 5.5. ChatML Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a chart analysis assistant..."},
    {"role": "user", "content": "Chart type: bar\nAxis: x=[0, 10], y=[0, 100]\nOCR texts: ...\nQuestion: What is the highest value?"},
    {"role": "assistant", "content": "The highest value is 95, corresponding to..."}
  ]
}
```

## 6. SLM Training Plan (In Progress)

### 6.1. Configuration
| Parameter | Value |
| --- | --- |
| Base model | Qwen2.5-1.5B-Instruct (PRIMARY) |
| Alternative | Llama-3.2-1B-Instruct |
| Method | QLoRA (4-bit NF4 + LoRA rank 16) |
| VRAM requirement | ~4 GB (fits RTX 3060 6GB) |
| LoRA params | 11.27M trainable (0.9% of total) |
| Epochs | 3 (3 sessions x 1 epoch x ~14 hours) |
| Batch size | 4 (effective 16 with accumulation) |
| Learning rate | 2e-4, cosine schedule |
| Max sequence length | 512 tokens |

### 6.2. Curriculum Learning (4 stages)
| Stage | Focus | Difficulty |
| --- | --- | --- |
| 1 | Chart type + axis labels | Easy |
| 2 | Value extraction + OCR correction | Medium |
| 3 | Trends + comparisons (multi-series) | Hard |
| 4 | Noisy OCR + complex layouts | Very Hard |

### 6.3. Evaluation Targets
| Metric | Target |
| --- | --- |
| JSON Valid Rate | >95% |
| Field Accuracy | >90% |
| Numeric Accuracy | >85% |
| Latency | <2s per chart |

## 7. Lessons Learned

1. **Atomic file writes** prevented data corruption during long-running batch jobs (crash recovery).
2. **np.int64 serialization bug** was the #1 data quality issue -- always cast NumPy types to Python natives before JSON.
3. **Chart_id-based splitting** prevents data leakage in SLM training (same chart's questions must be in same split).
4. **QA cross-validation** (87% OCR overlap) confirms Stage 3 extraction quality.
5. **Imbalanced dataset**: area charts have 1,999 samples vs line charts 108,419 -- curriculum learning and oversampling planned.

## 8. Storage Summary

| Directory | Size | Files | Notes |
| --- | --- | --- | --- |
| data/academic_dataset/images/ | 49.96 GB | 218,887 | Raw page images |
| data/academic_dataset/detected_charts/ | 11.68 GB | 50,865 | YOLO crops |
| data/academic_dataset/classified_charts/ | 11.23 GB | 46,911 | Type-sorted |
| data/academic_dataset/stage3_features/ | 927.8 MB | 32,364 | Feature JSONs |
| data/cache/ocr_cache.json | 588.8 MB | 1 | OCR results |
| data/slm_training_v3/ | 330.6 MB | 4 | Final dataset |
| **Total data/** | **120.80 GB** | **468,320** | |
