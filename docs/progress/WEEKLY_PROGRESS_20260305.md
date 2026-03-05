# Weekly Progress Report - 2026-03-05

| Property | Value |
| --- | --- |
| **Week** | 2026-03-03 to 2026-03-05 |
| **Author** | That Le |
| **Focus** | Stage 3 Benchmark Evaluation + Gemini Vision Ground Truth |

---

## Summary

This week focused on establishing a rigorous benchmark evaluation for Stage 3 (Structural Extraction) using 50 stratified real-world charts with Gemini Vision-generated ground truth annotations. The benchmark validates the strengths and weaknesses identified in the earlier code audit, providing quantitative evidence for the thesis.

---

## Completed Tasks

### 1. Stage 3 Bug Fixes (4 bugs) - COMPLETED

| Bug | Severity | File | Fix |
| --- | --- | --- | --- |
| BUG-1: Pie slice detection dead code | CRITICAL | element_detector.py | Wired `_detect_pie_slices_by_kmeans()` |
| BUG-2: Missing chart_type param | MODERATE | element_detector.py | Pass chart_type to hybrid detection |
| BUG-3: ExtractionConfidence always 0.0 | CRITICAL | s3_extraction.py | Use `compute_overall()` |
| BUG-4: Skeletonizer overflow | MINOR | skeletonizer.py | float64 for stroke width |

Test suite: 294 -> 300 tests (6 new pie slice tests), all passing.

### 2. Benchmark Infrastructure - COMPLETED

| Component | Script | Purpose |
| --- | --- | --- |
| Stratified sampler | `scripts/evaluation/benchmark/stratified_sampler.py` | 50 charts (10 bar, 10 line, 10 pie, 10 scatter, 3 area, 3 histogram, 2 box, 2 heatmap) |
| Evaluator | `scripts/evaluation/benchmark/evaluate.py` | Compare predictions vs GT across 4 metrics |
| QA enrichment v2 | `scripts/evaluation/benchmark/enrich_annotations_v2.py` | Extract GT from existing QA data |
| Gemini annotator | `scripts/evaluation/benchmark/gemini_annotate.py` | Direct Vision API annotation |

### 3. Gemini Vision Ground Truth - COMPLETED

- **50/50 charts** annotated with gemini-2.5-flash
- **0 errors** (retry logic + JSON cleanup handled edge cases)
- Rich structured data: element_count, axis ranges, title, texts, data_values
- Raw responses saved to `data/benchmark/results/gemini_raw/`
- Annotations written to `data/benchmark/annotations/`

### 4. Gemini Model Upgrade - COMPLETED

| Change | Before | After |
| --- | --- | --- |
| Default model | gemini-2.0-flash | gemini-2.5-flash |
| Max output tokens | 8,192 | 65,536 |
| Files updated | - | .env, config/models.yaml, gemini_annotate.py |

New reference document: `docs/guides/GEMINI_MODELS_API.md`

### 5. Benchmark Results - COMPLETED

| Metric | Score | Target | Verdict |
| --- | --- | --- | --- |
| Classification Accuracy | **92.0%** | >= 90% | PASS |
| Element Count (+-25%) | **16.0%** | >= 70% | FAIL |
| Element Type Accuracy | **86.0%** | - | - |
| Axis Range (+-15%) | **0.0%** | >= 60% | FAIL |

**By Chart Type:**

| Type | N | Cls | Elem Acc | Insight |
| --- | --- | --- | --- | --- |
| bar | 10 | 90% | 50% | Best performer |
| pie | 10 | 100% | 30% | Some over-detection |
| line | 10 | 80% | 0% | Under/over-count |
| scatter | 10 | 100% | 0% | Can't count dense points |
| area | 3 | 100% | 0% | Under-detected |
| histogram | 3 | 100% | 0% | Under-detected |
| box | 2 | 100% | 0% | 0 elements found |
| heatmap | 2 | 50% | 0% | 0 elements found |

---

## Key Findings

### Stage 3 Strengths
1. **Classification (92%)**: ResNet-18 reliably identifies chart types on real-world academic charts
2. **Architecture**: Pipeline runs crash-free on all 50 diverse chart images
3. **Element type detection (86%)**: Correct element types identified for most charts

### Stage 3 Weaknesses (Quantitatively Confirmed)
1. **Axis calibrator non-functional**: 0/40 non-pie charts detected any axis (all null)
2. **Element counting unreliable**: Only bar (50%) and pie (30%) above zero
3. **OCR completely silent**: 0/50 charts produced any text output
4. **Box/heatmap unsupported**: Stage 3 detects 0 elements for these types
5. **Scatter over-/under-detection**: GT ranges from 10 to 7000 points, S3 finds <100

### Implications for Thesis
- The geometric-only approach has a clear **ceiling**: high classification, poor extraction
- This validates the hybrid neuro-symbolic architecture: Stage 4 (AI reasoning) is essential
- The benchmark provides academic-grade evidence for the thesis results chapter

---

## Files Created/Modified

| File | Action | Description |
| --- | --- | --- |
| `scripts/evaluation/benchmark/gemini_annotate.py` | Created + Modified | Gemini Vision annotation pipeline |
| `scripts/evaluation/benchmark/evaluate.py` | Modified | String element_count handling, type-filtered counting |
| `scripts/evaluation/benchmark/enrich_annotations_v2.py` | Created | QA-to-annotation enrichment |
| `scripts/evaluation/benchmark/stratified_sampler.py` | Created | 50-chart stratified sampling |
| `data/benchmark/annotations/*.json` (50) | Created + Overwritten | Gemini Vision GT |
| `data/benchmark/results/gemini_raw/*.json` (50) | Created | Raw Gemini responses |
| `data/benchmark/results/evaluation_report.md` | Generated | Full evaluation report |
| `docs/guides/GEMINI_MODELS_API.md` | Created | Gemini model reference |
| `docs/reports/stage3_weakness_report.md` | Modified | Added Section 7: Benchmark Validation |
| `docs/progress/SESSION_LOG_20260304_BENCHMARK.md` | Created + Updated | Full session log |
| `.env` | Modified | Model upgrade to gemini-2.5-flash |
| `config/models.yaml` | Modified | Gemini provider model updated |
| `docs/CHANGELOG.md` | Modified | v4.4.0 entry |
| `docs/MASTER_CONTEXT.md` | Modified | Benchmark section, model info, phase status |

---

## Next Steps

| Priority | Task | Effort |
| --- | --- | --- |
| 1 | SLM model selection on cloud GPU (micro-training pipeline ready) | 2-4 hours |
| 2 | Full SLM training on 228k samples (Llama-3.2-1B or Qwen-2.5-1.5B) | 4-8 hours |
| 3 | Stage 3 axis calibrator fix (biggest extraction weakness) | Research needed |
| 4 | Enable OCR pipeline (currently silent) | Debug + threshold tuning |
| 5 | FastAPI + Celery serving layer | 1-2 days |
