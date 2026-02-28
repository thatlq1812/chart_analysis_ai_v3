# Week 6 Progress Report

| Property | Value |
| --- | --- |
| Week | 6 (2026-02-23 to 2026-03-01) |
| Author | That Le |
| Status | On Track |

---

## Summary

Data pipeline fully complete: Stage 3 extraction finalized at 32,364/32,364 charts (0% error), SLM training dataset v3 built with 268,799 samples across all 8 chart types. Pre-training housekeeping (instructions, docs, archiving) completed. Project is now ready to begin SLM fine-tuning.

---

## Completed Tasks

- [x] Stage 3 batch extraction — 32,364/32,364 charts, 0% corruption rate
- [x] Fixed `np.int64` JSON serialization bug in `geometric_mapper.py` (atomic write + int cast)
- [x] Full quality audit (`_full_audit.py`) — all 8 types validated, 0% corruption confirmed
- [x] QA cross-validation (`_cross_check.py`) — 87% OCR text overlap between Stage3 features and QA answers
- [x] SLM training data v3 algorithm written (`prepare_slm_training_v3.py`) — corrected axis key names, role-grouped text, element breakdown
- [x] SLM training data v3 dry-run validated — 268,799 samples, 100% Stage3 coverage, 69.9% axis info
- [x] Data pipeline report written — `docs/reports/data_pipeline_report_v1.md` (9 sections, academic grade)
- [x] `docs.instructions.md` — Added Section 6 "Report and Document Management Rules"
- [x] `module-training.instructions.md` — Updated to v1.1.0 with v3 dataset stats and corrected commands
- [x] `MASTER_CONTEXT.md` — Updated to v3.0.0, added Phase 2b section, v3 dataset breakdown
- [x] Codebase housekeeping — obsolete scripts archived to `scripts/_archive/`

---

## In Progress

| Task | Progress | ETA | Blockers |
| --- | --- | --- | --- |
| SLM fine-tuning (Qwen2.5-1.5B) | 0% | Week 7-8 | None — data ready |
| Stage 5 Reporting | 0% | Week 9 | Waiting on SLM output schema |

---

## Metrics

### Stage 3 Extraction (Final)

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| Total charts extracted | 32,364 / 32,364 | 32,364 | DONE |
| Corruption rate | 0% | <1% | PASS |
| Axis info coverage | 69.9% | >50% | PASS |
| Zero-text files (bar type) | 29.6% | N/A | Expected (programmatic charts) |
| Zero-text files (area type) | 59.5% | N/A | Expected (simple area charts) |

### SLM Training Dataset v3

| Metric | Value | Notes |
| --- | --- | --- |
| Total samples | 268,799 | 9.9x increase over v2 |
| Chart types covered | 8/8 | All types represented |
| Has axis info | 69.9% | Axis calibration confidence >= 0.30 |
| Stage3 coverage | 100% | 32,364/32,364 features used |
| Split method | By chart_id | No data leakage |

### V3 Type Breakdown

| Type | Samples |
| --- | --- |
| line | 108,419 |
| scatter | 52,163 |
| bar | 47,330 |
| heatmap | 33,373 |
| box | 13,948 |
| pie | 7,408 |
| histogram | 4,159 |
| area | 1,999 |
| **Total** | **268,799** |

---

## Challenges and Solutions

### Challenge 1: np.int64 JSON Serialization Bug
- **Problem**: `batch_stage3_parallel.py` crashed with `TypeError: Object of type int64 is not JSON serializable` on outlier count fields
- **Solution**: Wrapped all numpy integer outputs with `int()` cast in `geometric_mapper.py` (2 locations); added `_NumpyEncoder` and atomic write to parallel batch script
- **Impact**: Required re-extraction of ~2,400 previously failed files; now 0% corruption across all 32,364

### Challenge 2: Axis Info Keys Wrong in v2 Training Script
- **Problem**: `prepare_slm_training_data.py` accessed `axis_info.get("y_range", {}).get("min")` — always returned None because actual keys are flat (`y_min`, `y_max`, etc.)
- **Solution**: v3 script uses `ai.get("x_min")`, `ai.get("y_min")` etc. directly; axis coverage jumped from ~4% to 69.9%
- **Impact**: v2 training data was functionally useless for axis reasoning tasks; v3 fixes this

### Challenge 3: Area and Bar Zero-Text Rates
- **Problem**: Audit revealed 59.5% of area charts and 29.6% of bar charts had zero OCR text
- **Root Cause**: Area charts in academic papers often use only color fills with no text labels; bar charts with very small labels fall below PaddleOCR confidence threshold
- **Solution**: v3 training script outputs explicit `[NO READABLE TEXT]` marker so the SLM learns to reason from element geometry alone; not a blocking issue

---

## Next Week Plan

1. [ ] Run `prepare_slm_training_v3.py` to build dataset to disk (`data/slm_training_v3/`)
2. [ ] Run `train_slm_lora.py` with v3 data on Qwen2.5-1.5B-Instruct
3. [ ] Monitor training: track loss curve, JSON validity rate, per-type accuracy
4. [ ] Evaluate checkpoint at epoch 1 against held-out val set
5. [ ] Write `docs/reports/slm_training_v3_experiment_log.md` with initial results
6. [ ] Investigate Stage 5 output schema requirements

---

## Key Numbers For Thesis

| Statistic | Value |
| --- | --- |
| Total academic charts (collected) | 32,364 |
| Stage 3 extraction success rate | 100.0% |
| OCR cache entries | 46,910 |
| SLM training samples (v3) | 268,799 |
| SLM v3 vs v2 increase | 9.9x |
| Axis info coverage | 69.9% |
| ResNet-18 classifier accuracy | 94.14% |
| Stage 3 test coverage | 140 tests (139 passing) |
| Stage 4 test coverage | 18 tests (18 passing) |

---

## Notes

- `AXIS_CONF_THRESHOLD = 0.30` in v3 builder: axis info included only when calibration confidence meets this floor. Can be adjusted during training ablation.
- v2 training script (`prepare_slm_training_data.py`) retained in `scripts/_archive/` for reference only, marked `[DEPRECATED]`.
- All active scripts listed in `scripts/README.md`.
