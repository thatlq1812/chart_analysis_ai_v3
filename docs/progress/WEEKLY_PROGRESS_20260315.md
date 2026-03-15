# Weekly Progress Report - 2026-03-15

| Property | Value |
| --- | --- |
| **Week** | 2026-03-12 to 2026-03-15 |
| **Author** | That Le |
| **Focus** | PaddleOCR-VL Microservice Integration + Full Pipeline Demo |

---

## Summary

This week focused on integrating PaddleOCR-VL (Vintern-1B fine-tuned) as a 5th AI provider
via a standalone FastAPI microservice, resolving a `transformers` version conflict between
PaddleOCR-VL (`>=4.45`) and Vintern (`==4.44.2`). A full end-to-end pipeline demo (S1→S5)
was also completed, exercising EfficientNet-B0 classification + DePlot extraction + AIRouter.

---

## Completed Tasks

### 1. PaddleOCR-VL Microservice - COMPLETED

| Component | Description |
| --- | --- |
| `paddle_server.py` | FastAPI app (port 8001), loads `models/paddleocr_vl/` at startup |
| `/extract` endpoint | POST multipart image, returns structured data table |
| `/health` endpoint | GET health check for AIRouter ping |
| Prompt | `"Chart Recognition:"` instructs model to return structured data table |
| Isolation strategy | Separate venv (`setup_paddle_env.sh`) for `transformers>=4.45` |

### 2. PaddleVLAdapter - COMPLETED

| Component | Description |
| --- | --- |
| `src/core_engine/ai/adapters/paddlevl_adapter.py` | HTTP adapter, provider_id="paddlevl" |
| `health_check()` | Pings `/health`, returns False if server offline |
| Fallback | Router auto-falls back to Gemini vision if server not running |
| Task type | `TaskType.DATA_EXTRACTION` added to `task_types.py` |
| Router chains | `DATA_EXTRACTION: ["paddlevl", "gemini"]` wired in `router.py` |

### 3. Model Files Added - COMPLETED

| Path | Description |
| --- | --- |
| `models/paddleocr_vl/` | PaddleOCR-VL model (Vintern-1B fine-tuned, ~8GB) |
| `models/slm/vintern_finetuned/` | Vintern fine-tuned config files |
| PP-DocLayoutV2 layout model | Included in `models/paddleocr_vl/` |

### 4. Full Pipeline Demo - COMPLETED

- `scripts/pipeline/run_demo_s1_s5.py` (522 lines) — runs S1→S5 on sample bar/line/pie
- Stage 3: EfficientNet-B0 classifier + DePlot extractor
- Stage 4: AIRouter → Gemini / OpenAI fallback chain
- CLI: `--no-llm` (skip Stage 4), `--image <path>` (custom input)

### 5. EfficientNetClassifier in resnet_classifier.py - COMPLETED

- `EfficientNetClassifier` class and `create_efficientnet_classifier()` factory added
  alongside existing `ResNet18Classifier`
- Stage 3 now imports `EfficientNetClassifier` from `resnet_classifier.py`

### 6. Documentation Update - COMPLETED

| Document | Update |
| --- | --- |
| `README.md` | v7.0.0 — reflects PaddleOCR-VL, EfficientNet-B0, 299 tests, corrected structure |
| `docs/MASTER_CONTEXT.md` | v7.0.0 — PaddleOCR-VL provider, updated adapters table, new section 5.7 |
| `docs/CHANGELOG.md` | [7.0.0] entry added |

---

## Key Findings

1. **Dependency isolation via microservice**: `paddle_server.py` pattern cleanly resolves
   the `transformers` version conflict between PaddleOCR-VL and the main venv.
2. **Graceful fallback**: When paddle_server is offline, router transparently falls back
   to Gemini vision for DATA_EXTRACTION tasks — no code changes required in pipeline stages.
3. **Full demo validated**: `run_demo_s1_s5.py` runs the complete pipeline without errors
   using EfficientNet-B0 + DePlot + AIRouter.

---

## Files Created / Modified

| File | Action | Description |
| --- | --- | --- |
| `paddle_server.py` | Created | PaddleOCR-VL FastAPI microservice |
| `src/core_engine/ai/adapters/paddlevl_adapter.py` | Created | HTTP adapter for paddle_server |
| `src/core_engine/ai/task_types.py` | Modified | Added `DATA_EXTRACTION` task type |
| `src/core_engine/ai/router.py` | Modified | PaddleVLAdapter registered, DATA_EXTRACTION chains |
| `src/core_engine/ai/adapters/__init__.py` | Modified | Export PaddleVLAdapter |
| `src/core_engine/stages/s3_extraction/resnet_classifier.py` | Modified | Added EfficientNetClassifier |
| `src/core_engine/stages/s3_extraction/s3_extraction.py` | Modified | Import EfficientNetClassifier |
| `src/core_engine/stages/s3_extraction/__init__.py` | Modified | Export EfficientNetClassifier |
| `scripts/pipeline/run_demo_s1_s5.py` | Created | Full S1→S5 demo script |
| `models/paddleocr_vl/` | Created | PaddleOCR-VL model files (~8GB) |
| `models/slm/vintern_finetuned/` | Created | Vintern config files |
| `config/models.yaml` | Modified | paddlevl provider section added |
| `notebooks/01d_chart_classification.ipynb` | Modified | Minor updates |
| `README.md` | Modified | v7.0.0 |
| `docs/MASTER_CONTEXT.md` | Modified | v7.0.0 |
| `docs/CHANGELOG.md` | Modified | [7.0.0] entry |
| `docs/progress/WEEKLY_PROGRESS_20260315.md` | Created | This file |

---

## Next Steps

| Priority | Task | Effort |
| --- | --- | --- |
| 1 | SLM full training on cloud GPU (Llama-3.2-1B, v4 config) | 4-8 hours GPU |
| 2 | Benchmark PaddleOCR-VL vs DePlot on 50-chart test set | 2-4 hours |
| 3 | `src/api/` + `src/worker/` FastAPI + Celery serving layer | 1-2 days |
| 4 | Docker Compose multi-container deployment | 1 day |
| 5 | Thesis appendix: add PaddleOCR-VL results section | 2-3 hours |
