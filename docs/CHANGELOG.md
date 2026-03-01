# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [4.0.0] - 2026-03-02

### Full Pipeline Complete + Academic Thesis

#### Added
- **Academic Thesis** (`docs/thesis_capstone/`)
  - 7 content chapters: introduction, literature review, methodology, system design and implementation, results and discussion, project management, conclusion
  - 25 visual assets: 7 PDF figures + 12 LaTeX tables + 6 TikZ diagrams
  - 21 bibliography entries (refs.bib)
  - Vietnamese content integration across all chapters (Core-first, Localize-second architecture)
  - XeLaTeX build with fontspec + babel[vietnamese]
  - 39 pages, 0 LaTeX errors, 0 undefined references

- **Thesis Figure Generation Scripts**
  - Scripts to generate all 7 PDF figures from project data
  - Scripts to generate all 12 LaTeX table .tex files
  - 6 TikZ architecture/flow diagrams (pipeline, AI router, layer architecture, etc.)

- **Stage 5: Reporting** (`src/core_engine/stages/s5_reporting.py`)
  - Insight generation: summary stats, trend (linear regression), comparison, anomaly (z-score)
  - JSON + human-readable text report output to `data/output/`
  - `ReportingConfig` Pydantic config model

- **Pipeline Wiring** (`src/core_engine/pipeline.py`)
  - `_initialize_stages()`: All 5 stages now live-instantiated from OmegaConf config
  - `run()`: Full sequential stage pipeline with proper session_id-context logging

- **AI Routing Layer** (`src/core_engine/ai/`)
  - `task_types.py`: `TaskType` enum (CHART_REASONING, OCR_CORRECTION, DESCRIPTION_GEN, DATA_VALIDATION)
  - `exceptions.py`: Typed exception hierarchy (AIProviderError, AIRateLimitError, AIAuthenticationError, AITimeoutError, AIInvalidResponseError, AIProviderExhaustedError)
  - `prompts.py`: All system prompts and user prompt formatter functions (versioned)
  - `adapters/base.py`: `BaseAIAdapter` ABC and `AIResponse` standardized dataclass
  - `adapters/gemini_adapter.py`: Google Generative AI SDK adapter with vision support
  - `adapters/openai_adapter.py`: OpenAI Chat Completions adapter with vision support
  - `adapters/local_slm_adapter.py`: HuggingFace Transformers adapter (4-bit quantization, LoRA; enabled=False until training complete)
  - `router.py`: `AIRouter` with per-task fallback chains, exponential backoff, health check caching, and `route_sync()` for non-async callers

- **Stage 4: AIRouterEngine** (`src/core_engine/stages/s4_reasoning/router_engine.py`)
  - Adapter bridge: implements `ReasoningEngine` ABC but delegates to `AIRouter`
  - Supports `engine="router"` option in `Stage4Reasoning._initialize_engine()`

- **AI Routing Tests** (`tests/test_ai/`)
  - 55 unit tests covering task_types, exceptions, prompts, adapter base, and router fallback logic
  - All tests use mock adapters, no real API calls required

#### Fixed
- **LaTeX Compilation** (76 errors -> 0 errors)
  - Stripped float wrappers from 12 table + 6 TikZ input files (nested float fix)
  - Added `fontspec` + `babel[vietnamese]` for XeLaTeX in main.tex
  - Replaced all `\foreignlanguage{vietnamese}{...}` with direct UTF-8 Vietnamese text
  - Added `align=center` to 3 TikZ node styles requiring it
  - Removed stray `\end{itemize}` from system_design_and_implementation.tex

#### Changed
- **Documentation overhaul**: README.md, MASTER_CONTEXT.md, CHANGELOG.md all updated to v4.0.0
- **Test count**: 232 tests collected (was 177), all passing
- **Project phases**: Phase 2 (Core Engine) marked COMPLETED, Phase 4 (Thesis) marked COMPLETED

---

## [Unreleased]

### Added
- **Adaptive LoRA target_modules detection** (`scripts/train_slm_lora.py`)
  - `detect_lora_target_modules(model_path)`: reads model `config.json` (no weight download)
    and maps `model_type` / `architectures` to correct LoRA target layers
  - Architecture map covers: Qwen2, Llama, Mistral, Mixtral, Phi, Phi3, Gemma, Gemma2, GPT-2, OPT, Falcon
  - Falls back to universal set (`q_proj`...`down_proj`) for unknown architectures
  - `setup_lora_config()` now accepts `target_modules` parameter
  - `train()` calls `detect_lora_target_modules(model_path)` automatically before creating LoraConfig
  - Default `DATA_PATH` updated to `data/slm_training_v2` (32k balanced dataset)

- **AIRouter local-only policy** (`src/core_engine/ai/router.py`)
  - `LOCAL_ONLY_CHAINS` constant: maps all TaskTypes to `["local_slm"]` only
  - `local_only: bool = False` parameter on `AIRouter.__init__()`
  - When `local_only=True`, cloud providers (Gemini, OpenAI) are completely excluded,
    `fallback_chains` argument is ignored, warning is logged
  - `[POLICY]` comment marks this as the required mode for production inference
  - Log message now includes `local_only=` flag

- **Stage 3 batch extraction: CUDA-aware rewrite** (`scripts/batch_stage3_parallel.py`)
  - Full rewrite with `multiprocessing.set_start_method("spawn")` — required for CUDA safety on Windows
  - `--gpu-workers N`: enable CUDA PaddleOCR in N workers (recommended 1-2 for 6GB VRAM)
  - `--no-gpu`: force CPU-only mode for all workers
  - `--status`: print per-type progress bar without running extraction
  - Worker init: explicit GPU/CPU tag in log; forces OCR cache load on startup
  - Processing order: smallest chart types first (area, box, pie...) for fast early progress
  - `print_status()`: shows ASCII progress bars per chart type
  - All previous flags retained: `--workers`, `--chart-type`, `--limit`

### Fixed
- **Stage 3 `process_image` always returned `axis_info=None`**
  (`src/core_engine/stages/s3_extraction/s3_extraction.py`)
  - `process_image()` (used by batch extraction script) never called `_calibrate_axes()`
  - Root cause: method was written as a simplified test path; fix was omitted
  - Fix: added `axis_info = self._calibrate_axes(texts, w, h)` after element detection
  - Fix: `axis_calibration_confidence` now computed from x/y calibration confidences
  - Fix: `axis_info` now passed to returned `RawMetadata` (was hardcoded `None`)
  - Impact: all future batch extraction runs will produce axis_info with min/max/scale_factor

### Notes
- **Dataset v1 deprecation**: `data/slm_training/` (9.7k line-heavy samples) is superseded by
  `data/slm_training_v2/` (32k balanced). Safe to delete after qwen-0.5b training completes.
  Do NOT delete while training terminal is still running.
- **QA count 32k vs 47k explained**: 46,910 classified images include non-chart classes
  (diagram, not_a_chart, other, table, uncertain = ~14,500 images). QA was generated only for
  the 8 valid chart types (bar/line/scatter/heatmap/box/histogram/pie/area = 32,364 images).


  - `scripts/prepare_slm_training_data.py` now fully consumes 32,445 per-chart Gemini QA JSON files
    (`data/academic_dataset/chart_qa_v2/generated/{type}/`) across 8 chart types
  - Target: ~30k balanced training samples (4,000 per chart type x 8 types)
  - Both `.json` (array) and `.jsonl` (line-delimited) saved for trainer compatibility
  - Metadata fields added: `image_id`, `difficulty`, `source` (generator model)

- **`download_models.py` improvements** (`scripts/download_models.py`)
  - `requires_token: True` for llama-1b and llama-3b (was incorrectly `False`)
  - `license_url` field added to Llama model registry entries; shown in error messages
  - `check_hf_token()` now reads `~/.cache/huggingface/token` (set by `huggingface_hub.login()`)
    in addition to env vars and `.env` file
  - Docstring updated: Llama-3.2 requires token + Meta license acceptance (not "public")
  - Llama 4 note added: Scout 17B / Maverick 17B are too large for 6GB VRAM training
  - 403 / GatedRepo error now shows specific license URL and step-by-step instructions
  - 401 error now includes `huggingface_hub.login()` as alternative to env var

- **HuggingFace Authentication**: logged in as `thatlq1812` via `huggingface_hub.login()`

### Fixed
- **`prepare_slm_training_data.py` -- key bug** (`scripts/prepare_slm_training_data.py`)
  - `qa.get("type")` replaced with `qa.get("question_type", qa.get("type", "unknown"))` (4 call sites)
    Previously ALL QA records were classified as "unknown" type, breaking distribution tracking
  - `process_qa_pair()` now uses caption/context from QA record as fallback when Stage 3 feature
    JSON is absent -- previously produced empty context strings for ~99.9% of records
  - `format_chart_context()` appends `[CAPTION]: ...` and `[CONTEXT]: ...` from QA record
  - Root cause of "84 samples" training run (Jan 30): script joined QA records against only
    406 Stage 3 feature files (all area/ only) -- inner join produced nearly empty dataset

### Notes
- **Llama download blocked**: must accept Meta license at
  `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct` (and 3B variant) before token access
  is granted. Requests processed hourly by Meta.
- **qwen-0.5b training** (running): step ~1800/2913 (62%), epoch 1.65, loss 0.33, token_accuracy 91%


  - `scripts/download_models.py`: Catalog-driven download script for Qwen2.5 (0.5B, 1.5B) and Llama-3.2 (1B, 3B) from HuggingFace Hub; supports `--list`, `--model`, `--all`, `--verify`, `--force` CLI flags; HF token resolution from env or `config/secrets/.env`
  - `scripts/train_slm_lora.py`: Full rewrite for trl 0.29 API compatibility
    - `MODEL_REGISTRY` with 4 models (qwen-0.5b, qwen-1.5b, llama-1b, llama-3b) and automatic local/remote path resolution
    - `format_conversation()`: uses `tokenizer.apply_chat_template` with ChatML/Llama-3 fallback
    - Uses `SFTConfig(max_length=, dataset_text_field=)` + `SFTTrainer(processing_class=, peft_config=)` (trl 0.29 breaking-change API)
    - Uses BFloat16 (`bf16=True`) and `dtype=` parameter -- correct for Qwen 2.5 / Llama 3.2
    - `--smoke-test` flag: 2 training steps to verify end-to-end pipeline
    - `--model` flag to select from registry
  - Smoke test verified: 18.4M trainable params (1.18% of 1.56B), 2 steps in 5s on RTX 3060 6GB
  - Installed: `peft==0.18.1`, `trl==0.29.0`

- **Stage 3: Hybrid Classifier** (`src/core_engine/stages/s3_extraction/s3_extraction.py`)
  - `ExtractionConfig.resnet_confidence_threshold: float = 0.65` -- new configurable threshold
  - `_classify_chart()` now returns `(ChartType, float)` tuple instead of `ChartType`
  - Step 7: captures `resnet_classification_confidence` without redundant second inference
  - Step 10: rule-based tie-breaker triggers when `confidence < threshold` OR `chart_type == UNKNOWN` (was UNKNOWN-only); geometry is more reliable for ambiguous area/line cases

### Fixed
- **Stage 3 test fixtures** (`tests/test_s3_extraction/test_stage3_integration.py`)
  - Removed incorrect `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` from 3 chart-save calls (bar, line, pie); OpenCV draws natively in BGR so double-converting caused R/B channel swap, confusing ResNet into predicting AREA for LINE charts
  - All 13 Stage 3 integration tests now pass (was 12/13)


- **SLM Training Dataset v2** (`data/slm_training/`)
  - Rebuilt from `qa_pairs_v2.json` (1,000 LINE chart records, generated by `gemini-2.5-flash`)
  - 9,620 Gemini QA conversations + 84 legacy bar chart conversations = **9,704 total**
  - Split: 7,763 train / 970 val / 971 test (80/10/10)
  - 13 question types: structural, extraction, multi_hop, trend, comparison, interpolation, threshold, range, counting, why_reasoning, percentage_change, optimal_point, caption_aware

- **Full Training Run: qwen-0.5b** (RUNNING as of 2026-02-28)
  - `python train_slm_lora.py --model qwen-0.5b --epochs 3 --batch-size 2 --lora-rank 16`
  - Trainable params: 8.8M / 502M (1.75%), 2,913 total steps, ETA ~2h on RTX 3060 6GB
  - Output: `models/slm/qwen2.5-0.5b-instruct-chart-lora/`

- **AI Routing Layer** (`src/core_engine/ai/`)
  - `task_types.py`: `TaskType` enum (CHART_REASONING, OCR_CORRECTION, DESCRIPTION_GEN, DATA_VALIDATION)
  - `exceptions.py`: Typed exception hierarchy (AIProviderError, AIRateLimitError, AIAuthenticationError, AITimeoutError, AIInvalidResponseError, AIProviderExhaustedError)
  - `prompts.py`: All system prompts and user prompt formatter functions (versioned)
  - `adapters/base.py`: `BaseAIAdapter` ABC and `AIResponse` standardized dataclass
  - `adapters/gemini_adapter.py`: Google Generative AI SDK adapter with vision support
  - `adapters/openai_adapter.py`: OpenAI Chat Completions adapter with vision support
  - `adapters/local_slm_adapter.py`: HuggingFace Transformers adapter (4-bit quantization, LoRA; enabled=False until training complete)
  - `router.py`: `AIRouter` with per-task fallback chains, exponential backoff, health check caching, and `route_sync()` for non-async callers
- **Stage 4: AIRouterEngine** (`src/core_engine/stages/s4_reasoning/router_engine.py`)
  - Adapter bridge: implements `ReasoningEngine` ABC but delegates to `AIRouter`
  - Supports `engine="router"` option in `Stage4Reasoning._initialize_engine()`
- **Stage 5: Reporting** (`src/core_engine/stages/s5_reporting.py`)
  - Insight generation: summary stats, trend (linear regression), comparison, anomaly (z-score)
  - JSON + human-readable text report output to `data/output/`
  - `ReportingConfig` Pydantic config model
- **Pipeline Wiring** (`src/core_engine/pipeline.py`)
  - `_initialize_stages()`: All 5 stages now live-instantiated from OmegaConf config (no more placeholders)
  - `run()`: Full sequential stage pipeline with proper session_id-context logging
- **Tests** (`tests/test_ai/`)
  - 55 unit tests covering task_types, exceptions, prompts, adapter base, and router fallback logic
  - All tests use mock adapters -- no real API calls required
- **Environment template** (`.env.example`)
  - Added `OPENAI_API_KEY` and `OPENAI_MODEL` entries
  - Clarified `GOOGLE_API_KEY` dual use (Gemini AI + Google Search)

### Planned
- Local SLM: Evaluate qwen-0.5b adapter after training; follow up with qwen-1.5b full training
- Local SLM: Enable `local_slm_adapter.py` in Stage 4 after evaluation
- Llama models: require `HF_TOKEN` with Meta license accepted at huggingface.co/meta-llama
- Data expansion: Run Gemini QA generation for bar/scatter/area charts (2,575 images in manifests)
- PaddleOCR compatibility fix for Windows
- Stage 5 + full pipeline integration test

---

## [0.4.0] - 2026-01-26

### Week 2: Stage 4 Reasoning with Gemini API [PARTIAL]

#### Added
- **Stage 4 Reasoning Module** (`src/core_engine/stages/s4_reasoning/`)
  - `s4_reasoning.py`: Main Stage4Reasoning orchestrator
  - `gemini_engine.py`: Google Gemini API integration
  - `base_engine.py`: Abstract base class for reasoning engines
  - Features:
    - OCR error correction (loo→100, O→0, 2O25→2025)
    - Academic-style description generation
    - Value mapping from pixel coordinates
    - Legend-color association
    - Rule-based fallback when API unavailable

- **Gemini API Configuration**
  - Model: `gemini-3.0-flash-preview`
  - Temperature: 0.3 (deterministic)
  - Max tokens: 2048
  - Vision support: Prepared (not yet enabled)

- **Notebook: Stage 4 Demo** (`notebooks/04_stage4_reasoning.ipynb`)
  - 18 cells demonstrating Stage 4 capabilities
  - Tests: Mock data, OCR correction, real chart processing
  - Full pipeline test (Stage 3 → Stage 4)

#### Changed
- **Default OCR Engine**: Changed from `paddleocr` to `easyocr`
  - Reason: PaddleOCR 3.3.x has oneDNN compatibility issues on Windows
  - EasyOCR works reliably on Windows Python 3.12
  - File: `src/core_engine/stages/s3_extraction/s3_extraction.py`

#### Fixed
- **Gemini Engine NoneType Error**: Safe null checks for `corrections` and `color_rgb`
- **OCR Engine Fallback**: Auto-fallback to EasyOCR when PaddleOCR fails

#### Known Issues
- PaddleOCR 3.3.x crashes on Windows with `NotImplementedError: ConvertPirAttribute2RuntimeAttribute`
- Gemini API occasionally returns unparseable JSON (fallback works)

#### Next Steps
- [ ] Implement local SLM (Qwen-2.5 / Llama-3.2)
- [ ] Add vision model support to Gemini engine
- [ ] Improve value mapping with geometric calibration
- [ ] Build Stage 5: Reporting

---

## [0.3.0] - 2026-01-25

### Week 1: ResNet-18 Classifier [COMPLETED]

#### Added
- **ResNet-18 Chart Classifier**
  - Model: ResNet-18 with 8-class output (area, bar, box, heatmap, histogram, line, pie, scatter)
  - Accuracy: 94.66% test accuracy (vs 37.5% baseline SimpleChartClassifier)
  - Training: 27 minutes on NVIDIA GPU, early stopping at epoch 15/100
  - Dataset: Academic dataset with stratified train/test split
  
- **Grad-CAM Explainability** (`scripts/generate_gradcam.py`)
  - Visual explanation of model attention regions
  - Target layer: ResNet-18 layer4[-1].conv2 (last convolutional layer)
  - Generated: 8 per-class visualizations + 1 summary (9 files total)
  - Output: `models/explainability/gradcam_*.png`
  
- **ONNX Model Export** (`scripts/export_resnet18_onnx.py`)
  - Cross-platform deployment format (42.64 MB)
  - Inference speed: 6.90ms mean (CPU), 144.9 images/sec throughput
  - Validation: PyTorch vs ONNX predictions match (max diff 0.000982)
  - Output: `models/onnx/resnet18_chart_classifier.onnx` + metadata JSON
  
- **Pipeline Integration** (`src/core_engine/stages/s3_extraction/resnet_classifier.py`)
  - Production wrapper: `ResNet18Classifier` class
  - API methods: `predict()`, `predict_with_confidence()`, `predict_batch()`, `get_class_probabilities()`
  - Device support: Auto-detection (CUDA > MPS > CPU)
  - Configuration: `config/models.yaml` updated with ResNet-18 settings

#### Changed
- **Stage 3 Classifier**: Replaced SimpleChartClassifier (37.5%) with ResNet-18 (94.66%)
- **Configuration**: Updated `config/models.yaml` with ResNet-18 paths and 8 chart classes

#### Fixed
- ONNX export device mismatch: Model on GPU, input on CPU
- Model loading structure: Checkpoint uses ResNetWrapper with "resnet." prefix
- Integration test validation: 93.75% accuracy (15/16 correct)

#### Tested
- Integration test: `scripts/test_resnet_integration.py`
  - Overall: 93.75% accuracy (15/16 samples)
  - Per-class: 7/8 types at 100%, area at 50% (1 misclassification)
  - Pass threshold: 90% (PASSED)
  - Output: Visualization grid + JSON results

#### Documentation
- `docs/reports/WEEK1_COMPLETION_SUMMARY.md`: Comprehensive completion report
- Evaluation results: Confusion matrix, per-class metrics, training curves
- Explainability: Grad-CAM visualizations showing model attention

---

## [0.2.0] - 2026-01-24

### Phase 1: Foundation [COMPLETED]

#### Added
- **Chart QA Dataset**: 2,852 classified charts with 13,297 QA pairs
  - Source: Arxiv academic papers (800+ PDFs)
  - Classification via Google Gemini API
  - 5 QA pairs per chart (structural, counting, comparison, reasoning, extraction)
  
- **Data Factory Tools** (`tools/data_factory/`)
  - PDF Miner: Extract images from PDF documents
  - Gemini Classifier: Chart detection and type classification
  - QA Generator: Automated QA pair generation
  
- **Core Engine Stages**
  - Stage 1: Ingestion (`src/core_engine/stages/s1_ingestion.py`)
  - Stage 2: Detection (`src/core_engine/stages/s2_detection.py`)
  
- **Documentation**
  - MASTER_CONTEXT.md: Project overview
  - PIPELINE_FLOW.md: 5-stage pipeline architecture
  - SYSTEM_OVERVIEW.md: System design
  - CHART_QA_GUIDE.md: Chart QA generation guide
  - ARXIV_DOWNLOAD_GUIDE.md: PDF download instructions

#### Data Statistics
| Metric | Value |
| --- | --- |
| Total Images Processed | 2,852 |
| Total QA Pairs Generated | 13,297 |
| Source PDFs | 800+ |
| Chart Types | bar, line, pie, scatter, area, other |

---

## [0.1.0] - 2026-01-19

### Project Initialization

#### Added
- Initial project structure (V3)
- Configuration files (`config/base.yaml`, `config/models.yaml`, `config/pipeline.yaml`)
- GitHub instructions for AI agents
- Basic schema definitions (`src/core_engine/schemas/`)
- Test fixtures and configuration

#### Structure
```
chart_analysis_ai_v3/
├── .github/instructions/    # AI agent guidelines
├── config/                  # YAML configurations
├── data/                    # Data directories
├── docs/                    # Documentation
├── src/core_engine/         # Main engine code
├── tests/                   # Test suite
└── tools/                   # Utility tools
```

---

## Versioning

- **Major version (X.0.0)**: Breaking changes, architecture redesign
- **Minor version (0.X.0)**: New features, phase completion
- **Patch version (0.0.X)**: Bug fixes, minor improvements
