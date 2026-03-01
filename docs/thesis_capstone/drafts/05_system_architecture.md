# System Architecture - Geo-SLM Chart Analysis

## 1. Overview

Geo-SLM Chart Analysis is a modular, hybrid AI system for extracting structured numerical data from chart images. The system combines three complementary paradigms:

| Paradigm | Role | Technology |
| --- | --- | --- |
| Deep Learning | Perception (detect, classify) | YOLOv8, ResNet-18 |
| Symbolic/Geometric | Precision (measure, calibrate) | OpenCV, NumPy, RANSAC |
| Language Model | Reasoning (correct, interpret) | Qwen-2.5 / Gemini API |

**Core thesis**: Treating charts as **geometric entities** rather than pixel patterns yields higher extraction accuracy than pure end-to-end multimodal approaches (e.g., GPT-4V, DePlot), while maintaining local inference capability.

## 2. Layered Architecture

The system follows a strict **5-layer separation**:

```
Layer 5 - Interface:      CLI (Typer) | Demo UI (Streamlit) | API (FastAPI)
Layer 4 - Task Queue:     Celery + Redis (async job dispatch)
Layer 3 - Core Engine:    5-Stage Pipeline Orchestrator
Layer 2 - AI Routing:     AIRouter -> [LocalSLM | Gemini | OpenAI] adapters
Layer 1 - Data/State:     PostgreSQL | Redis Cache | File Storage | Model Weights
```

**Design constraint**: Core Engine (Layer 3) has **zero dependency** on any web framework. The pipeline can run as a standalone Python library, a CLI tool, or behind FastAPI - without code changes.

## 3. Pipeline Architecture

### 3.1. Stage Abstraction

Every pipeline stage inherits from `BaseStage[InputT, OutputT]` (Generic ABC):

```python
class BaseStage(ABC, Generic[InputT, OutputT]):
    def process(self, input_data: InputT) -> OutputT: ...
    def validate_input(self, input_data: InputT) -> bool: ...
    def get_fallback_output(self, input_data: InputT) -> OutputT: ...
```

This enables:
- Independent development and testing of each stage
- Swapping implementations without pipeline changes
- Graceful degradation via `get_fallback_output()`

### 3.2. Data Flow

```
Input (PDF/Image)
  |
  v
Stage 1: Ingestion        -> Stage1Output (List[CleanImage])
  |  PyMuPDF, Pillow, OpenCV
  v
Stage 2: Detection         -> Stage2Output (List[DetectedChart])
  |  Ultralytics YOLO
  v
Stage 3: Extraction         -> Stage3Output (List[RawMetadata])
  |  PaddleOCR, OpenCV, ResNet-18, NumPy
  |  7 submodules: preprocessor, skeletonizer, vectorizer,
  |  OCR engine, element detector, geometric mapper, classifier
  v
Stage 4: Reasoning          -> Stage4Output (List[RefinedChartData])
  |  AIRouter -> LocalSLM / Gemini / OpenAI
  |  GeometricValueMapper + PromptBuilder + ReasoningEngine
  v
Stage 5: Reporting          -> PipelineResult (List[FinalChartResult])
  |  Insight generation, schema validation, JSON/Markdown export
  v
Output (JSON + Report)
```

### 3.3. Schema System

All inter-stage data is validated through **Pydantic v2 models** (frozen, extra="forbid"):

| Schema | Stage | Key Fields |
| --- | --- | --- |
| `CleanImage` | S1 -> S2 | image_path, width, height, page_number |
| `DetectedChart` | S2 -> S3 | chart_id, cropped_path, bbox, confidence |
| `RawMetadata` | S3 -> S4 | chart_type, texts[], elements[], axis_info, confidence |
| `RefinedChartData` | S4 -> S5 | title, series[], description, confidence |
| `FinalChartResult` | S5 out | chart data + insights[] + source_info |
| `PipelineResult` | Final | session, charts[], summary, processing_time |

Enums are centralized in `schemas/enums.py` (single source of truth): `ChartType` (8+3 types), `StageStatus`, `TextRole`, `ElementType`, `InsightType`.

## 4. AI Routing Layer

### 4.1. Adapter Pattern

All AI providers implement `BaseAIAdapter` (ABC):

```python
class BaseAIAdapter(ABC):
    provider_id: str
    async def reason(self, system_prompt, user_prompt, model_id, image_path, **kwargs) -> AIResponse: ...
    async def correct_ocr(self, ...) -> AIResponse: ...
    async def health_check(self) -> bool: ...
```

Concrete adapters: `GeminiAdapter`, `OpenAIAdapter`, `LocalSLMAdapter`.

`AIResponse` is a standardized dataclass: content, model_used, provider, confidence, usage, success, error_message.

### 4.2. Router with Fallback Chains

`AIRouter` walks a per-task fallback chain:

| TaskType | Default Chain |
| --- | --- |
| CHART_REASONING | local_slm -> gemini -> openai |
| OCR_CORRECTION | local_slm -> gemini |
| DESCRIPTION_GEN | local_slm -> gemini -> openai |
| DATA_VALIDATION | gemini -> openai |

Algorithm:
1. Get fallback chain for TaskType
2. Health-check each provider in order
3. Attempt first healthy provider
4. If confidence < threshold (0.7), try next
5. If all fail, raise `AIProviderExhaustedError`

**Local-only mode**: For production (privacy/cost), `local_only=True` restricts to local SLM only.

## 5. Configuration System

Three-tier YAML hierarchy merged via OmegaConf:

| File | Scope | Content |
| --- | --- | --- |
| `base.yaml` | Shared | Logging, paths, session config |
| `models.yaml` | Models | YOLO, ResNet, OCR, SLM, AI routing |
| `pipeline.yaml` | Pipeline | Stage toggles, thresholds, data factory |
| `training.yaml` | Training | LoRA config, QLoRA, curriculum |

Secrets (API keys) are loaded from `.env` via `python-dotenv`, never committed.

## 6. Error Handling

Hierarchical exception system:

```
ChartAnalysisError (base)
  +-- PipelineError (stage, recoverable, original_error)
  |     +-- StageInputError (expected_type, received_type)
  |     +-- StageProcessingError (fallback_available)
  +-- ConfigurationError (config_key)
  +-- ModelNotLoadedError
  +-- AIProviderError
        +-- AIAuthenticationError
        +-- AIRateLimitError
        +-- AIProviderExhaustedError
```

Every error carries `stage`, `recoverable`, and `fallback_available` flags for the orchestrator to decide whether to continue, retry, or abort.

## 7. Modularity and Extensibility

| Extension Point | Mechanism | Example |
| --- | --- | --- |
| New chart type | Add enum to `ChartType`, add element detection rules | Adding "waterfall" chart |
| New AI provider | Implement `BaseAIAdapter`, register in router | Adding Claude adapter |
| New pipeline stage | Implement `BaseStage[In, Out]`, register in orchestrator | Adding "Stage 6: Export" |
| New output format | Add to `OutputFormat` enum, add formatter in S5 | Adding Excel export |
| New OCR engine | Implement same interface as `OCREngine` | Adding EasyOCR v3 |
| Config override | Add YAML key, merge via OmegaConf | Tuning thresholds |

The entire pipeline is **config-driven**: all model paths, thresholds, stage toggles, and AI routing chains are externalized to YAML. No code changes needed to switch models or adjust parameters.

## 8. Technology Stack Summary

| Component | Technology | Version | Lines of Code |
| --- | --- | --- | --- |
| Pipeline orchestrator | Python + OmegaConf | 3.11+ | ~310 |
| Stage 1: Ingestion | PyMuPDF + Pillow + OpenCV | - | ~510 |
| Stage 2: Detection | Ultralytics YOLO | v8/v11 | ~340 |
| Stage 3: Extraction | PaddleOCR + OpenCV + NumPy | - | ~5,400 (7 modules) |
| Stage 4: Reasoning | Gemini API + custom prompts | - | ~2,500 (6 modules) |
| Stage 5: Reporting | Pydantic + JSON | - | ~635 |
| AI Routing Layer | Custom adapter/router | - | ~1,100 (5 modules) |
| Schemas | Pydantic v2 | - | ~1,000 (5 modules) |
| Tests | pytest | - | ~1,300 (43 files) |
| **Total core engine** | | | **~12,000+** |
