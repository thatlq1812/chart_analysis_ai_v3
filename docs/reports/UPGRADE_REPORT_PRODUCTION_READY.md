# UPGRADE REPORT: caav3 Production-Ready Transformation

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-02-28 | That Le | Gap Analysis + Architecture Proposal (elix as reference) |

---

## STEP 1: Gap Analysis (elix vs caav3)

### 1.1. Documentation & Instructions Gap

| Dimension | elix (Reference) | caav3 (Current) | Gap Severity |
| --- | --- | --- | --- |
| Hierarchical Instructions | 25 files, module-level granularity | 7 files, flat structure | HIGH |
| On-Demand Loading | Module instructions loaded by task context | All instructions always loaded | MEDIUM |
| Confirmation Protocol | `docs/confirmation/` with AGENT_CONFIRM workflow | None | HIGH |
| Tech Specs | `docs/tech/DATABASE_SCHEMA.md`, `API_CONTRACTS.md` | Missing entirely | HIGH |
| Session Docs | `docs/sessions/` for dev session tracking | None | LOW |
| Workflow CI/CD | `workflow.instructions.md` + 3 GitHub Actions | No CI/CD instructions or pipelines | HIGH |
| Module Instructions | 1 file per module (billing, presentation, AI routing...) | Only `pipeline.instructions.md` covers all stages | MEDIUM |
| ROADMAP / STATUS | Dedicated `ROADMAP.md`, `STATUS.md` | Embedded inside `MASTER_CONTEXT.md` | LOW |

**Key Insight from elix:** The hierarchical instruction system saves 30-40% context tokens per AI conversation. caav3 loads everything every time, wasting context on irrelevant rules when working on a single stage.

### 1.2. Code Architecture Gap

| Dimension | elix (Reference) | caav3 (Current) | Gap Severity |
| --- | --- | --- | --- |
| AI Provider Routing | `BaseAIAdapter` + `AIRouter` + Fallback Chains | Single `GeminiReasoningEngine` hardcoded | CRITICAL |
| Async Task Queue | Celery + Redis with 8 task modules | Synchronous batch scripts | CRITICAL |
| Database Layer | SQLAlchemy + Alembic migrations | File-based JSON/cache, no DB | HIGH |
| State Management | JSONB columns, DB-backed state per entity | In-memory, session-based, no persistence | HIGH |
| API Layer | FastAPI with versioned routes (`api/v1/`) | Placeholder `interface/` directory | HIGH |
| Config Management | Pydantic Settings + `.env` + centralized `config.py` | OmegaConf YAML (good for ML, weak for secrets) | MEDIUM |
| DDD Module Structure | `models.py`, `schemas.py`, `repository.py`, `service.py` per module | Stages organized by pipeline order, not by domain | MEDIUM |
| Error Handling | Provider-specific exceptions + `retriable` flag + wrapping | Basic hierarchy exists but no retry/fallback logic | MEDIUM |
| Testing | pytest with fixtures + coverage + CI enforcement | 176/177 tests (good), but no CI enforcement | LOW |

### 1.3. Bottlenecks for Scaling

1. **Single AI Provider Lock-in**: `GeminiReasoningEngine` is tightly coupled. If Gemini API goes down or rate-limits, the entire Stage 4 fails with no fallback.

2. **Synchronous Pipeline**: `pipeline.run()` blocks sequentially. Processing 1,000 charts means waiting for each to complete before starting the next. No parallelism.

3. **No State Persistence**: If the pipeline crashes at Stage 3, all Stage 1-2 work is lost. No "resume from checkpoint" capability.

4. **Pipeline Orchestrator is a Skeleton**: `_initialize_stages()` in [pipeline.py](../../src/core_engine/pipeline.py) has all stage imports commented out with TODOs. The orchestrator cannot actually run end-to-end.

5. **Script-Driven Workflow**: `scripts/batch_stage3_parallel.py` and `scripts/test_full_pipeline.py` are ad-hoc scripts, not composable services. Each script reinvents initialization logic.

6. **No API Surface**: No way for external systems to submit charts for analysis. Everything runs via scripts or notebooks.

---

## STEP 2: Documentation & Instructions Restructure

### 2.1. Proposed `.github/instructions/` Structure

```
.github/instructions/
    README.md                                   # [EXISTS] Index of all instructions
    system.instructions.md                      # [UPGRADE] Global rules (enhanced)
    project.instructions.md                     # [UPGRADE] Project-specific rules (enhanced)
    workflow.instructions.md                    # [NEW] CI/CD + Git workflow
    coding-standards.instructions.md            # [EXISTS] Python coding standards
    docs.instructions.md                        # [EXISTS] Documentation standards
    pipeline.instructions.md                    # [EXISTS] Pipeline schemas
    research.instructions.md                    # [EXISTS] Research workflow
    module-detection.instructions.md            # [NEW] Stage 2 YOLO module
    module-extraction.instructions.md           # [NEW] Stage 3 OCR+Geometry module
    module-reasoning.instructions.md            # [NEW] Stage 4 AI Reasoning module
    module-serving.instructions.md              # [NEW] API + async task module
    module-training.instructions.md             # [NEW] Model training module
    references/
        shared_vocabulary.md                    # [EXISTS] Domain glossary
```

### 2.2. Proposed `docs/` Structure

```
docs/
    MASTER_CONTEXT.md                           # [EXISTS] Project overview
    README.md                                   # [EXISTS] Docs index
    architecture/
        SYSTEM_OVERVIEW.md                      # [EXISTS]
        PIPELINE_FLOW.md                        # [EXISTS]
        STAGE3_EXTRACTION.md                    # [EXISTS]
        STAGE4_REASONING.md                     # [EXISTS]
        STAGE5_REPORTING.md                     # [EXISTS]
        AI_ROUTING.md                           # [NEW] Multi-provider routing design
        ASYNC_ARCHITECTURE.md                   # [NEW] Celery/task queue design
        DATABASE_SCHEMA.md                      # [NEW] SQLAlchemy models
        API_CONTRACTS.md                        # [NEW] REST API specifications
    research/                                   # [EXISTS] Research docs
    guides/
        QUICK_START.md                          # [EXISTS]
        DEVELOPMENT.md                          # [EXISTS]
        CHART_QA_GUIDE.md                       # [EXISTS]
        ARXIV_DOWNLOAD_GUIDE.md                 # [EXISTS]
        DEPLOYMENT.md                           # [NEW] Docker + Cloud Run deployment
        API_USAGE.md                            # [NEW] API client guide
    reports/                                    # [EXISTS]
    confirmation/                               # [NEW] Agent confirmation protocol
        AGENT_CONFIRM_AI_ROUTING.md             # [NEW] AI routing decisions
        AGENT_CONFIRM_ASYNC.md                  # [NEW] Async architecture decisions
```

### 2.3. Key Instruction Files (Content Below)

The following 3 files are generated in **Step 2.3a/b/c**.

---

### 2.3a. `system.instructions.md` (Enhanced)

See file: `.github/instructions/system.instructions.md` (overwrite existing)

**Key Enhancements:**
- Added Hierarchical Instructions System (Section 0) from elix
- Added Agent Confirmation Protocol (Section 8) from elix
- Added Terminal/Background Service rules (Section 7.4) from elix
- Added Windows-specific Python venv rules (preserved from current)
- Added structured logging context rules (preserved from current)

### 2.3b. `project.instructions.md` (Enhanced)

See file: `.github/instructions/project.instructions.md` (overwrite existing)

**Key Enhancements:**
- Added Hierarchical Instructions hierarchy with on-demand loading
- Added new Technology Stack entries (Celery, Redis, SQLAlchemy, FastAPI)
- Added AI Provider Strategy section (multi-provider routing)
- Added Development Phases with concrete targets
- Added API-first architecture blueprint

### 2.3c. `workflow.instructions.md` (New)

See file: `.github/instructions/workflow.instructions.md` (new file)

**Key Enhancements:**
- GitFlow branch strategy adapted from elix
- CI/CD pipeline design for ML projects (test + lint + model validation)
- Docker-based deployment workflow
- Release process with model versioning

---

## STEP 3: Code Architecture Upgrade

### 3.1. Proposed `src/` Directory Tree

```
src/
    core_engine/
        __init__.py
        pipeline.py                             # [UPGRADE] Orchestrator with async support
        exceptions.py                           # [EXISTS] Enhanced with retry flags
        
        # --- Schemas (Pydantic) ---
        schemas/
            __init__.py
            common.py                           # [EXISTS] Shared types
            enums.py                            # [EXISTS] Chart types, status enums
            extraction.py                       # [EXISTS] Stage 3 schemas
            qa_schemas.py                       # [EXISTS] QA generation schemas
            stage_outputs.py                    # [EXISTS] Stage I/O contracts
            api_schemas.py                      # [NEW] Request/Response for API
            task_schemas.py                     # [NEW] Async task state schemas
        
        # --- AI Provider Routing (NEW - from elix pattern) ---
        ai/
            __init__.py
            router.py                           # AI Router (resolve task -> adapter)
            task_types.py                       # TaskType enum, fallback chains
            adapters/
                __init__.py
                base.py                         # BaseAIAdapter ABC
                gemini_adapter.py               # Google Gemini wrapper
                openai_adapter.py               # OpenAI GPT wrapper
                local_slm_adapter.py            # Local Qwen/Llama wrapper
            
        # --- Pipeline Stages ---
        stages/
            __init__.py
            base.py                             # [EXISTS] BaseStage ABC
            s1_ingestion.py                     # [EXISTS]
            s2_detection.py                     # [EXISTS]
            s3_extraction/                      # [EXISTS] OCR + Geometry sub-modules
                __init__.py
                classifier.py
                element_detector.py
                geometric_mapper.py
                ocr_engine.py
                preprocessor.py
                ...
            s4_reasoning/                       # [UPGRADE] Decoupled from Gemini
                __init__.py
                reasoning_engine.py             # [EXISTS] ABC for reasoning
                gemini_engine.py                # [REFACTOR] -> uses ai/adapters/
                local_slm_engine.py             # [NEW] Local model reasoning
                prompt_builder.py               # [EXISTS]
                value_mapper.py                 # [EXISTS]
                prompts/                        # [EXISTS]
            s5_reporting/                       # [NEW] Reporting stage
                __init__.py
                report_generator.py
                templates/
        
        # --- Data Factory ---
        data_factory/
            __init__.py
            qa_generator.py                     # [EXISTS]
        
        # --- Validators ---
        validators/
            __init__.py
            gemini_validator.py                 # [EXISTS]
            schema_validator.py                 # [NEW] Output schema validation
        
        # --- State Management (NEW - from elix pattern) ---
        state/
            __init__.py
            models.py                           # SQLAlchemy models (Chart, Job, Stage)
            repository.py                       # DB operations (CRUD)
            database.py                         # Engine + session factory
            migrations/                         # Alembic migrations
    
    # --- Async Task Queue (NEW - from elix Celery pattern) ---
    tasks/
        __init__.py
        celery_app.py                           # Celery config + broker
        pipeline_tasks.py                       # Full pipeline as Celery task
        stage_tasks.py                          # Individual stage tasks
        
    # --- API Server (NEW - from elix FastAPI pattern) ---
    api/
        __init__.py
        main.py                                 # FastAPI app factory
        config.py                               # Pydantic Settings
        v1/
            __init__.py
            routes_analyze.py                   # POST /analyze-chart, GET /status/{id}
            routes_models.py                    # GET /models, GET /health
            deps.py                             # Dependency injection (DB session, etc.)
```

### 3.2. Design Patterns Applied

#### Pattern 1: Adapter Pattern (AI Provider Routing)

**Source:** elix `server/src/core/ai/adapters/`

```python
# src/core_engine/ai/adapters/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningResult:
    """Standardized result from any AI reasoning provider."""
    content: str
    model_used: str
    provider: str
    corrections: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    token_usage: int = 0
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)


class BaseAIAdapter(ABC):
    """Abstract base for all AI reasoning adapters."""
    
    provider_id: str
    
    @abstractmethod
    async def reason(
        self,
        prompt: str,
        model_id: str,
        image_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> ReasoningResult:
        """Run reasoning on chart metadata + optional image."""
        ...
    
    @abstractmethod
    async def correct_ocr(
        self,
        ocr_texts: List[str],
        context: str,
        model_id: str,
    ) -> ReasoningResult:
        """Correct OCR errors using contextual reasoning."""
        ...
```

```python
# src/core_engine/ai/router.py
from .task_types import TaskType, FALLBACK_CHAINS, DEFAULT_MODELS
from .adapters.base import BaseAIAdapter
from .adapters.gemini_adapter import GeminiAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.local_slm_adapter import LocalSLMAdapter


class AIRouter:
    """Route AI tasks to the best available provider."""
    
    def __init__(self, config: dict):
        self._config = config
        self._adapters: dict[str, BaseAIAdapter] = {}
    
    def resolve(self, task_type: TaskType) -> tuple[BaseAIAdapter, str]:
        """
        Resolve adapter + model for a task type.
        
        Resolution order:
        1. Config override for this task_type
        2. Fallback chain (first available provider)
        
        Returns:
            Tuple of (adapter_instance, model_id)
            
        Raises:
            AllProvidersFailedError: No provider available
        """
        chain = FALLBACK_CHAINS[task_type]
        for provider_id in chain:
            if self._is_available(provider_id):
                adapter = self._get_or_create(provider_id)
                model_id = DEFAULT_MODELS[task_type][provider_id]
                return adapter, model_id
        raise AllProvidersFailedError(task_type)
```

#### Pattern 2: Async Task Queue (Celery)

**Source:** elix `server/src/celery_app.py` + `server/src/tasks/`

```python
# src/tasks/celery_app.py
from celery import Celery

celery_app = Celery(
    "caav3",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=[
        "src.tasks.pipeline_tasks",
        "src.tasks.stage_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_time_limit=600,        # 10 min max per task
    worker_prefetch_multiplier=1,
    worker_concurrency=4,       # 4 parallel chart processing
)
```

```python
# src/tasks/pipeline_tasks.py
from .celery_app import celery_app
from src.core_engine.pipeline import ChartAnalysisPipeline


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def analyze_chart(self, image_path: str, config_overrides: dict = None):
    """
    Full pipeline as a Celery task.
    
    - Auto-retries on transient failures (API timeout, etc.)
    - State tracked in Redis backend
    - Results stored in database
    """
    try:
        pipeline = ChartAnalysisPipeline.from_config(overrides=config_overrides)
        result = pipeline.run(image_path)
        return result.to_dict()
    except Exception as exc:
        self.retry(exc=exc)
```

#### Pattern 3: API-fication (FastAPI)

**Source:** elix `server/src/api/v1/`

```python
# src/api/v1/routes_analyze.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from src.tasks.pipeline_tasks import analyze_chart

router = APIRouter(prefix="/analyze", tags=["Chart Analysis"])


@router.post("/")
async def submit_chart(file: UploadFile = File(...)):
    """
    Submit a chart image for analysis.
    
    Returns task_id for polling status.
    """
    # Save uploaded file
    file_path = save_upload(file)
    
    # Submit to Celery
    task = analyze_chart.delay(str(file_path))
    
    return {"task_id": task.id, "status": "queued"}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Poll analysis status."""
    result = analyze_chart.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }
```

#### Pattern 4: State Persistence (SQLAlchemy)

**Source:** elix `server/src/core/database.py` + `server/alembic/`

```python
# src/core_engine/state/models.py
from sqlalchemy import Column, String, JSON, Float, Enum, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class AnalysisJob(Base):
    """Track each chart analysis job through the pipeline."""
    __tablename__ = "analysis_jobs"
    
    id = Column(String, primary_key=True)           # UUID
    source_file = Column(String, nullable=False)
    status = Column(
        Enum("queued", "processing", "completed", "failed", name="job_status"),
        default="queued",
    )
    current_stage = Column(String, nullable=True)    # s1, s2, s3, s4, s5
    stage_results = Column(JSON, default=dict)       # Per-stage outputs
    error_message = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

### 3.3. Migration Roadmap

| Phase | Action | Priority | Effort |
| --- | --- | --- | --- |
| Phase A | Create `ai/` directory with Router + Adapters | CRITICAL | 2-3 days |
| Phase B | Refactor `GeminiReasoningEngine` to use `GeminiAdapter` | CRITICAL | 1-2 days |
| Phase C | Add `LocalSLMAdapter` for Qwen/Llama local inference | HIGH | 2-3 days |
| Phase D | Add `state/` with SQLAlchemy models + Alembic | HIGH | 2-3 days |
| Phase E | Add `tasks/` with Celery integration | HIGH | 2-3 days |
| Phase F | Add `api/` with FastAPI endpoints | MEDIUM | 2-3 days |
| Phase G | Wire pipeline orchestrator to use Router + State | MEDIUM | 1-2 days |
| Phase H | CI/CD with GitHub Actions | LOW | 1 day |

**Total estimated effort: 2-3 weeks for full production-ready transformation.**

### 3.4. Dependency Additions

```toml
# pyproject.toml additions
[project.optional-dependencies]
serving = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "celery[redis]>=5.3.0",
    "redis>=5.0.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "alembic>=1.13.0",
    "asyncpg>=0.29.0",          # PostgreSQL async driver
    "python-multipart>=0.0.6",  # File uploads
    "httpx>=0.27.0",            # Async HTTP for AI APIs
]
```

This keeps the core engine installable without serving dependencies (`pip install .`), while `pip install ".[serving]"` brings in the full stack.

---

## Summary

| Category | Current State | Target State |
| --- | --- | --- |
| AI Providers | 1 (Gemini, hardcoded) | 3+ (Gemini, OpenAI, Local SLM) with auto-fallback |
| Processing | Synchronous, sequential | Async with Celery, parallel workers |
| State | In-memory, file-based | Database-backed with resume capability |
| Interface | Scripts + notebooks | REST API + CLI + Streamlit |
| CI/CD | None | GitHub Actions (test + lint + deploy) |
| Instructions | 7 flat files | 13+ hierarchical files with on-demand loading |
| Documentation | Good foundation | Enhanced with API specs, DB schema, deployment guides |

