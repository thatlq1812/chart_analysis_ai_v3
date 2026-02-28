---
applyTo: 'src/api/**,src/worker/**,server/**,docker-compose.yml'
---

# MODULE INSTRUCTIONS - Serving Layer & API

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-02-28 | That Le | API, task queue, and deployment architecture |

---

## 1. Overview

**Serving Module** provides the production-ready API layer, async task processing, and deployment infrastructure. This layer turns the chart analysis pipeline from a batch script into a deployable service.

**Architecture Stack:**
| Layer | Technology | Purpose |
| --- | --- | --- |
| API Gateway | FastAPI | REST endpoints + OpenAPI docs |
| Task Queue | Celery + Redis | Async pipeline job processing |
| State Management | SQLAlchemy + SQLite/PostgreSQL | Job tracking + result storage |
| Deployment | Docker Compose | Multi-container orchestration |

---

## 2. API Design (FastAPI)

### 2.1. Directory Structure

```
src/
  api/
    __init__.py
    main.py             # FastAPI app factory
    routes/
      __init__.py
      charts.py         # Chart analysis endpoints
      health.py         # Health check + status
      models.py         # Model management endpoints
    schemas/
      __init__.py
      requests.py       # Pydantic request models
      responses.py      # Pydantic response models
    dependencies.py     # FastAPI dependencies (DI)
    middleware.py       # CORS, logging, error handling
```

### 2.2. Core Endpoints

```python
# POST /api/v1/charts/analyze
# Submit a chart image for analysis
# Returns: job_id for async tracking
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_chart(
    file: UploadFile,
    options: AnalyzeOptions = Depends(),
    queue: CeleryQueue = Depends(get_queue),
) -> AnalyzeResponse:
    job_id = await queue.submit_analysis(file, options)
    return AnalyzeResponse(job_id=job_id, status="queued")

# GET /api/v1/charts/{job_id}/status
# Check analysis job status
@router.get("/{job_id}/status", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobStatus:
    ...

# GET /api/v1/charts/{job_id}/result
# Get analysis result (after completion)
@router.get("/{job_id}/result", response_model=ChartResult)
async def get_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> ChartResult:
    ...

# GET /api/v1/health
# Health check for all services
@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "queue_connected": True,
    }
```

### 2.3. API Rules

1. **All endpoints** return Pydantic response models (no raw dicts)
2. **File uploads** max 10MB, supported formats: PNG, JPG, PDF
3. **Analysis** is always async -- return `job_id` immediately
4. **Versioned** API paths: `/api/v1/...`
5. **CORS** configured per environment (dev: allow all, prod: whitelist)
6. **Error responses** follow RFC 7807 Problem Details format

---

## 3. Task Queue (Celery + Redis)

### 3.1. Worker Architecture

```python
# src/worker/tasks.py
from celery import Celery

app = Celery("chart_analysis")
app.config_from_object("config.celery_config")

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_chart_task(self, image_path: str, options: dict) -> dict:
    """Run full 5-stage pipeline on a chart image."""
    try:
        pipeline = ChartAnalysisPipeline(config)
        result = pipeline.process(image_path, options)
        return result.model_dump()
    except Exception as exc:
        self.retry(exc=exc)
```

### 3.2. Queue Configuration

```python
# config/celery_config.py
broker_url = "redis://localhost:6379/0"
result_backend = "redis://localhost:6379/1"

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

task_track_started = True
task_time_limit = 300       # 5 min hard limit per task
task_soft_time_limit = 240  # 4 min soft limit (cleanup)

worker_concurrency = 2      # RTX 3060 can handle 2 parallel GPU tasks
worker_prefetch_multiplier = 1  # Don't prefetch (GPU memory)
```

### 3.3. Task Queue Rules

1. **All pipeline execution** goes through Celery tasks -- never directly from API handlers
2. **Idempotent tasks** -- same input should produce same output (for retries)
3. **Monitor** queue length and worker health via Flower dashboard
4. **GPU tasks** limited to `worker_concurrency=2` (VRAM constraint)
5. **Results** stored in Redis with 24h TTL, then persisted to database

---

## 4. State Management (SQLAlchemy)

### 4.1. Database Models

```python
# src/api/models/job.py
from sqlalchemy import Column, String, DateTime, JSON, Enum
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    
    id = Column(String, primary_key=True)          # UUID
    status = Column(Enum("queued", "processing", "completed", "failed"))
    input_path = Column(String, nullable=False)
    result = Column(JSON, nullable=True)            # RefinedChartData JSON
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    processing_time_ms = Column(Integer, nullable=True)
```

### 4.2. Database Rules

1. **SQLite** for development, **PostgreSQL** for production
2. **Alembic** for all schema migrations (never raw SQL ALTER)
3. **Async sessions** (`AsyncSession`) in API routes
4. **Repository pattern** for database operations

---

## 5. Docker Deployment

### 5.1. Service Architecture

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis, worker]
    environment:
      - DATABASE_URL=sqlite:///data/db.sqlite3
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./models:/app/models
  
  worker:
    build: .
    command: celery -A src.worker.tasks worker --loglevel=info
    depends_on: [redis]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### 5.2. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps (OpenCV, PaddleOCR)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[serve]"

# Copy source
COPY src/ src/
COPY config/ config/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.3. Deployment Rules

1. **Docker images** must be <2GB (use slim base, multi-stage builds)
2. **Model weights** are volume-mounted, NOT baked into images
3. **Secrets** via environment variables, NEVER in Dockerfiles or compose files
4. **Health checks** in docker-compose for all services
5. **GPU access** only for worker service (not API)

---

## 6. Development vs Production

| Aspect | Development | Production |
| --- | --- | --- |
| Database | SQLite | PostgreSQL |
| Queue | Redis (localhost) | Redis (managed) |
| API | uvicorn --reload | gunicorn + uvicorn workers |
| GPU | Direct access | Docker NVIDIA runtime |
| CORS | Allow all | Whitelist origins |
| Logging | DEBUG to console | INFO to file + monitoring |

---

## 7. Implementation Priority

This module is Phase 3 of the migration roadmap. Build order:

| Step | Component | Depends On | Effort |
| --- | --- | --- | --- |
| 1 | FastAPI app skeleton + health endpoint | Nothing | 2h |
| 2 | Request/Response Pydantic schemas | Existing schemas | 2h |
| 3 | Chart analysis endpoint (sync first) | Working pipeline | 4h |
| 4 | SQLAlchemy models + Alembic setup | Nothing | 3h |
| 5 | Celery task + Redis config | Docker Redis | 4h |
| 6 | Wire async endpoint -> Celery task | Steps 3-5 | 3h |
| 7 | Docker Compose full stack | All above | 4h |
| 8 | Flower monitoring dashboard | Step 5 | 1h |

**Total estimated:** ~23h

---

## 8. Rules Summary

1. **API responses** are always Pydantic models
2. **Pipeline execution** always goes through Celery (never inline in API)
3. **Database** accessed via Repository pattern with async sessions
4. **Migrations** managed by Alembic (no manual schema changes)
5. **Docker** is the deployment target -- all services must run in containers
6. **Model weights** are NEVER in Docker images (use volume mounts)
7. **Secrets** via env vars only
8. **Health checks** on every service
