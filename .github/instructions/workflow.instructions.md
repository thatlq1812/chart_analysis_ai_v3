---
applyTo: '.github/workflows/**'
---

# WORKFLOW INSTRUCTIONS - CI/CD Pipeline

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-02-28 | That Le | GitFlow with staging, model versioning, Docker deployment |

## 1. Overview

This document defines the CI/CD workflow strategy for Geo-SLM Chart Analysis using **GitFlow** adapted for ML projects. It covers code quality, model validation, and deployment.

### 1.1. Branch Strategy

```
main (development + staging)
  |
  +-- feature/feature-name      (create PR to main)
  +-- fix/bug-name              (create PR to main)
  +-- research/experiment-name  (experimental, no PR required)
      |
   [Tested & Stable] --> Tag release (v1.x.x)
      |
production (tagged releases only)
  |
  [Docker image built and pushed]
```

### 1.2. Key Differences from Web Projects

| Aspect | Web Project (elix) | ML Project (caav3) |
| --- | --- | --- |
| Deployment artifact | Docker image + DB migration | Docker image + model weights |
| Test suite | Unit + integration + E2E | Unit + integration + model validation |
| Release gate | All tests pass | All tests pass + model metrics above threshold |
| Data versioning | Not needed | DVC or hash-based versioning |
| CI cost | Low (fast builds) | Higher (GPU tests, large models) |

## 2. Development Workflow

### 2.1. Creating a New Feature

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/ai-routing

# Make changes and commit (Conventional Commits)
git commit -m "feat(ai): add BaseAIAdapter and router"

# Push to GitHub
git push origin feature/ai-routing
```

### 2.2. Opening Pull Request

1. Go to GitHub -> Pull Requests -> New Pull Request
2. **Base**: `main`
3. **Compare**: `feature/ai-routing`
4. Add clear description with:
   - What changed
   - How to test
   - Related issues

**GitHub Actions will automatically:**
- Run `ruff check` and `black --check`
- Run `mypy` type checking
- Run `pytest` with coverage report
- Validate model config files (YAML schema check)

### 2.3. Research Branches

Research branches (`research/*`) are exempt from PR requirements:

```bash
git checkout -b research/slm-distillation-v2

# Experiment freely, commit as needed
git commit -m "research: test LoRA rank 16 vs 32"

# If results are promising, create PR to main
# If not, archive branch notes and delete
```

## 3. CI/CD Pipelines

### 3.1. Code Quality Workflow (`code-quality.yml`)

**Triggers:** Push to any branch, Pull Request to `main`

```yaml
name: Code Quality
on:
  push:
    paths: ['src/**', 'tests/**', 'pyproject.toml']
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff black mypy
      - run: ruff check src/ tests/
      - run: black --check src/ tests/
      - run: mypy src/core_engine/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4
        if: env.CODECOV_TOKEN
```

### 3.2. Model Validation Workflow (`model-validation.yml`)

**Triggers:** Changes to `models/`, `config/models.yaml`, or manual trigger

```yaml
name: Model Validation
on:
  push:
    paths: ['models/**', 'config/models.yaml']
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - name: Validate YOLO model
        run: |
          python -c "
          from ultralytics import YOLO
          model = YOLO('models/weights/chart_detector.pt')
          print(f'Model loaded: {model.model.names}')
          "
      - name: Run sample inference
        run: |
          python scripts/test_full_pipeline.py --sample-only
```

### 3.3. Docker Build Workflow (`docker-build.yml`)

**Triggers:** Tag push (`v*`), manual trigger

```yaml
name: Docker Build
on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
            ghcr.io/${{ github.repository }}:latest
```

## 4. Release Process

### 4.1. Versioning Strategy

Follow **Semantic Versioning** with model version tracking:

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR: Breaking API changes or major architecture shifts
MINOR: New features, new stages, new AI providers
PATCH: Bug fixes, model weight updates, config changes
```

**Model versions are tracked separately:**

```yaml
# config/models.yaml
yolo:
  version: "1.2.0"      # Updated when weights change
  path: "models/weights/chart_detector_v1.2.0.pt"

classifier:
  version: "1.0.0"
  path: "models/weights/resnet18_best_v1.0.0.pt"

slm:
  version: "0.1.0"      # Pre-release during training
  name: "Qwen/Qwen2.5-1.5B-Instruct"
```

### 4.2. Creating a Release

```bash
# Update CHANGELOG.md
# Update version in pyproject.toml
# Commit and tag
git commit -m "release: v1.2.0"
git tag v1.2.0
git push origin main --tags
```

### 4.3. Docker Image Contents

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install ".[serving]"

COPY src/ src/
COPY config/ config/
COPY models/weights/ models/weights/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 5. Environment Configuration

### 5.1. Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | No | Gemini API (fallback provider) |
| `OPENAI_API_KEY` | No | OpenAI API (fallback provider) |
| `DATABASE_URL` | For serving | PostgreSQL connection string |
| `REDIS_URL` | For serving | Redis connection string |
| `MODEL_WEIGHTS_DIR` | No | Override model weights directory |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### 5.2. `.env.example`

```env
# AI Provider Keys (at least one required for Stage 4 cloud fallback)
GOOGLE_API_KEY=
OPENAI_API_KEY=

# Database (required for API server)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/caav3

# Redis (required for Celery task queue)
REDIS_URL=redis://localhost:6379/0

# Paths
MODEL_WEIGHTS_DIR=models/weights

# Logging
LOG_LEVEL=INFO
```

## 6. Makefile Commands

```makefile
# Development
make install        # pip install -e ".[dev]"
make test           # pytest with coverage
make lint           # ruff + black + mypy
make format         # black + ruff --fix

# Serving
make serve          # uvicorn src.api.main:app --reload
make worker         # celery -A src.tasks.celery_app worker
make migrate        # alembic upgrade head

# Docker
make docker-build   # docker build -t caav3 .
make docker-up      # docker-compose up -d
make docker-down    # docker-compose down

# Research
make notebook       # jupyter lab
make train-yolo     # python scripts/train_yolo_chart_detector.py
make train-slm      # python scripts/train_slm_lora.py
```
