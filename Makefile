# Makefile for Geo-SLM Chart Analysis
# Common development commands

.PHONY: help install install-dev install-api test lint format clean run-demo serve serve-dev ocr-build ocr-up ocr-down ocr-logs demo data-audit gcp-train gcp-sync-up gcp-pull-model thesis-update

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make install-api  Install API serving dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code"
	@echo "  make clean        Clean build artifacts"
	@echo "  make run-demo     Run Streamlit demo"
	@echo "  make serve        Start API server (production mode)"
	@echo "  make serve-dev    Start API server with hot-reload (dev mode)"
	@echo "  make ocr-build    Build OCR Docker service"
	@echo "  make ocr-up       Start OCR Docker service"
	@echo "  make ocr-down     Stop all Docker services"
	@echo "  make ocr-logs     Follow OCR service logs"
	@echo "  make demo         Start Gradio demo app"
	@echo "  make data-audit   Run data audit and generate manifest"
	@echo "  make gcp-train    Submit GCP training job"
	@echo "  make thesis-update Generate thesis tables and figures"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

install-api:
	pip install -e ".[api]"

# API Server
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

serve-dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow"

# Linting & Formatting
lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check src/ tests/ --fix

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Demo
demo:
	.venv\Scripts\python.exe interface/demo_app.py

# Running
run-demo:
	streamlit run interface/demo/app.py

run-cli:
	python -m interface.cli --help

# Development
dev-setup: install-dev
	pre-commit install

# OCR Service (Docker)
ocr-build:
	docker compose build ocr-service

ocr-up:
	docker compose up -d ocr-service

ocr-down:
	docker compose down

ocr-logs:
	docker compose logs -f ocr-service

# Data Audit
data-audit:
	.venv\Scripts\python.exe scripts/data_management/audit.py --output output/data_manifest.json

# Documentation
docs-serve:
	mkdocs serve

docs-build:
	mkdocs build

# GCP Training
GCS_BUCKET ?= chart-analysis-data

gcp-train:
	gcloud ai custom-jobs create --region=asia-southeast1 \
		--display-name=slm-training-$$(date +%Y%m%d) \
		--config=config/gcp/training_job.yaml

gcp-sync-up:
	gcloud storage cp -r data/slm_training_v3/ gs://$(GCS_BUCKET)/data/slm_training_v3/

gcp-pull-model:
	gcloud storage cp -r gs://$(GCS_BUCKET)/models/slm/latest/ models/slm/

# Thesis
thesis-update:
	.venv\Scripts\python.exe scripts/utils/generate_thesis_tables.py
	.venv\Scripts\python.exe scripts/utils/generate_thesis_figures.py
