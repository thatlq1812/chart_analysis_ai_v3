# Makefile for Geo-SLM Chart Analysis
# Common development commands

.PHONY: help install install-dev test lint format clean run-demo

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code"
	@echo "  make clean        Clean build artifacts"
	@echo "  make run-demo     Run Streamlit demo"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

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

# Running
run-demo:
	streamlit run interface/demo/app.py

run-cli:
	python -m interface.cli --help

# Development
dev-setup: install-dev
	pre-commit install

# Documentation
docs-serve:
	mkdocs serve

docs-build:
	mkdocs build
