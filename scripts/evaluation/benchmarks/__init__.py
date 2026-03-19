"""
Unified Benchmark Framework for Geo-SLM Chart Analysis.

Suites:
    vlm_extraction  - DePlot vs MatCha vs Pix2Struct vs SVLM on 50-chart set
    ocr_quality     - PaddleOCR vs EasyOCR on chart text detection
    classifier      - EfficientNet-B0 chart type classification accuracy
    slm_reasoning   - SLM QA accuracy (after model training)
    baseline_vlm    - Zero-shot Gemini/GPT-4o baseline (thesis comparison)
    e2e_pipeline    - Full S1->S5 pipeline accuracy + latency
    ablation        - Systematic component ablation study

Usage:
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --suites vlm_extraction ocr_quality
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --suites baseline_vlm --baseline-models gemini
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --suites all
"""

from .registry import BenchmarkSuite, BenchmarkResult, SuiteRegistry, REGISTRY
from .runner import BenchmarkRunner

__all__ = ["BenchmarkSuite", "BenchmarkResult", "SuiteRegistry", "REGISTRY", "BenchmarkRunner"]
