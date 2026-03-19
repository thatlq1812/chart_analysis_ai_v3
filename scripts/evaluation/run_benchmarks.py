#!/usr/bin/env python3
"""
Unified Benchmark Runner - CLI Entry Point

Runs one or more benchmark suites on the 50-chart annotated set and
saves structured results to data/benchmark/results/runs/.

Usage:
    # Run all suites
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --suites all

    # Run specific suites
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --suites vlm_extraction ocr_quality

    # VLM extraction with specific models
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py \\
        --suites vlm_extraction \\
        --vlm-models deplot matcha

    # OCR comparison with specific engines
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py \\
        --suites ocr_quality \\
        --ocr-engines paddleocr easyocr

    # Classifier benchmark
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --suites classifier

    # SLM reasoning (after training)
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py \\
        --suites slm_reasoning \\
        --slm-model models/slm/qwen2.5-7b-chart-lora-v4/final

    # Show all past runs
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py --list-runs

    # Quick test with 5 charts
    .venv/Scripts/python.exe scripts/evaluation/run_benchmarks.py \\
        --suites vlm_extraction --n-charts 5 --vlm-models deplot

Available suites:
    vlm_extraction  - DePlot vs MatCha vs Pix2Struct on chart-to-table
    ocr_quality     - PaddleOCR vs EasyOCR on chart text
    classifier      - EfficientNet-B0 chart type accuracy
    slm_reasoning   - SLM QA (requires trained model)
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "evaluation"))

from benchmarks import BenchmarkRunner, REGISTRY

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for Geo-SLM Chart Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--suites",
        nargs="+",
        default=["vlm_extraction"],
        help="Suite names to run. Use 'all' to run everything. "
             "Choices: vlm_extraction ocr_quality classifier slm_reasoning",
    )

    # Chart set
    parser.add_argument(
        "--n-charts",
        type=int,
        default=None,
        help="Limit number of charts (default: all 50 from manifest)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to benchmark_manifest.json (default: data/benchmark/benchmark_manifest.json)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for model inference (default: cpu)",
    )

    # VLM extraction options
    parser.add_argument(
        "--vlm-models",
        nargs="+",
        default=["deplot", "matcha", "pix2struct"],
        help="VLM backends to benchmark (default: deplot matcha pix2struct). "
             "Choices: deplot matcha pix2struct svlm",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=512,
        help="Max image patches for Pix2Struct-family models (default: 512)",
    )

    # OCR options
    parser.add_argument(
        "--ocr-engines",
        nargs="+",
        default=["paddleocr", "easyocr"],
        help="OCR engines to benchmark (default: paddleocr easyocr). "
             "Choices: paddleocr easyocr paddlevl qwen2vl",
    )
    parser.add_argument(
        "--ocr-lang",
        default="en",
        help="OCR language code (default: en)",
    )

    # Classifier options
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="Path to EfficientNet-B0 weights (default: from config/models.yaml)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Classifier confidence threshold (default: 0.70)",
    )

    # SLM options
    parser.add_argument(
        "--slm-model",
        default=None,
        help="Path to trained LoRA adapter directory (required for slm_reasoning)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for SLM generation (default: 256)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=5,
        help="Max QA pairs per chart for SLM benchmark (default: 5)",
    )

    # Baseline VLM options
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        default=["gemini"],
        help="VLM APIs for baseline benchmark (default: gemini). Choices: gemini openai",
    )

    # E2E pipeline options
    parser.add_argument(
        "--extractor-backend",
        default="deplot",
        help="VLM extractor backend for e2e/ablation (default: deplot). "
             "Choices: deplot matcha pix2struct svlm",
    )

    # Ablation options
    parser.add_argument(
        "--ablation-configs",
        nargs="+",
        default=["full_pipeline", "no_classifier", "deplot_only", "no_s4_reasoning"],
        help="Ablation configurations to test (default: full_pipeline no_classifier "
             "deplot_only no_s4_reasoning)",
    )

    # Utility
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print all past benchmark runs and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and chart list, then exit without running",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runner = BenchmarkRunner(
        manifest_path=args.manifest,
        n_charts=args.n_charts,
    )

    # -- List runs and exit --
    if args.list_runs:
        runner.print_registry()
        return

    # -- Resolve suite list --
    if "all" in args.suites:
        suite_names = REGISTRY.list_names()
    else:
        suite_names = args.suites

    # Validate suite names
    available = REGISTRY.list_names()
    for name in suite_names:
        if name not in available:
            print(f"ERROR: Unknown suite '{name}'. Available: {available}")
            sys.exit(1)

    # -- Print plan --
    print(f"\n{'='*60}")
    print(f"Benchmark Plan")
    print(f"{'='*60}")
    print(f"Suites     : {suite_names}")
    print(f"Charts     : {args.n_charts or 'all (50)'}")
    print(f"Device     : {args.device}")
    if "vlm_extraction" in suite_names:
        print(f"VLM models : {args.vlm_models}")
    if "ocr_quality" in suite_names:
        print(f"OCR engines: {args.ocr_engines}")
    if "slm_reasoning" in suite_names:
        print(f"SLM model  : {args.slm_model or 'NOT SET'}")
    if "baseline_vlm" in suite_names:
        print(f"Baseline   : {args.baseline_models}")
    if "e2e_pipeline" in suite_names:
        print(f"Extractor  : {args.extractor_backend}")
    if "ablation" in suite_names:
        print(f"Ablation   : {args.ablation_configs}")
    print(f"{'='*60}\n")

    if args.dry_run:
        chart_ids = runner._load_manifest()
        print(f"Charts that would be evaluated ({len(chart_ids)}):")
        for cid in chart_ids:
            print(f"  {cid}")
        print("\nDry run complete. Use --dry-run=false to execute.")
        return

    # -- Run suites --
    for suite_name in suite_names:
        kwargs: dict = {"device": args.device}

        if suite_name == "vlm_extraction":
            kwargs["models"] = args.vlm_models
            kwargs["max_patches"] = args.max_patches

        elif suite_name == "ocr_quality":
            kwargs["engines"] = args.ocr_engines
            kwargs["lang"] = args.ocr_lang

        elif suite_name == "classifier":
            kwargs["model_path"] = args.classifier_model
            kwargs["confidence_threshold"] = args.confidence_threshold

        elif suite_name == "slm_reasoning":
            kwargs["model_path"] = args.slm_model
            kwargs["max_new_tokens"] = args.max_new_tokens
            kwargs["n_questions_per_chart"] = args.n_questions

        elif suite_name == "baseline_vlm":
            kwargs["models"] = args.baseline_models

        elif suite_name == "e2e_pipeline":
            kwargs["extractor_backend"] = args.extractor_backend

        elif suite_name == "ablation":
            kwargs["configs"] = args.ablation_configs
            kwargs["extractor_backend"] = args.extractor_backend

        runner.run_suite(suite_name, **kwargs)

    runner.print_registry()
    print("\nAll benchmarks complete. Results saved to data/benchmark/results/runs/")


if __name__ == "__main__":
    main()
