#!/usr/bin/env python3
"""
Multi-Model Selection Runner for SLM Micro-Training.

Trains multiple candidate models on a mini-dataset, evaluates each,
and produces a comparison table for model selection. This is the
"Eval for Selection" step before committing to full-scale training.

Workflow:
    1. Train each candidate model on mini-dataset (3 epochs)
    2. Evaluate each on the test set
    3. Generate comparison table (Markdown)
    4. Recommend best model based on accuracy + latency tradeoff

Candidate Models:
    - llama-1b:  Llama-3.2-1B-Instruct (1.24B params)
    - qwen-1.5b: Qwen2.5-1.5B-Instruct (1.54B params)
    - qwen-0.5b: Qwen2.5-0.5B-Instruct (0.49B params)
    - llama-3b:  Llama-3.2-3B-Instruct (3.21B params) [optional, needs >24GB]

Usage:
    # Default: train llama-1b + qwen-1.5b on mini-dataset
    python scripts/training/run_model_selection.py \
        --data-dir data/slm_training_mini \
        --models llama-1b qwen-1.5b

    # Include all models
    python scripts/training/run_model_selection.py \
        --data-dir data/slm_training_mini \
        --models llama-1b qwen-1.5b qwen-0.5b

    # Custom epochs and eval samples
    python scripts/training/run_model_selection.py \
        --data-dir data/slm_training_mini \
        --models llama-1b qwen-1.5b \
        --epochs 3 --eval-samples 200

    # Skip training, only evaluate existing models
    python scripts/training/run_model_selection.py \
        --eval-only \
        --eval-dirs models/slm/llama-3.2-1b-instruct-chart-lora-micro/final \
                    models/slm/qwen2.5-1.5b-instruct-chart-lora-micro/final

    # Dry run (show plan without executing)
    python scripts/training/run_model_selection.py --dry-run --models llama-1b qwen-1.5b qwen-0.5b
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "slm"
EVAL_DIR = PROJECT_ROOT / "models" / "evaluation"

# Model registry (must match train_slm_lora.py)
MODEL_INFO = {
    "qwen-0.5b": {
        "display_name": "Qwen2.5-0.5B",
        "local_name": "qwen2.5-0.5b-instruct",
        "params": "0.49B",
        "base_model_arg": "qwen-0.5b",
    },
    "llama-1b": {
        "display_name": "Llama-3.2-1B",
        "local_name": "llama-3.2-1b-instruct",
        "params": "1.24B",
        "base_model_arg": "llama-1b",
    },
    "qwen-1.5b": {
        "display_name": "Qwen2.5-1.5B",
        "local_name": "qwen2.5-1.5b-instruct",
        "params": "1.54B",
        "base_model_arg": "qwen-1.5b",
    },
    "llama-3b": {
        "display_name": "Llama-3.2-3B",
        "local_name": "llama-3.2-3b-instruct",
        "params": "3.21B",
        "base_model_arg": "llama-3b",
    },
}


def get_python_cmd() -> str:
    """Get the correct Python command for the current environment."""
    # Check .venv first
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    venv_python_unix = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        return str(venv_python_unix)

    return sys.executable


def run_command(cmd: List[str], description: str) -> subprocess.CompletedProcess:
    """
    Run a command with real-time output streaming.

    Args:
        cmd: Command and arguments
        description: Human-readable description for logging

    Returns:
        CompletedProcess result
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")

    return result


def train_model(
    model_key: str,
    data_dir: Path,
    epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 4096,
    gradient_accumulation_steps: int = 8,
    lora_rank: int = 16,
    eval_steps: int = 50,
    save_steps: int = 100,
) -> Optional[Path]:
    """
    Train a single model on the mini-dataset.

    Args:
        model_key: Key from MODEL_INFO (e.g. 'llama-1b')
        data_dir: Path to mini-dataset directory
        epochs: Number of training epochs
        batch_size: Per-device batch size
        max_length: Max token sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        lora_rank: LoRA rank
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps

    Returns:
        Path to final adapter directory, or None if training failed
    """
    info = MODEL_INFO[model_key]
    output_dir = MODELS_DIR / f"{info['local_name']}-chart-lora-micro"

    logger.info(f"Training {info['display_name']} ({info['params']})")
    logger.info(f"  Output: {output_dir}")

    python_cmd = get_python_cmd()
    cmd = [
        python_cmd, "scripts/training/train_slm_lora.py",
        "--model", model_key,
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--lora-rank", str(lora_rank),
        "--eval-steps", str(eval_steps),
        "--save-steps", str(save_steps),
    ]

    t0 = time.time()
    result = run_command(cmd, f"Training {info['display_name']}")
    elapsed = time.time() - t0

    final_dir = output_dir / "final"
    if result.returncode == 0 and final_dir.exists():
        logger.info(
            f"Training complete | model={info['display_name']} | "
            f"time={elapsed:.0f}s | output={final_dir}"
        )
        return final_dir
    else:
        logger.error(f"Training FAILED for {info['display_name']}")
        return None


def evaluate_model(
    model_key: str,
    lora_path: Optional[Path],
    test_data_path: Path,
    eval_samples: int = 200,
    output_tag: str = "micro",
) -> Optional[Path]:
    """
    Evaluate a model (base or fine-tuned) on the test set.

    Args:
        model_key: Key from MODEL_INFO
        lora_path: Path to LoRA adapter (None for base model eval)
        test_data_path: Path to test.json
        eval_samples: Number of test samples
        output_tag: Tag for output filename

    Returns:
        Path to evaluation results JSON, or None if failed
    """
    info = MODEL_INFO[model_key]

    # Resolve base model path
    base_model_path = MODELS_DIR / info["local_name"]
    if not base_model_path.exists():
        # Fallback to HuggingFace repo_id format
        repo_map = {
            "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
            "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
            "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        }
        base_model_str = repo_map.get(model_key, str(base_model_path))
    else:
        base_model_str = str(base_model_path)

    suffix = "lora" if lora_path else "base"
    output_path = EVAL_DIR / f"{model_key}-{suffix}-{output_tag}.json"

    logger.info(
        f"Evaluating {info['display_name']} ({suffix}) | "
        f"samples={eval_samples}"
    )

    python_cmd = get_python_cmd()
    cmd = [
        python_cmd, "scripts/evaluation/evaluate_slm.py",
        "--base-model", base_model_str,
        "--test-data", str(test_data_path),
        "--output", str(output_path),
        "--max-samples", str(eval_samples),
        "--stratified",
    ]

    if lora_path:
        cmd.extend(["--lora-path", str(lora_path)])

    t0 = time.time()
    result = run_command(cmd, f"Evaluating {info['display_name']} ({suffix})")
    elapsed = time.time() - t0

    if result.returncode == 0 and output_path.exists():
        logger.info(
            f"Evaluation complete | model={info['display_name']} | "
            f"time={elapsed:.0f}s | output={output_path}"
        )
        return output_path
    else:
        logger.error(f"Evaluation FAILED for {info['display_name']} ({suffix})")
        return None


def generate_comparison(
    eval_paths: List[Path],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comparison table from evaluation results.

    Args:
        eval_paths: List of evaluation JSON file paths
        output_path: Optional output path for markdown file

    Returns:
        Markdown formatted comparison table
    """
    python_cmd = get_python_cmd()
    cmd = [
        python_cmd, "scripts/evaluation/evaluate_slm.py",
        "--compare",
    ] + [str(p) for p in eval_paths]

    if output_path:
        cmd.extend(["--compare-output", str(output_path)])

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    return result.stdout


def print_selection_report(
    eval_paths: List[Path],
) -> None:
    """
    Print a comprehensive model selection report with recommendation.

    Args:
        eval_paths: List of evaluation JSON paths
    """
    results = []
    for path in eval_paths:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    print()
    print("=" * 75)
    print("  MODEL SELECTION REPORT")
    print(f"  Generated: {datetime.now().isoformat()}")
    print("=" * 75)
    print()

    # Summary table
    header = f"{'Model':<30} {'EM%':>6} {'Contains%':>10} {'Numeric%':>9} {'BLEU-1':>7} {'Latency':>8} {'VRAM':>6}"
    print(header)
    print("-" * 75)

    best_em = -1
    best_model = ""

    for r in results:
        agg = r["aggregate"]
        name = r["model_name"]
        em = agg["exact_match"] * 100
        cont = agg["contains_match"] * 100
        num = (agg["numeric_accuracy"] * 100) if agg["numeric_accuracy"] is not None else 0
        bleu = agg["bleu_1"]
        lat = agg["mean_latency_s"]
        vram = agg["vram_peak_mb"]

        print(
            f"{name:<30} {em:>5.1f}% {cont:>9.1f}% {num:>8.1f}% "
            f"{bleu:>7.4f} {lat:>7.2f}s {vram:>5.0f}M"
        )

        # Simple scoring: weighted combination
        score = em * 0.4 + cont * 0.3 + num * 0.2 + bleu * 100 * 0.1
        if score > best_em:
            best_em = score
            best_model = name

    print("-" * 75)
    print()
    print(f"  RECOMMENDATION: {best_model}")
    print(f"  (Weighted score: EM*0.4 + Contains*0.3 + Numeric*0.2 + BLEU*0.1)")
    print()
    print("=" * 75)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model selection via micro-training + evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_INFO.keys()),
        default=["llama-1b", "qwen-1.5b"],
        help="Models to train and evaluate (default: llama-1b qwen-1.5b)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "slm_training_mini",
        help="Training data directory (default: data/slm_training_mini)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Test data path (default: data_dir/test.json)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument(
        "--max-length", type=int, default=4096,
        help="Max token sequence length (default: 4096)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument(
        "--eval-samples", type=int, default=200,
        help="Number of test samples for evaluation (default: 200)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate existing models",
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Existing LoRA adapter paths for --eval-only mode",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate base models (zero-shot) for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR,
        help="Output directory for evaluation results (default: models/evaluation/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running anything",
    )

    args = parser.parse_args()

    # Resolve test data path
    test_data = args.test_data
    if test_data is None:
        test_data = args.data_dir / "test.json"
        # Fall back to full test set if mini doesn't have it
        if not test_data.exists():
            test_data = PROJECT_ROOT / "data" / "slm_training_v3" / "test.json"

    if not test_data.exists():
        logger.error(f"Test data not found: {test_data}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Dry run ───
    if args.dry_run:
        print()
        print("=" * 60)
        print("  EXECUTION PLAN (dry run)")
        print("=" * 60)
        print()
        for i, model in enumerate(args.models, 1):
            info = MODEL_INFO[model]
            print(f"  Step {i}a: Train {info['display_name']} ({info['params']})")
            print(f"           data={args.data_dir}, epochs={args.epochs}")
            print(f"           max_length={args.max_length}, lora_rank={args.lora_rank}")
            print(f"  Step {i}b: Evaluate {info['display_name']} (LoRA)")
            if args.include_base:
                print(f"  Step {i}c: Evaluate {info['display_name']} (base/zero-shot)")
            print()
        print(f"  Final: Generate comparison table")
        print(f"         test_data={test_data}")
        print(f"         eval_samples={args.eval_samples}")
        print()
        est_time = len(args.models) * 2  # rough hours estimate
        print(f"  Estimated time: ~{est_time}-{est_time*2}h (depends on GPU)")
        print()
        return

    # ─── Run model selection ───
    start_time = time.time()
    eval_paths: List[Path] = []

    for model_key in args.models:
        info = MODEL_INFO[model_key]
        logger.info(f"{'='*60}")
        logger.info(f"Processing: {info['display_name']} ({info['params']})")
        logger.info(f"{'='*60}")

        lora_path = None

        if not args.eval_only:
            # Train
            lora_path = train_model(
                model_key=model_key,
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                lora_rank=args.lora_rank,
                eval_steps=50,
                save_steps=100,
            )
            if lora_path is None:
                logger.error(f"Skipping evaluation for {info['display_name']} (training failed)")
                continue

        elif args.eval_dirs:
            # Use provided eval dirs
            idx = args.models.index(model_key)
            if idx < len(args.eval_dirs):
                lora_path = args.eval_dirs[idx]
        else:
            # Try to find existing micro-training output
            micro_dir = MODELS_DIR / f"{info['local_name']}-chart-lora-micro" / "final"
            if micro_dir.exists():
                lora_path = micro_dir
            else:
                logger.warning(
                    f"No LoRA adapter found for {info['display_name']}. "
                    "Run without --eval-only first."
                )
                continue

        # Evaluate LoRA
        eval_path = evaluate_model(
            model_key=model_key,
            lora_path=lora_path,
            test_data_path=test_data,
            eval_samples=args.eval_samples,
            output_tag="micro",
        )
        if eval_path:
            eval_paths.append(eval_path)

        # Evaluate base model if requested
        if args.include_base:
            base_eval_path = evaluate_model(
                model_key=model_key,
                lora_path=None,
                test_data_path=test_data,
                eval_samples=args.eval_samples,
                output_tag="micro",
            )
            if base_eval_path:
                eval_paths.append(base_eval_path)

    # Generate comparison
    if len(eval_paths) >= 2:
        comparison_output = args.output_dir / "model_selection_comparison.md"
        table = generate_comparison(eval_paths, comparison_output)
        print(table)
        print_selection_report(eval_paths)

        logger.info(f"Comparison table saved: {comparison_output}")
    elif len(eval_paths) == 1:
        logger.info("Only 1 model evaluated, no comparison table generated")
        # Still print results
        with open(eval_paths[0]) as f:
            data = json.load(f)
        agg = data["aggregate"]
        print(f"\nResults for {data['model_name']}:")
        print(f"  EM={agg['exact_match']*100:.1f}% | Contains={agg['contains_match']*100:.1f}% | "
              f"Numeric={agg.get('numeric_accuracy', 0)*100:.1f}% | BLEU-1={agg['bleu_1']:.4f}")
    else:
        logger.warning("No models evaluated successfully")

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    # Save run metadata
    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "models": args.models,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "eval_samples": args.eval_samples,
        "eval_results": [str(p) for p in eval_paths],
        "total_time_s": round(total_time, 1),
    }
    meta_path = args.output_dir / "model_selection_run.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.info(f"Run metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
