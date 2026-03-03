#!/usr/bin/env python3
"""
Mini Ablation Study Runner.

Trains a single model under multiple hyperparameter configurations on a
tiny dataset (~400 samples) to:
  1. Verify the training pipeline works end-to-end (sanity check)
  2. Test overfitting capability (model can memorize small dataset)
  3. Measure hyperparameter sensitivity (LR, LoRA rank, grad accum)
  4. Compare VRAM usage across configs

This is NOT for finding the best accuracy; it is for validating the
pipeline and narrowing hyperparameter ranges before expensive cloud runs.

Ablation Runs:
  Run 0 (baseline): rank=16, lr=2e-4, grad_accum=8, epochs=10
  Run 1 (low_lr):   lr=1e-5 (20x smaller than baseline)
  Run 2 (rank8):    rank=8, alpha=16 (half capacity)
  Run 3 (no_accum): grad_accum=1 (different effective batch size)

After all runs complete, a summary report is printed and saved.

Usage:
    # Default: ablation on llama-1b with mini dataset
    python scripts/training/run_mini_ablation.py

    # Different model
    python scripts/training/run_mini_ablation.py --model qwen-1.5b

    # Custom data and epochs
    python scripts/training/run_mini_ablation.py \
        --data-dir data/slm_training_mini --epochs 15

    # Dry run (show plan)
    python scripts/training/run_mini_ablation.py --dry-run

    # Run only specific ablations
    python scripts/training/run_mini_ablation.py --runs baseline low_lr

    # Skip training, only compare existing run directories
    python scripts/training/run_mini_ablation.py --compare-only \
        --run-dirs runs/ablation_baseline_* runs/ablation_low_lr_*
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
RUNS_DIR = PROJECT_ROOT / "runs"


# =============================================================================
# Ablation Configuration
# =============================================================================


@dataclass
class AblationRun:
    """Configuration for a single ablation experiment."""

    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)


# Default ablation configurations
ABLATION_CONFIGS: Dict[str, AblationRun] = {
    "baseline": AblationRun(
        name="baseline",
        description="Default config (rank=16, lr=2e-4, grad_accum=8)",
        overrides={},
    ),
    "low_lr": AblationRun(
        name="low_lr",
        description="Low learning rate (1e-5, 20x smaller)",
        overrides={
            "learning-rate": 1e-5,
        },
    ),
    "rank8": AblationRun(
        name="rank8",
        description="Lower LoRA rank (rank=8, alpha=16)",
        overrides={
            "lora-rank": 8,
            "lora-alpha": 16,
        },
    ),
    "no_accum": AblationRun(
        name="no_accum",
        description="No gradient accumulation (grad_accum=1, effective batch=2)",
        overrides={
            "gradient-accumulation-steps": 1,
        },
    ),
}


def get_python_cmd() -> str:
    """Get the correct Python command for the current environment."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    venv_python_unix = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        return str(venv_python_unix)
    return sys.executable


def _find_trainer_state(run_output: Path) -> Optional[Dict[str, Any]]:
    """
    Parse trainer_state.json from checkpoints as fallback for metrics.

    Searches for the latest checkpoint directory and extracts final
    train/eval loss, accuracy, and best metric from HuggingFace Trainer state.

    Args:
        run_output: Path to the ablation run output directory.

    Returns:
        Dict with extracted metrics, or None if not found.
    """
    # Find all checkpoint dirs, pick the one with highest step
    ckpt_dirs = sorted(
        run_output.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if not ckpt_dirs:
        return None

    state_path = ckpt_dirs[-1] / "trainer_state.json"
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        log_history = state.get("log_history", [])
        train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
        eval_logs = [l for l in log_history if "eval_loss" in l]

        result: Dict[str, Any] = {}
        if train_logs:
            result["final_train_loss"] = train_logs[-1].get("loss")
            result["final_train_accuracy"] = train_logs[-1].get("mean_token_accuracy")
        if eval_logs:
            result["final_eval_loss"] = eval_logs[-1].get("eval_loss")
            result["final_eval_accuracy"] = eval_logs[-1].get("eval_mean_token_accuracy")
        if state.get("best_metric") is not None:
            result["best_eval_loss"] = state["best_metric"]
        return result if result else None
    except Exception:
        return None


def run_training(
    model: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    ablation: AblationRun,
    max_length: int = 1024,
    eval_steps: int = 10,
    save_steps: int = 50,
) -> Dict[str, Any]:
    """
    Execute a single training run for an ablation experiment.

    Args:
        model: Model key (e.g. 'llama-1b')
        data_dir: Path to training data directory
        output_dir: Base output directory for this run
        epochs: Number of training epochs
        ablation: Ablation configuration
        max_length: Max sequence length
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps

    Returns:
        Dict with run metadata (timing, exit code, output path)
    """
    run_output = output_dir / f"ablation_{ablation.name}"

    python_cmd = get_python_cmd()
    cmd = [
        python_cmd, "scripts/training/train_slm_lora.py",
        "--model", model,
        "--data-dir", str(data_dir),
        "--output-dir", str(run_output),
        "--epochs", str(epochs),
        "--max-length", str(max_length),
        "--eval-steps", str(eval_steps),
        "--save-steps", str(save_steps),
        "--batch-size", "2",
    ]

    # Apply baseline defaults
    if "learning-rate" not in ablation.overrides:
        cmd.extend(["--learning-rate", "2e-4"])
    if "lora-rank" not in ablation.overrides:
        cmd.extend(["--lora-rank", "16"])
    if "gradient-accumulation-steps" not in ablation.overrides:
        cmd.extend(["--gradient-accumulation-steps", "8"])

    # Apply ablation overrides
    for key, value in ablation.overrides.items():
        cmd.extend([f"--{key}", str(value)])

    logger.info(f"[{ablation.name}] {ablation.description}")
    logger.info(f"[{ablation.name}] Command: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    final_dir = run_output / "final"
    success = result.returncode == 0 and final_dir.exists()

    run_meta = {
        "name": ablation.name,
        "description": ablation.description,
        "overrides": {k: str(v) for k, v in ablation.overrides.items()},
        "model": model,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "success": success,
        "output_dir": str(run_output),
        "final_dir": str(final_dir) if success else None,
    }

    # Try to extract training info from training_info.json
    training_info_path = run_output / "training_info.json"
    if training_info_path.exists():
        try:
            info = json.loads(training_info_path.read_text(encoding="utf-8"))
            run_meta["final_train_loss"] = info.get("final_train_loss")
            run_meta["final_eval_loss"] = info.get("final_eval_loss")
            run_meta["best_eval_loss"] = info.get("best_eval_loss")
            run_meta["final_train_accuracy"] = info.get("final_train_accuracy")
            run_meta["final_eval_accuracy"] = info.get("final_eval_accuracy")
            run_meta["trainable_params"] = info.get("trainable_params")
            run_meta["total_params"] = info.get("total_params")
            run_meta["trainable_pct"] = info.get("trainable_pct")
            run_meta["vram_peak_mb"] = info.get("vram_peak_mb")
        except Exception:
            pass

    # Fallback: read from trainer_state.json if metrics still missing
    if not run_meta.get("final_train_loss"):
        trainer_state = _find_trainer_state(run_output)
        if trainer_state:
            run_meta.update(trainer_state)

    if success:
        logger.info(
            f"[{ablation.name}] DONE | time={elapsed:.0f}s | "
            f"loss={run_meta.get('final_train_loss', '?')}"
        )
    else:
        logger.error(f"[{ablation.name}] FAILED | exit_code={result.returncode}")

    return run_meta


def run_evaluation(
    model: str,
    final_dir: Path,
    test_data: Path,
    eval_samples: int = 80,
    output_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a trained model on the test set.

    Args:
        model: Model key
        final_dir: Path to final LoRA adapter
        test_data: Path to test.json
        eval_samples: Number of test samples
        output_path: Where to save evaluation JSON

    Returns:
        Evaluation results dict, or None if failed
    """
    python_cmd = get_python_cmd()

    # Resolve base model path
    from pathlib import Path as P
    models_dir = PROJECT_ROOT / "models" / "slm"
    model_map = {
        "llama-1b": "llama-3.2-1b-instruct",
        "llama-3b": "llama-3.2-3b-instruct",
        "qwen-0.5b": "qwen2.5-0.5b-instruct",
        "qwen-1.5b": "qwen2.5-1.5b-instruct",
    }
    local_name = model_map.get(model, model)
    base_path = models_dir / local_name
    if not base_path.exists():
        repo_map = {
            "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
            "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
            "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        }
        base_str = repo_map.get(model, str(base_path))
    else:
        base_str = str(base_path)

    if output_path is None:
        output_path = final_dir.parent / "eval_results.json"

    cmd = [
        python_cmd, "scripts/evaluation/evaluate_slm.py",
        "--base-model", base_str,
        "--lora-path", str(final_dir),
        "--test-data", str(test_data),
        "--output", str(output_path),
        "--max-samples", str(eval_samples),
        "--stratified",
    ]

    logger.info(f"Evaluating {final_dir.parent.name} | samples={eval_samples}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0 and output_path.exists():
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    logger.error(f"Evaluation failed for {final_dir}")
    if result.stderr:
        logger.error(f"stderr: {result.stderr[-500:]}")
    return None


def print_ablation_report(
    run_results: List[Dict[str, Any]],
    eval_results: Dict[str, Optional[Dict[str, Any]]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Print and optionally save the ablation comparison report.

    Args:
        run_results: List of training run metadata dicts
        eval_results: Map of run name -> evaluation results
        output_path: Optional path to save markdown report

    Returns:
        Markdown-formatted report string
    """
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  MINI ABLATION STUDY REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)
    lines.append("")

    # Training results table
    lines.append("--- Training Results ---")
    lines.append("")
    header = (
        f"{'Run':<15} {'Status':<6} {'Time':>6} {'Train Loss':>11} "
        f"{'Eval Loss':>10} {'Best Eval':>10} {'Train Acc':>10} "
        f"{'Eval Acc':>9} {'Params':>10} {'VRAM':>7}"
    )
    lines.append(header)
    lines.append("-" * 105)

    for r in run_results:
        status = "OK" if r["success"] else "FAIL"
        time_str = f"{r['elapsed_s']:.0f}s"
        train_loss = f"{r.get('final_train_loss', 0):.4f}" if r.get("final_train_loss") else "N/A"
        eval_loss = f"{r.get('final_eval_loss', 0):.4f}" if r.get("final_eval_loss") else "N/A"
        best_eval = f"{r.get('best_eval_loss', 0):.4f}" if r.get("best_eval_loss") else "N/A"
        train_acc = f"{r.get('final_train_accuracy', 0):.1%}" if r.get("final_train_accuracy") else "N/A"
        eval_acc = f"{r.get('final_eval_accuracy', 0):.1%}" if r.get("final_eval_accuracy") else "N/A"
        params = r.get("trainable_params", "N/A")
        if isinstance(params, (int, float)):
            params = f"{params/1e6:.1f}M"
        vram = f"{r.get('vram_peak_mb', 0):.0f}M" if r.get("vram_peak_mb") else "N/A"

        lines.append(
            f"{r['name']:<15} {status:<6} {time_str:>6} {train_loss:>11} "
            f"{eval_loss:>10} {best_eval:>10} {train_acc:>10} "
            f"{eval_acc:>9} {str(params):>10} {vram:>7}"
        )

    lines.append("")

    # Evaluation results table (if available)
    has_evals = any(v is not None for v in eval_results.values())
    if has_evals:
        lines.append("--- Evaluation Results ---")
        lines.append("")
        eval_header = (
            f"{'Run':<15} {'EM%':>6} {'Contains%':>10} {'Numeric%':>9} "
            f"{'BLEU-1':>7} {'Latency':>8}"
        )
        lines.append(eval_header)
        lines.append("-" * 60)

        for name, ev in eval_results.items():
            if ev is None:
                lines.append(f"{name:<15} {'(eval failed)':>45}")
                continue
            agg = ev.get("aggregate", {})
            em = agg.get("exact_match", 0) * 100
            cont = agg.get("contains_match", 0) * 100
            num_acc = agg.get("numeric_accuracy")
            num_str = f"{num_acc * 100:.1f}%" if num_acc is not None else "N/A"
            bleu = agg.get("bleu_1", 0)
            lat = agg.get("mean_latency_s", 0)

            lines.append(
                f"{name:<15} {em:>5.1f}% {cont:>9.1f}% {num_str:>9} "
                f"{bleu:>7.4f} {lat:>7.2f}s"
            )

        lines.append("")

    # Analysis
    lines.append("--- Analysis ---")
    lines.append("")

    successful = [r for r in run_results if r["success"]]
    if successful:
        baseline = next((r for r in successful if r["name"] == "baseline"), None)
        if baseline and baseline.get("final_train_loss"):
            bl_loss = baseline["final_train_loss"]
            if bl_loss < 0.5:
                lines.append(
                    f"[OK] Baseline train loss = {bl_loss:.4f} "
                    "(model can overfit mini dataset)"
                )
            elif bl_loss < 1.5:
                lines.append(
                    f"[WARN] Baseline train loss = {bl_loss:.4f} "
                    "(partial convergence, may need more epochs)"
                )
            else:
                lines.append(
                    f"[FAIL] Baseline train loss = {bl_loss:.4f} "
                    "(model is not learning -- check data format or tokenization)"
                )

        for r in successful:
            if r["name"] != "baseline" and r.get("final_train_loss") and baseline and baseline.get("final_train_loss"):
                diff = r["final_train_loss"] - baseline["final_train_loss"]
                if diff > 0.5:
                    lines.append(
                        f"[INFO] {r['name']}: loss {diff:+.4f} vs baseline "
                        "(significantly worse convergence)"
                    )
                elif diff > 0.1:
                    lines.append(
                        f"[INFO] {r['name']}: loss {diff:+.4f} vs baseline "
                        "(slightly worse)"
                    )
                else:
                    lines.append(
                        f"[INFO] {r['name']}: loss {diff:+.4f} vs baseline "
                        "(similar or better)"
                    )
    else:
        lines.append("[FAIL] No successful training runs!")

    lines.append("")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)

    if output_path:
        output_path.write_text(report, encoding="utf-8")
        logger.info(f"Report saved: {output_path}")

    return report


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini ablation study for SLM hyperparameter sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["llama-1b", "llama-3b", "qwen-0.5b", "qwen-1.5b"],
        default="llama-1b",
        help="Model to use for ablation (default: llama-1b)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "slm_training_mini",
        help="Training data directory (default: data/slm_training_mini)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per run (default: 10, enough to overfit 400 samples)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Evaluate every N steps (default: 10, frequent for mini dataset)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(ABLATION_CONFIGS.keys()),
        default=list(ABLATION_CONFIGS.keys()),
        help="Which ablation runs to execute (default: all)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=80,
        help="Number of test samples for evaluation (default: 80)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (default: models/slm/<model>-ablation/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training, only compare existing run directories",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Existing run directories for --compare-only mode",
    )

    args = parser.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        model_map = {
            "llama-1b": "llama-3.2-1b-instruct",
            "llama-3b": "llama-3.2-3b-instruct",
            "qwen-0.5b": "qwen2.5-0.5b-instruct",
            "qwen-1.5b": "qwen2.5-1.5b-instruct",
        }
        local_name = model_map.get(args.model, args.model)
        args.output_dir = PROJECT_ROOT / "models" / "slm" / f"{local_name}-ablation"

    # Resolve test data
    test_data = args.data_dir / "test.json"
    if not test_data.exists():
        test_data = PROJECT_ROOT / "data" / "slm_training_v3" / "test.json"

    # Build run list
    selected_runs = [ABLATION_CONFIGS[name] for name in args.runs]

    # ─── Dry run ───
    if args.dry_run:
        print()
        print("=" * 65)
        print("  MINI ABLATION - EXECUTION PLAN")
        print("=" * 65)
        print()
        print(f"  Model:    {args.model}")
        print(f"  Data:     {args.data_dir}")
        print(f"  Epochs:   {args.epochs}")
        print(f"  Output:   {args.output_dir}")
        print(f"  Test:     {test_data}")
        print()
        for i, run in enumerate(selected_runs, 1):
            overrides_str = ", ".join(f"{k}={v}" for k, v in run.overrides.items()) or "(defaults)"
            print(f"  Run {i}/{len(selected_runs)}: {run.name}")
            print(f"    Description: {run.description}")
            print(f"    Overrides:   {overrides_str}")
            print()
        est = len(selected_runs) * 15  # ~15 min per run on RTX 3060 with 400 samples
        print(f"  Estimated time: ~{est} min ({est/60:.1f}h) on RTX 3060")
        print()
        return

    # ─── Compare only ───
    if args.compare_only:
        # Auto-discover run dirs from output dir if --run-dirs not given
        run_dirs = args.run_dirs
        if not run_dirs:
            ablation_dir = args.output_dir
            if not ablation_dir.exists():
                logger.error(
                    f"--compare-only: output dir {ablation_dir} does not exist. "
                    "Provide --run-dirs or ensure ablation results exist."
                )
                sys.exit(1)
            run_dirs = sorted([
                d for d in ablation_dir.iterdir()
                if d.is_dir() and d.name.startswith("ablation_")
            ])
            if not run_dirs:
                logger.error(f"No ablation_* dirs found in {ablation_dir}")
                sys.exit(1)
            logger.info(f"Auto-discovered {len(run_dirs)} run dirs in {ablation_dir}")

        # Load training info from each dir
        run_results = []
        eval_results = {}
        for rd in run_dirs:
            rd = Path(rd)
            run_meta: Dict[str, Any] = {
                "name": rd.name.replace("ablation_", ""),
                "success": True,
                "elapsed_s": 0,
            }
            # Try training_info.json first
            info_path = rd / "training_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    for key in [
                        "final_train_loss", "final_eval_loss", "best_eval_loss",
                        "final_train_accuracy", "final_eval_accuracy",
                        "trainable_params", "total_params", "trainable_pct",
                        "vram_peak_mb", "elapsed_s",
                    ]:
                        if info.get(key) is not None:
                            run_meta[key] = info[key]
                except Exception:
                    pass

            # Fallback to trainer_state.json
            if not run_meta.get("final_train_loss"):
                state_metrics = _find_trainer_state(rd)
                if state_metrics:
                    run_meta.update(state_metrics)

            run_results.append(run_meta)

            eval_path = rd / "eval_results.json"
            if eval_path.exists():
                eval_results[run_meta["name"]] = json.loads(
                    eval_path.read_text(encoding="utf-8")
                )
            else:
                eval_results[run_meta["name"]] = None

        report_path = args.output_dir / "ablation_report.txt"
        print_ablation_report(run_results, eval_results, output_path=report_path)

        # Also save summary JSON
        summary_path = args.output_dir / "ablation_summary.json"
        summary_path.write_text(
            json.dumps({"runs": run_results, "generated": datetime.now().isoformat()},
                       indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Summary saved: {summary_path}")
        return

    # ─── Run ablation study ───
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_results: List[Dict[str, Any]] = []
    eval_results: Dict[str, Optional[Dict[str, Any]]] = {}

    for i, ablation in enumerate(selected_runs, 1):
        logger.info(f"{'='*65}")
        logger.info(
            f"[{i}/{len(selected_runs)}] {ablation.name}: {ablation.description}"
        )
        logger.info(f"{'='*65}")

        # Train
        result = run_training(
            model=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            ablation=ablation,
            max_length=args.max_length,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
        )
        run_results.append(result)

        # Evaluate
        if not args.skip_eval and result["success"] and result["final_dir"]:
            ev = run_evaluation(
                model=args.model,
                final_dir=Path(result["final_dir"]),
                test_data=test_data,
                eval_samples=args.eval_samples,
            )
            eval_results[ablation.name] = ev
        else:
            eval_results[ablation.name] = None

    # Generate report
    total_time = time.time() - start_time
    report_path = args.output_dir / "ablation_report.txt"
    print_ablation_report(run_results, eval_results, report_path)

    # Save full results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "total_time_s": round(total_time, 1),
        "runs": run_results,
    }
    summary_path = args.output_dir / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"Ablation study complete | total_time={total_time/60:.1f}min")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
