#!/usr/bin/env python3
"""
Medium-Scale Ablation Study Runner (Progressive Scaling Stage 2).

After the micro-ablation (800 samples) established baseline hyperparameters
(LR=2e-4, rank=16, grad_accum=8), this medium-scale study (4000 samples)
tests architectural decisions:

  Config 1 (llama_baseline):  Llama-3.2-1B baseline on 4000 samples
  Config 2 (qwen_baseline):   Qwen-2.5-1.5B baseline (model comparison)
  Config 3 (linear_sched):    Linear LR scheduler (vs cosine)
  Config 4 (ctx2048):         max_seq_length=2048 (context length stress test)

This study answers three questions:
  1. Which base model generalizes better on chart QA? (Llama vs Qwen)
  2. Does cosine LR scheduler outperform linear at scale?
  3. Does max_seq_length=1024 truncate important data?

Usage:
    # Full medium ablation (4 configs, ~4-5h on RTX 3060)
    python scripts/training/run_medium_ablation.py

    # Dry run
    python scripts/training/run_medium_ablation.py --dry-run

    # Run specific configs only
    python scripts/training/run_medium_ablation.py --runs llama_baseline qwen_baseline

    # Compare existing results without retraining
    python scripts/training/run_medium_ablation.py --compare-only

    # Include evaluation on test set
    python scripts/training/run_medium_ablation.py --runs llama_baseline --eval-samples 500
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


# =============================================================================
# Medium Ablation Configurations
# =============================================================================


@dataclass
class MediumAblationRun:
    """Configuration for a single medium-scale ablation experiment."""

    name: str
    description: str
    model: str = "llama-1b"
    overrides: Dict[str, Any] = field(default_factory=dict)


# Anchored hyperparameters from micro-ablation:
#   LR=2e-4, rank=16, alpha=32, grad_accum=8, batch=2
# These are NOT varied -- only architectural factors change.

MEDIUM_CONFIGS: Dict[str, MediumAblationRun] = {
    "llama_baseline": MediumAblationRun(
        name="llama_baseline",
        description="Llama-3.2-1B baseline (anchored config from micro-ablation)",
        model="llama-1b",
    ),
    "qwen_baseline": MediumAblationRun(
        name="qwen_baseline",
        description="Qwen-2.5-1.5B baseline (cross-architecture comparison)",
        model="qwen-1.5b",
    ),
    "linear_sched": MediumAblationRun(
        name="linear_sched",
        description="Linear LR scheduler (vs default cosine)",
        model="llama-1b",
        overrides={"lr-scheduler": "linear"},
    ),
    "ctx2048": MediumAblationRun(
        name="ctx2048",
        description="Context length 2048 (stress test for long samples)",
        model="llama-1b",
        overrides={"max-length": 2048},
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
    Parse trainer_state.json from the latest checkpoint as metrics fallback.

    Args:
        run_output: Path to the ablation run output directory.

    Returns:
        Dict with extracted metrics, or None if not found.
    """
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


# =============================================================================
# Training Runner
# =============================================================================


def run_training(
    ablation: MediumAblationRun,
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    max_length: int = 1024,
    eval_steps: int = 10,
    save_steps: int = 50,
) -> Dict[str, Any]:
    """
    Execute a single training run for a medium ablation experiment.

    Args:
        ablation: Medium ablation configuration.
        data_dir: Path to training data directory.
        output_dir: Base output directory for this study.
        epochs: Number of training epochs.
        max_length: Default max sequence length.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.

    Returns:
        Dict with run metadata (timing, exit code, output path, metrics).
    """
    run_output = output_dir / f"med_{ablation.name}"

    python_cmd = get_python_cmd()
    cmd = [
        python_cmd, "scripts/training/train_slm_lora.py",
        "--model", ablation.model,
        "--data-dir", str(data_dir),
        "--output-dir", str(run_output),
        "--epochs", str(epochs),
        "--max-length", str(max_length),
        "--eval-steps", str(eval_steps),
        "--save-steps", str(save_steps),
        "--batch-size", "2",
        # Anchored from micro-ablation
        "--learning-rate", "2e-4",
        "--lora-rank", "16",
        "--lora-alpha", "32",
        "--gradient-accumulation-steps", "8",
    ]

    # Apply config-specific overrides
    for key, val in ablation.overrides.items():
        cmd.extend([f"--{key}", str(val)])

    logger.info(f"[{ablation.name}] Starting | model={ablation.model} | {ablation.description}")
    logger.info(f"[{ablation.name}] CMD: {' '.join(cmd[-10:])}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    final_dir = run_output / "final"
    success = result.returncode == 0 and final_dir.exists()

    run_meta: Dict[str, Any] = {
        "name": ablation.name,
        "description": ablation.description,
        "model": ablation.model,
        "overrides": {k: str(v) for k, v in ablation.overrides.items()},
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "success": success,
        "output_dir": str(run_output),
        "final_dir": str(final_dir) if success else None,
    }

    # Extract metrics from training_info.json
    training_info_path = run_output / "training_info.json"
    if training_info_path.exists():
        try:
            info = json.loads(training_info_path.read_text(encoding="utf-8"))
            for key in [
                "final_train_loss", "final_eval_loss", "best_eval_loss",
                "final_train_accuracy", "final_eval_accuracy",
                "trainable_params", "total_params", "trainable_pct",
                "vram_peak_mb", "lr_scheduler_type",
            ]:
                if info.get(key) is not None:
                    run_meta[key] = info[key]
        except Exception:
            pass

    # Fallback: read from trainer_state.json
    if not run_meta.get("final_train_loss"):
        state_metrics = _find_trainer_state(run_output)
        if state_metrics:
            run_meta.update(state_metrics)

    if success:
        logger.info(
            f"[{ablation.name}] DONE | time={elapsed:.0f}s | "
            f"train_loss={run_meta.get('final_train_loss', '?')} | "
            f"eval_loss={run_meta.get('final_eval_loss', '?')}"
        )
    else:
        logger.error(f"[{ablation.name}] FAILED | exit_code={result.returncode}")
        if result.stderr:
            logger.error(f"[{ablation.name}] stderr (last 500 chars): {result.stderr[-500:]}")

    return run_meta


# =============================================================================
# Evaluation Runner
# =============================================================================


def run_evaluation(
    model: str,
    final_dir: Path,
    output_path: Path,
    eval_samples: int = 500,
) -> Optional[Dict[str, Any]]:
    """
    Run evaluate_slm.py on a trained adapter.

    Args:
        model: Model key for the base model.
        final_dir: Path to the LoRA adapter directory.
        output_path: Where to save evaluation results JSON.
        eval_samples: Number of test samples to evaluate.

    Returns:
        Evaluation results dict, or None if failed.
    """
    python_cmd = get_python_cmd()
    cmd = [
        python_cmd, "scripts/evaluation/evaluate_slm.py",
        "--model", model,
        "--adapter-dir", str(final_dir),
        "--output", str(output_path),
        "--max-samples", str(eval_samples),
    ]

    logger.info(f"Evaluating {final_dir.parent.name} | model={model} | samples={eval_samples}")

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


# =============================================================================
# Report Generator
# =============================================================================


def print_medium_report(
    run_results: List[Dict[str, Any]],
    eval_results: Dict[str, Optional[Dict[str, Any]]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate and print the medium ablation comparison report.

    Args:
        run_results: List of training run metadata dicts.
        eval_results: Map of run name -> evaluation results.
        output_path: Optional path to save the report.

    Returns:
        Formatted report string.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("  MEDIUM ABLATION STUDY REPORT (Progressive Scaling Stage 2)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 110)
    lines.append("")

    # Training results table
    lines.append("--- Training Results ---")
    lines.append("")
    header = (
        f"{'Run':<18} {'Model':<12} {'Status':<6} {'Time':>7} "
        f"{'Train Loss':>11} {'Eval Loss':>10} {'Best Eval':>10} "
        f"{'Train Acc':>10} {'Eval Acc':>9} {'Params':>10} {'VRAM':>7}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for r in run_results:
        status = "OK" if r.get("success") else "FAIL"
        time_str = f"{r['elapsed_s']:.0f}s" if r.get("elapsed_s") else "N/A"
        train_loss = f"{r['final_train_loss']:.4f}" if r.get("final_train_loss") else "N/A"
        eval_loss = f"{r['final_eval_loss']:.4f}" if r.get("final_eval_loss") else "N/A"
        best_eval = f"{r['best_eval_loss']:.4f}" if r.get("best_eval_loss") else "N/A"
        train_acc = f"{r['final_train_accuracy']:.1%}" if r.get("final_train_accuracy") else "N/A"
        eval_acc = f"{r['final_eval_accuracy']:.1%}" if r.get("final_eval_accuracy") else "N/A"
        params = r.get("trainable_params", "N/A")
        if isinstance(params, (int, float)):
            params = f"{params / 1e6:.1f}M"
        vram = f"{r['vram_peak_mb']:.0f}M" if r.get("vram_peak_mb") else "N/A"
        model = r.get("model", "?")

        lines.append(
            f"{r['name']:<18} {model:<12} {status:<6} {time_str:>7} "
            f"{train_loss:>11} {eval_loss:>10} {best_eval:>10} "
            f"{train_acc:>10} {eval_acc:>9} {str(params):>10} {vram:>7}"
        )

    lines.append("")

    # Evaluation results table
    has_evals = any(v is not None for v in eval_results.values())
    if has_evals:
        lines.append("--- Evaluation Results (Test Set) ---")
        lines.append("")
        eval_header = (
            f"{'Run':<18} {'Model':<12} {'EM%':>6} {'Contains%':>10} "
            f"{'Numeric%':>9} {'BLEU-1':>7} {'Latency':>8}"
        )
        lines.append(eval_header)
        lines.append("-" * 75)

        for r in run_results:
            ev = eval_results.get(r["name"])
            if ev is None:
                lines.append(f"{r['name']:<18} {r.get('model', '?'):<12} {'(not evaluated)':>45}")
                continue
            agg = ev.get("aggregate", {})
            em = agg.get("exact_match", 0) * 100
            cont = agg.get("contains_match", 0) * 100
            num_acc = agg.get("numeric_accuracy")
            num_str = f"{num_acc * 100:.1f}%" if num_acc is not None else "N/A"
            bleu = agg.get("bleu_1", 0)
            lat = agg.get("mean_latency_s", 0)
            lines.append(
                f"{r['name']:<18} {r.get('model', '?'):<12} {em:>5.1f}% "
                f"{cont:>9.1f}% {num_str:>9} {bleu:>7.4f} {lat:>7.2f}s"
            )
        lines.append("")

    # Analysis section
    lines.append("--- Analysis ---")
    lines.append("")

    successful = [r for r in run_results if r.get("success")]

    # Model Comparison: llama vs qwen
    llama = next((r for r in successful if r["name"] == "llama_baseline"), None)
    qwen = next((r for r in successful if r["name"] == "qwen_baseline"), None)
    if llama and qwen:
        lines.append("[MODEL COMPARISON] Llama-3.2-1B vs Qwen-2.5-1.5B:")
        for metric, key, fmt in [
            ("Train Loss", "final_train_loss", ".4f"),
            ("Eval Loss", "final_eval_loss", ".4f"),
            ("Best Eval Loss", "best_eval_loss", ".4f"),
            ("Eval Accuracy", "final_eval_accuracy", ".1%"),
        ]:
            lv = llama.get(key)
            qv = qwen.get(key)
            if lv is not None and qv is not None:
                ls = f"{lv:{fmt}}"
                qs = f"{qv:{fmt}}"
                winner = "Llama" if lv < qv else "Qwen" if qv < lv else "Tie"
                if key == "final_eval_accuracy":
                    winner = "Llama" if lv > qv else "Qwen" if qv > lv else "Tie"
                lines.append(f"  {metric:<20}: Llama={ls}  Qwen={qs}  -> {winner}")
        lines.append("")

    # Scheduler Comparison: cosine vs linear
    cosine = llama  # baseline uses cosine
    linear = next((r for r in successful if r["name"] == "linear_sched"), None)
    if cosine and linear:
        lines.append("[SCHEDULER] Cosine vs Linear:")
        for metric, key, fmt in [
            ("Eval Loss", "final_eval_loss", ".4f"),
            ("Best Eval Loss", "best_eval_loss", ".4f"),
            ("Eval Accuracy", "final_eval_accuracy", ".1%"),
        ]:
            cv = cosine.get(key)
            lv = linear.get(key)
            if cv is not None and lv is not None:
                cs = f"{cv:{fmt}}"
                ls = f"{lv:{fmt}}"
                winner = "Cosine" if cv < lv else "Linear" if lv < cv else "Tie"
                if key == "final_eval_accuracy":
                    winner = "Cosine" if cv > lv else "Linear" if lv > cv else "Tie"
                lines.append(f"  {metric:<20}: Cosine={cs}  Linear={ls}  -> {winner}")
        lines.append("")

    # Context Length: 1024 vs 2048
    ctx1024 = llama  # baseline uses 1024
    ctx2048 = next((r for r in successful if r["name"] == "ctx2048"), None)
    if ctx1024 and ctx2048:
        lines.append("[CONTEXT LENGTH] 1024 vs 2048:")
        for metric, key, fmt in [
            ("Eval Loss", "final_eval_loss", ".4f"),
            ("Best Eval Loss", "best_eval_loss", ".4f"),
            ("Eval Accuracy", "final_eval_accuracy", ".1%"),
        ]:
            v1 = ctx1024.get(key)
            v2 = ctx2048.get(key)
            if v1 is not None and v2 is not None:
                s1 = f"{v1:{fmt}}"
                s2 = f"{v2:{fmt}}"
                lines.append(f"  {metric:<20}: 1024={s1}  2048={s2}")
        t1 = ctx1024.get("elapsed_s", 0)
        t2 = ctx2048.get("elapsed_s", 0)
        if t1 and t2:
            overhead = ((t2 - t1) / t1 * 100) if t1 > 0 else 0
            lines.append(f"  {'Time Overhead':<20}: 1024={t1:.0f}s  2048={t2:.0f}s  ({overhead:+.1f}%)")
        lines.append("")

    # Overall recommendation
    if successful:
        lines.append("[RECOMMENDATION]")
        best_by_eval = min(
            [r for r in successful if r.get("best_eval_loss")],
            key=lambda r: r["best_eval_loss"],
            default=None,
        )
        if best_by_eval:
            lines.append(
                f"  Best config by eval loss: {best_by_eval['name']} "
                f"(model={best_by_eval.get('model')}, "
                f"best_eval_loss={best_by_eval['best_eval_loss']:.4f})"
            )

    lines.append("")
    lines.append("=" * 110)

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
        description="Medium-scale ablation study (Progressive Scaling Stage 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "slm_training_medium",
        help="Training data directory (default: data/slm_training_medium)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/slm/medium-ablation/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per run (default: 5, sufficient for 4000 samples)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Default max sequence length (default: 1024)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=25,
        help="Evaluate every N steps (default: 25)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(MEDIUM_CONFIGS.keys()),
        default=None,
        help="Run only specific configs (default: all)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of test samples for evaluation (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training, only generate report from existing results",
    )
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "models" / "slm" / "medium-ablation"

    # Select runs
    selected = list(MEDIUM_CONFIGS.values())
    if args.runs:
        selected = [MEDIUM_CONFIGS[r] for r in args.runs]

    # Estimate steps per epoch: 4000 samples / (batch=2 * accum=8) = 250 steps/epoch
    steps_per_epoch = 4000 // (2 * 8)

    # ─── Dry run ───
    if args.dry_run:
        print()
        print("=" * 70)
        print("  MEDIUM ABLATION - DRY RUN PLAN")
        print("=" * 70)
        print()
        print(f"  Data:       {args.data_dir}")
        print(f"  Output:     {args.output_dir}")
        print(f"  Epochs:     {args.epochs}")
        print(f"  Max length: {args.max_length}")
        print(f"  Steps/epoch: ~{steps_per_epoch}")
        print(f"  Total steps: ~{steps_per_epoch * args.epochs}")
        print()
        for i, cfg in enumerate(selected, 1):
            override_str = ", ".join(f"{k}={v}" for k, v in cfg.overrides.items()) or "(none)"
            print(f"  [{i}/{len(selected)}] {cfg.name:<18} model={cfg.model:<12} overrides={override_str}")
            print(f"           {cfg.description}")
        print()
        est_minutes = len(selected) * args.epochs * steps_per_epoch * 3.5 / 60  # ~3.5s/step on RTX 3060
        print(f"  Estimated total time: ~{est_minutes:.0f} min ({est_minutes / 60:.1f}h) on RTX 3060")
        print()
        return

    # ─── Compare only ───
    if args.compare_only:
        ablation_dir = args.output_dir
        if not ablation_dir.exists():
            logger.error(f"Output dir {ablation_dir} does not exist")
            sys.exit(1)
        run_dirs = sorted([
            d for d in ablation_dir.iterdir()
            if d.is_dir() and d.name.startswith("med_")
        ])
        if not run_dirs:
            logger.error(f"No med_* dirs found in {ablation_dir}")
            sys.exit(1)
        logger.info(f"Auto-discovered {len(run_dirs)} run dirs in {ablation_dir}")

        run_results: List[Dict[str, Any]] = []
        eval_results: Dict[str, Optional[Dict[str, Any]]] = {}
        for rd in run_dirs:
            config_name = rd.name.replace("med_", "")
            config = MEDIUM_CONFIGS.get(config_name)
            run_meta: Dict[str, Any] = {
                "name": config_name,
                "description": config.description if config else "",
                "model": config.model if config else "unknown",
                "success": True,
                "elapsed_s": 0,
            }
            # Try training_info.json
            info_path = rd / "training_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    for key in [
                        "final_train_loss", "final_eval_loss", "best_eval_loss",
                        "final_train_accuracy", "final_eval_accuracy",
                        "trainable_params", "total_params", "trainable_pct",
                        "vram_peak_mb", "lr_scheduler_type", "elapsed_s",
                    ]:
                        if info.get(key) is not None:
                            run_meta[key] = info[key]
                    # Override model from actual training info
                    if info.get("model_key"):
                        run_meta["model"] = info["model_key"]
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
                eval_results[config_name] = json.loads(
                    eval_path.read_text(encoding="utf-8")
                )
            else:
                eval_results[config_name] = None

        report_path = args.output_dir / "medium_ablation_report.txt"
        print_medium_report(run_results, eval_results, output_path=report_path)

        summary_path = args.output_dir / "medium_ablation_summary.json"
        summary_path.write_text(
            json.dumps({"runs": run_results, "generated": datetime.now().isoformat()},
                       indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"Summary saved: {summary_path}")
        return

    # ─── Run medium ablation study ───
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_results_list: List[Dict[str, Any]] = []
    eval_results_dict: Dict[str, Optional[Dict[str, Any]]] = {}

    for i, ablation in enumerate(selected, 1):
        logger.info(f"")
        logger.info(f"{'=' * 70}")
        logger.info(f"  RUN {i}/{len(selected)}: {ablation.name} ({ablation.model})")
        logger.info(f"  {ablation.description}")
        logger.info(f"{'=' * 70}")

        # Determine max_length for this config
        run_max_length = int(ablation.overrides.get("max-length", args.max_length))

        run_meta = run_training(
            ablation=ablation,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_length=run_max_length,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
        )
        run_results_list.append(run_meta)

        # Run evaluation if requested
        if not args.skip_eval and run_meta["success"] and run_meta.get("final_dir"):
            eval_output = args.output_dir / f"med_{ablation.name}" / "eval_results.json"
            eval_result = run_evaluation(
                model=ablation.model,
                final_dir=Path(run_meta["final_dir"]),
                output_path=eval_output,
                eval_samples=args.eval_samples,
            )
            eval_results_dict[ablation.name] = eval_result
        else:
            eval_results_dict[ablation.name] = None

    total_time = time.time() - start_time
    logger.info(f"")
    logger.info(f"All runs complete | total_time={total_time:.0f}s ({total_time / 3600:.1f}h)")

    # Generate report
    report_path = args.output_dir / "medium_ablation_report.txt"
    print_medium_report(run_results_list, eval_results_dict, output_path=report_path)

    # Save summary
    summary = {
        "runs": run_results_list,
        "total_time_s": round(total_time, 1),
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "generated": datetime.now().isoformat(),
    }
    summary_path = args.output_dir / "medium_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
