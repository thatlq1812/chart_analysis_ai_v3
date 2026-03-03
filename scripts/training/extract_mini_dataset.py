#!/usr/bin/env python3
"""
Extract stratified mini-datasets for micro-training experiments.

Produces train_mini.json and val_mini.json with balanced representation
across chart types, question types, and curriculum stages. Used for
quick model selection experiments before full-scale cloud training.

Usage:
    # Default: 5000 train + 500 val (stratified)
    python scripts/training/extract_mini_dataset.py

    # Custom sizes
    python scripts/training/extract_mini_dataset.py --train-size 10000 --val-size 1000

    # Different output directory
    python scripts/training/extract_mini_dataset.py --output-dir data/slm_training_mini

    # Analyze distribution only (no extraction)
    python scripts/training/extract_mini_dataset.py --analyze-only

    # Verify extracted mini-dataset
    python scripts/training/extract_mini_dataset.py --verify data/slm_training_mini
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "slm_training_v3"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "slm_training_mini"


# =============================================================================
# Stratified Sampling
# =============================================================================


def compute_distribution(
    data: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """
    Compute distribution statistics across chart_type, question_type,
    and curriculum_stage.

    Args:
        data: List of training samples with metadata

    Returns:
        Dict with 'chart_type', 'question_type', 'curriculum_stage' counts
    """
    stats: Dict[str, Counter] = {
        "chart_type": Counter(),
        "question_type": Counter(),
        "curriculum_stage": Counter(),
    }
    for sample in data:
        meta = sample.get("metadata", {})
        stats["chart_type"][meta.get("chart_type", "unknown")] += 1
        stats["question_type"][meta.get("question_type", "unknown")] += 1
        stats["curriculum_stage"][str(meta.get("curriculum_stage", "unknown"))] += 1

    return {k: dict(v.most_common()) for k, v in stats.items()}


def fixed_per_type_extract(
    data: List[Dict[str, Any]],
    per_type: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Extract exactly N samples per chart type for sanity-check experiments.

    This mode gives equal representation to every chart type regardless of
    the original dataset distribution. Useful for overfitting tests and
    mini ablation studies.

    Args:
        data: Full dataset
        per_type: Exact number of samples per chart type
        seed: Random seed for reproducibility

    Returns:
        Subset with exactly per_type samples per chart type
    """
    rng = random.Random(seed)

    # Group by chart_type
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in data:
        meta = sample.get("metadata", {})
        ct = meta.get("chart_type", "unknown")
        by_type[ct].append(sample)

    sampled: List[Dict[str, Any]] = []
    for ct in sorted(by_type.keys()):
        group = by_type[ct]
        rng.shuffle(group)
        n = min(per_type, len(group))
        sampled.extend(group[:n])
        if n < per_type:
            logger.warning(
                f"Chart type '{ct}' has only {len(group)} samples "
                f"(requested {per_type})"
            )

    rng.shuffle(sampled)
    return sampled


def stratified_extract(
    data: List[Dict[str, Any]],
    target_size: int,
    seed: int = 42,
    min_per_group: int = 2,
) -> List[Dict[str, Any]]:
    """
    Extract a stratified subset ensuring proportional representation
    across (chart_type, question_type) combinations.

    For rare combinations, at least `min_per_group` samples are included
    to prevent zero-coverage gaps.

    Args:
        data: Full dataset
        target_size: Desired number of samples
        seed: Random seed for reproducibility
        min_per_group: Minimum samples per (chart_type, question_type) group

    Returns:
        Stratified subset of data
    """
    rng = random.Random(seed)

    # Group by (chart_type, question_type)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in data:
        meta = sample.get("metadata", {})
        ct = meta.get("chart_type", "unknown")
        qt = meta.get("question_type", "unknown")
        key = f"{ct}|{qt}"
        groups[key].append(sample)

    total = len(data)
    sampled: List[Dict[str, Any]] = []

    # Phase 1: Ensure minimum coverage for every group
    for key, group in groups.items():
        rng.shuffle(group)
        n = min(min_per_group, len(group))
        sampled.extend(group[:n])

    # Track what is already sampled
    sampled_ids = {id(s) for s in sampled}
    remaining_budget = target_size - len(sampled)

    if remaining_budget <= 0:
        rng.shuffle(sampled)
        return sampled[:target_size]

    # Phase 2: Fill proportionally from remaining samples
    remaining_pool: Dict[str, List[Dict[str, Any]]] = {}
    for key, group in groups.items():
        remaining = [s for s in group if id(s) not in sampled_ids]
        if remaining:
            remaining_pool[key] = remaining

    total_remaining = sum(len(v) for v in remaining_pool.values())
    if total_remaining == 0:
        return sampled

    for key, group in remaining_pool.items():
        proportion = len(group) / total_remaining
        n = max(0, round(proportion * remaining_budget))
        n = min(n, len(group))
        rng.shuffle(group)
        sampled.extend(group[:n])

    # Trim if over-sampled
    if len(sampled) > target_size:
        rng.shuffle(sampled)
        sampled = sampled[:target_size]

    # Fill if under-sampled
    while len(sampled) < target_size:
        sampled_ids = {id(s) for s in sampled}
        all_remaining = [
            s
            for group in remaining_pool.values()
            for s in group
            if id(s) not in sampled_ids
        ]
        if not all_remaining:
            break
        rng.shuffle(all_remaining)
        need = target_size - len(sampled)
        sampled.extend(all_remaining[:need])

    return sampled


# =============================================================================
# Verification
# =============================================================================


def verify_mini_dataset(mini_dir: Path) -> bool:
    """
    Verify that a mini-dataset has valid structure and reasonable distribution.

    Args:
        mini_dir: Path to directory containing train_mini.json, val_mini.json

    Returns:
        True if validation passes
    """
    ok = True

    for split in ["train_mini", "val_mini"]:
        fpath = mini_dir / f"{split}.json"
        if not fpath.exists():
            logger.error(f"Missing file: {fpath}")
            ok = False
            continue

        data = json.loads(fpath.read_text(encoding="utf-8"))
        logger.info(f"{split}: {len(data)} samples")

        if len(data) == 0:
            logger.error(f"{split} is empty!")
            ok = False
            continue

        # Check all 8 chart types are present
        dist = compute_distribution(data)
        chart_types = set(dist["chart_type"].keys())
        expected_types = {"line", "bar", "scatter", "pie", "histogram", "heatmap", "box", "area"}
        missing = expected_types - chart_types
        if missing:
            logger.warning(f"{split}: Missing chart types: {missing}")

        # Check question type coverage
        qt_count = len(dist["question_type"])
        if qt_count < 8:
            logger.warning(f"{split}: Only {qt_count} question types (expected >=8)")

        # Print distribution table
        logger.info(f"{split} chart distribution:")
        for ct, count in sorted(dist["chart_type"].items()):
            pct = count / len(data) * 100
            logger.info(f"  {ct:<12} {count:>5} ({pct:.1f}%)")

        logger.info(f"{split} question distribution:")
        for qt, count in sorted(dist["question_type"].items()):
            pct = count / len(data) * 100
            logger.info(f"  {qt:<20} {count:>5} ({pct:.1f}%)")

        # Validate sample structure
        sample = data[0]
        if "conversations" not in sample:
            logger.error(f"{split}: Missing 'conversations' key")
            ok = False
        if "metadata" not in sample:
            logger.error(f"{split}: Missing 'metadata' key")
            ok = False

    if ok:
        logger.info("Verification PASSED")
    else:
        logger.error("Verification FAILED")

    return ok


# =============================================================================
# Main extraction
# =============================================================================


def extract_mini_datasets(
    source_dir: Path,
    output_dir: Path,
    train_size: int = 5000,
    val_size: int = 500,
    seed: int = 42,
    per_type: Optional[int] = None,
    val_per_type: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Extract stratified mini-datasets from the full training data.

    Args:
        source_dir: Directory with train.json, val.json
        output_dir: Output directory for train_mini.json, val_mini.json
        train_size: Target number of training samples (ignored if per_type set)
        val_size: Target number of validation samples (ignored if val_per_type set)
        seed: Random seed
        per_type: If set, use fixed_per_type_extract with N per chart type for train
        val_per_type: If set, use fixed_per_type_extract with N per chart type for val

    Returns:
        Tuple of (actual_train_size, actual_val_size)
    """
    # Load full datasets
    train_path = source_dir / "train.json"
    val_path = source_dir / "val.json"
    test_path = source_dir / "test.json"

    if not train_path.exists():
        logger.error(f"Train file not found: {train_path}")
        sys.exit(1)

    logger.info(f"Loading source data | dir={source_dir}")

    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    val_data = json.loads(val_path.read_text(encoding="utf-8"))

    logger.info(f"Source | train={len(train_data)} | val={len(val_data)}")

    # Clamp sizes
    train_size = min(train_size, len(train_data))
    val_size = min(val_size, len(val_data))

    # Extraction
    if per_type:
        logger.info(f"Extracting train_mini | per_type={per_type} | seed={seed}")
        train_mini = fixed_per_type_extract(train_data, per_type, seed=seed)
        vpt = val_per_type or max(1, per_type // 5)
        logger.info(f"Extracting val_mini | per_type={vpt} | seed={seed + 1}")
        val_mini = fixed_per_type_extract(val_data, vpt, seed=seed + 1)
    else:
        logger.info(f"Extracting train_mini | target={train_size} | seed={seed}")
        train_mini = stratified_extract(train_data, train_size, seed=seed)
        logger.info(f"Extracting val_mini | target={val_size} | seed={seed + 1}")
        val_mini = stratified_extract(val_data, val_size, seed=seed + 1)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train_mini.json"
    val_out = output_dir / "val_mini.json"

    train_out.write_text(
        json.dumps(train_mini, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    val_out.write_text(
        json.dumps(val_mini, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy test.json as-is (evaluation uses same test set for all models)
    if test_path.exists():
        test_out = output_dir / "test.json"
        test_out.write_text(test_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"Copied test.json (unchanged)")

    # Also create train.json and val.json symlinks/copies for compatibility
    # (train_slm_lora.py expects train.json/val.json)
    (output_dir / "train.json").write_text(
        json.dumps(train_mini, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "val.json").write_text(
        json.dumps(val_mini, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Extracted | train_mini={len(train_mini)} | val_mini={len(val_mini)} | "
        f"output={output_dir}"
    )

    # Print distribution summary
    print()
    print("=" * 60)
    print("  MINI-DATASET EXTRACTION SUMMARY")
    print("=" * 60)
    print()

    for name, data in [("train_mini", train_mini), ("val_mini", val_mini)]:
        dist = compute_distribution(data)
        print(f"--- {name} ({len(data)} samples) ---")
        print()
        print(f"  {'Chart Type':<12} {'Count':>6} {'%':>6}")
        print(f"  {'-'*12} {'-'*6} {'-'*6}")
        for ct, count in sorted(dist["chart_type"].items(), key=lambda x: -x[1]):
            print(f"  {ct:<12} {count:>6} {count/len(data)*100:>5.1f}%")
        print()
        print(f"  {'Question Type':<20} {'Count':>6} {'%':>6}")
        print(f"  {'-'*20} {'-'*6} {'-'*6}")
        for qt, count in sorted(dist["question_type"].items(), key=lambda x: -x[1]):
            print(f"  {qt:<20} {count:>6} {count/len(data)*100:>5.1f}%")
        print()

    # Write dataset_info.json
    info = {
        "source": str(source_dir),
        "extraction_mode": f"per_type={per_type}" if per_type else f"stratified",
        "train_mini_samples": len(train_mini),
        "val_mini_samples": len(val_mini),
        "test_samples": len(json.loads(test_path.read_text(encoding="utf-8"))) if test_path.exists() else 0,
        "seed": seed,
        "train_distribution": compute_distribution(train_mini),
        "val_distribution": compute_distribution(val_mini),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return len(train_mini), len(val_mini)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract stratified mini-datasets for micro-training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source dataset directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Number of validation samples (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        metavar="N",
        help="Extract exactly N samples per chart type (overrides --train-size). "
             "Validation gets N//5 per type. Example: --per-type 50 = 400 train + 80 val",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print source distribution without extracting",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        metavar="DIR",
        help="Verify an existing mini-dataset directory",
    )

    args = parser.parse_args()

    if args.verify:
        ok = verify_mini_dataset(args.verify)
        sys.exit(0 if ok else 1)

    if args.analyze_only:
        for split in ["train", "val", "test"]:
            fpath = args.source_dir / f"{split}.json"
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text(encoding="utf-8"))
            dist = compute_distribution(data)
            print(f"\n--- {split}.json ({len(data)} samples) ---")
            for field, counts in dist.items():
                print(f"  {field}:")
                for k, v in counts.items():
                    print(f"    {k}: {v} ({v/len(data)*100:.1f}%)")
        return

    extract_mini_datasets(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        per_type=args.per_type,
    )

    # Verify the result
    verify_mini_dataset(args.output_dir)


if __name__ == "__main__":
    main()
