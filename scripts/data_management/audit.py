"""
Data Audit Script

Scans the data/ directory recursively and produces a JSON manifest describing
all datasets, their sizes, distributions, and integrity status.

Usage:
    .venv\\Scripts\\python.exe scripts/data_management/audit.py
    .venv\\Scripts\\python.exe scripts/data_management/audit.py --check-integrity
    .venv\\Scripts\\python.exe scripts/data_management/audit.py --compare output/data_manifest.json
    .venv\\Scripts\\python.exe scripts/data_management/audit.py --output output/data_manifest.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_management.audit_schema import (
    BenchmarkStats,
    DataManifest,
    DiffReport,
    IntegrityReport,
    QAPairStats,
    SLMTrainingStats,
    SplitStats,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def _count_by_type(directory: Path, pattern: str = "*.json") -> Dict[str, int]:
    """Count files grouped by parent directory name (chart type)."""
    counts: Dict[str, int] = defaultdict(int)
    for f in directory.rglob(pattern):
        if f.is_file():
            chart_type = f.parent.name
            counts[chart_type] += 1
    return dict(sorted(counts.items()))


def _count_flat_json(directory: Path) -> Tuple[int, Dict[str, int]]:
    """Count JSON files and classify by chart_type field inside them."""
    total = 0
    by_type: Dict[str, int] = defaultdict(int)
    for f in directory.rglob("*.json"):
        if not f.is_file():
            continue
        total += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ct = data.get("chart_type", f.parent.name)
            by_type[ct] += 1
        except Exception:
            by_type["_parse_error"] += 1
    return total, dict(sorted(by_type.items()))


def _scan_academic_dataset(data_dir: Path) -> Dict[str, SplitStats]:
    """Scan academic dataset splits."""
    splits = {}
    academic = data_dir / "academic_dataset"
    if not academic.exists():
        return splits

    for split_name in ("train", "val", "test"):
        split_dir = academic / split_name
        if split_dir.exists():
            by_type = _count_by_type(split_dir, "*.png")
            # Also count jpg
            for ext in ("*.jpg", "*.jpeg"):
                for f in split_dir.rglob(ext):
                    if f.is_file():
                        chart_type = f.parent.name
                        by_type[chart_type] = by_type.get(chart_type, 0) + 1
            total = sum(by_type.values())
            splits[split_name] = SplitStats(count=total, by_type=by_type)

    return splits


def _scan_qa_pairs(data_dir: Path) -> QAPairStats:
    """Scan chart QA pair datasets."""
    qa_dirs = [
        data_dir / "academic_dataset" / "chart_qa_v2" / "generated",
        data_dir / "academic_dataset" / "chart_qa_v2",
    ]

    total = 0
    by_type: Dict[str, int] = defaultdict(int)

    for qa_dir in qa_dirs:
        if not qa_dir.exists():
            continue
        for f in qa_dir.rglob("*.json"):
            if not f.is_file():
                continue
            total += 1
            by_type[f.parent.name] += 1

    return QAPairStats(
        total=total,
        by_type=dict(sorted(by_type.items())),
    )


def _scan_slm_training(data_dir: Path) -> SLMTrainingStats:
    """Scan SLM training datasets (v2, v3)."""
    # Check v3 first, then v2
    for version_dir, version in [
        (data_dir / "slm_training_v3", "v3"),
        (data_dir / "slm_training_v2", "v2"),
        (data_dir / "slm_training", "v1"),
    ]:
        if not version_dir.exists():
            continue

        total = 0
        by_type: Dict[str, int] = defaultdict(int)

        # Check for pre-computed stats file first (fast path)
        stats_file = version_dir / "_stats.json"
        if stats_file.exists():
            try:
                stats = json.loads(stats_file.read_text(encoding="utf-8"))
                return SLMTrainingStats(
                    total=stats.get("total", 0),
                    by_type=stats.get("by_type", {}),
                    version=version,
                )
            except Exception:
                pass

        # Count records without loading entire files into memory
        for split_file in version_dir.glob("*.json"):
            if split_file.name.startswith("_"):
                continue
            try:
                # Stream-count: read line by line looking for "chart_type" fields
                # instead of parsing entire multi-hundred-MB JSON arrays
                file_size_mb = split_file.stat().st_size / (1024 * 1024)
                if file_size_mb > 50:
                    # Large file: estimate from file size or count array entries
                    logger.info(f"  Large file {split_file.name} ({file_size_mb:.0f}MB), "
                                "using fast count...")
                    with open(split_file, "r", encoding="utf-8") as f:
                        # Read first 1KB to check format
                        header = f.read(1024)
                        if header.strip().startswith("["):
                            # JSON array: count "conversations" keys as proxy
                            f.seek(0)
                            count = 0
                            for line in f:
                                count += line.count('"conversations"')
                            total += count
                            by_type["_estimated"] = by_type.get("_estimated", 0) + count
                else:
                    records = json.loads(split_file.read_text(encoding="utf-8"))
                    if isinstance(records, list):
                        for record in records:
                            total += 1
                            ct = record.get("metadata", {}).get("chart_type", "unknown")
                            by_type[ct] += 1
            except Exception as e:
                logger.warning(f"Failed to read {split_file.name}: {e}")

        if total > 0:
            return SLMTrainingStats(
                total=total,
                by_type=dict(sorted(by_type.items())),
                version=version,
            )
            try:
                records = json.loads(split_file.read_text(encoding="utf-8"))
                if isinstance(records, list):
                    for record in records:
                        total += 1
                        ct = record.get("metadata", {}).get("chart_type", "unknown")
                        by_type[ct] += 1
            except Exception:
                pass

        if total > 0:
            return SLMTrainingStats(
                total=total,
                by_type=dict(sorted(by_type.items())),
                version=version,
            )

    return SLMTrainingStats()


def _scan_benchmark(data_dir: Path) -> BenchmarkStats:
    """Scan benchmark dataset."""
    bench_dir = data_dir / "benchmark"
    if not bench_dir.exists():
        return BenchmarkStats()

    images_dir = bench_dir / "images"
    annotations_dir = bench_dir / "annotations"

    n_images = 0
    n_annotations = 0
    by_type: Dict[str, int] = defaultdict(int)

    if images_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            n_images += len(list(images_dir.glob(ext)))

    if annotations_dir.exists():
        for f in annotations_dir.glob("*.json"):
            n_annotations += 1
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                ct = data.get("chart_type", "unknown")
                by_type[ct] += 1
            except Exception:
                by_type["_parse_error"] += 1

    return BenchmarkStats(
        total_annotated=n_annotations,
        by_type=dict(sorted(by_type.items())),
        total_images=n_images,
    )


def _scan_stage3_features(data_dir: Path) -> Dict[str, Any]:
    """Scan Stage 3 feature extractions."""
    features_dir = data_dir / "academic_dataset" / "stage3_features"
    if not features_dir.exists():
        return {"total": 0, "by_type": {}}

    total, by_type = _count_flat_json(features_dir)
    return {"total": total, "by_type": by_type}


def _check_integrity(data_dir: Path) -> IntegrityReport:
    """Check data integrity: broken images, missing labels."""
    broken = 0
    details: List[str] = []

    try:
        from PIL import Image
    except ImportError:
        return IntegrityReport(details=["PIL not available, skipping image checks"])

    # Check benchmark images
    images_dir = data_dir / "benchmark" / "images"
    if images_dir.exists():
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    broken += 1
                    details.append(f"Broken image: {img_path.name} ({e})")

    # Check for annotation/image mismatch
    annotations_dir = data_dir / "benchmark" / "annotations"
    missing_labels = 0
    if images_dir.exists() and annotations_dir.exists():
        image_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        annotation_stems = {p.stem for p in annotations_dir.glob("*.json")}
        orphaned = image_stems - annotation_stems
        missing_labels = len(orphaned)
        if orphaned and len(orphaned) <= 10:
            details.append(f"Images without annotations: {sorted(orphaned)}")
        elif orphaned:
            details.append(f"Images without annotations: {len(orphaned)} files")

    return IntegrityReport(
        broken_images=broken,
        missing_labels=missing_labels,
        details=details,
    )


def _compute_diff(current: DataManifest, previous_path: Path) -> Optional[DiffReport]:
    """Compare current manifest with a previous one."""
    if not previous_path.exists():
        return None

    try:
        prev_data = json.loads(previous_path.read_text(encoding="utf-8"))
        prev_total = prev_data.get("total_chart_images", 0)
        curr_total = current.total_chart_images
        return DiffReport(
            added=max(0, curr_total - prev_total),
            removed=max(0, prev_total - curr_total),
            previous_manifest=str(previous_path),
        )
    except Exception:
        return None


def run_audit(
    data_dir: Path,
    check_integrity: bool = False,
    compare_path: Optional[Path] = None,
) -> DataManifest:
    """Run full data audit and return manifest."""
    logger.info(f"Starting data audit | data_dir={data_dir}")

    manifest = DataManifest(project_root=str(data_dir.parent))

    # Scan all data sections
    logger.info("Scanning academic dataset splits...")
    manifest.splits = _scan_academic_dataset(data_dir)

    logger.info("Scanning QA pairs...")
    manifest.qa_pairs = _scan_qa_pairs(data_dir)

    logger.info("Scanning SLM training data...")
    manifest.slm_training = _scan_slm_training(data_dir)

    logger.info("Scanning benchmark data...")
    manifest.benchmark = _scan_benchmark(data_dir)

    logger.info("Scanning Stage 3 features...")
    manifest.stage3_features = _scan_stage3_features(data_dir)

    # Compute totals
    manifest.total_chart_images = sum(s.count for s in manifest.splits.values())
    manifest.total_annotations = manifest.benchmark.total_annotated

    # Integrity check (optional, slower)
    if check_integrity:
        logger.info("Running integrity checks...")
        manifest.integrity = _check_integrity(data_dir)
    else:
        logger.info("Skipping integrity checks (use --check-integrity to enable)")

    # Diff with previous (optional)
    if compare_path:
        manifest.diff_from_previous = _compute_diff(manifest, compare_path)

    return manifest


def print_summary(manifest: DataManifest) -> None:
    """Print human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  DATA AUDIT MANIFEST")
    print(f"  Generated: {manifest.generated_at}")
    print(f"{'='*60}")

    print(f"\n-- Academic Dataset Splits --")
    for split_name, stats in manifest.splits.items():
        print(f"  {split_name:>8}: {stats.count:>8,} images")
        for ct, n in sorted(stats.by_type.items()):
            print(f"           {ct}: {n:,}")

    print(f"\n-- QA Pairs --")
    print(f"  Total: {manifest.qa_pairs.total:,}")

    print(f"\n-- SLM Training ({manifest.slm_training.version}) --")
    print(f"  Total: {manifest.slm_training.total:,}")
    for ct, n in sorted(manifest.slm_training.by_type.items()):
        print(f"    {ct}: {n:,}")

    print(f"\n-- Benchmark --")
    print(f"  Images: {manifest.benchmark.total_images}")
    print(f"  Annotations: {manifest.benchmark.total_annotated}")

    print(f"\n-- Stage 3 Features --")
    print(f"  Total: {manifest.stage3_features.get('total', 0):,}")

    if manifest.integrity.broken_images or manifest.integrity.missing_labels:
        print(f"\n-- Integrity Issues --")
        print(f"  Broken images: {manifest.integrity.broken_images}")
        print(f"  Missing labels: {manifest.integrity.missing_labels}")
        for detail in manifest.integrity.details[:5]:
            print(f"  {detail}")

    if manifest.diff_from_previous:
        print(f"\n-- Diff from Previous --")
        print(f"  Added: {manifest.diff_from_previous.added}")
        print(f"  Removed: {manifest.diff_from_previous.removed}")

    print(f"\n{'='*60}")
    print(f"  TOTALS: {manifest.total_chart_images:,} chart images | "
          f"{manifest.total_annotations} benchmark annotations")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data Audit: scan data/ and generate manifest JSON",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "data_manifest.json",
        help="Output manifest path (default: output/data_manifest.json)",
    )
    parser.add_argument(
        "--check-integrity",
        action="store_true",
        help="Run image integrity checks (slower)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Path to previous manifest for diff comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = run_audit(
        data_dir=args.data_dir,
        check_integrity=args.check_integrity,
        compare_path=args.compare,
    )

    print_summary(manifest)

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Manifest saved to {args.output}")

    # Also save timestamped copy
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = args.output.parent / f"data_manifest_{ts}.json"
    ts_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Timestamped copy saved to {ts_path}")


if __name__ == "__main__":
    main()
