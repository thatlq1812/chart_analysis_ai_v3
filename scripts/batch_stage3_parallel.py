#!/usr/bin/env python3
"""
Batch Stage 3 Extraction - PARALLEL VERSION (CUDA-aware)

Extract structural features (OCR, elements, axis calibration) using
multiprocessing with CUDA acceleration option for PaddleOCR.

Design:
- Uses 'spawn' start method (required for CUDA safety on Windows)
- Each worker loads OCR cache independently (read-only, 588MB)
- GPU mode: 1-4 workers recommended (VRAM shared)
- CPU mode: use cpu_count() workers
- Skip-existing always on (safe restart after interruption)
- Processes chart types smallest-count-first for visible early progress

Usage:
    # GPU accelerated (recommended): 2 workers with CUDA OCR
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --gpu-workers 2

    # CPU-only run
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --workers 8 --no-gpu

    # Single chart type
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --chart-type bar --gpu-workers 2

    # Check progress without running
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --status

    # Small test run
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --chart-type area --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path BEFORE multiprocessing import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Valid chart types (excludes not_a_chart, diagram, table, uncertain, other)
VALID_CHART_TYPES = {"area", "bar", "box", "heatmap", "histogram", "line", "pie", "scatter"}

# Smallest-count types first → fast early progress, then scale up
CHART_TYPE_ORDER = ["area", "box", "pie", "histogram", "heatmap", "bar", "scatter", "line"]


# ─────────────────────────────────────────────────────────────────────────────
# Worker state (initialized once per spawned process)
# ─────────────────────────────────────────────────────────────────────────────

_worker_stage3 = None
_worker_id: int = -1


def init_worker(worker_id: int, ocr_cache_file: Path, use_gpu: bool) -> None:
    """
    Initialize a worker process with Stage3Extraction.

    Called once per worker when the Pool is created. Loads PaddleOCR model
    and the 588MB OCR cache. Subsequent task() calls are fast.

    Args:
        worker_id: Unique ID for logging
        ocr_cache_file: Path to pre-built OCR cache
        use_gpu: Whether to use CUDA for PaddleOCR
    """
    global _worker_stage3, _worker_id
    _worker_id = worker_id

    try:
        from core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig

        config = ExtractionConfig(
            ocr_engine="paddleocr",
            use_resnet_classifier=False,    # Folder name IS the chart type
            use_ml_classifier=False,
            enable_classification=False,    # Skip all classifiers
            enable_vectorization=False,     # Skeletonization disabled (slow, not needed for data)
            enable_element_detection=True,
            enable_ocr=True,
        )
        config.ocr.engine = "paddleocr"
        config.ocr.use_gpu = use_gpu
        config.ocr.use_cache = True
        config.ocr.cache_file = ocr_cache_file
        config.ocr.cache_base_dir = (
            PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts"
        )

        _worker_stage3 = Stage3Extraction(config)

        # Force-load OCR cache immediately to avoid lazy-load race conditions
        if hasattr(_worker_stage3, "ocr_engine") and _worker_stage3.ocr_engine:
            if hasattr(_worker_stage3.ocr_engine, "_load_cache"):
                _worker_stage3.ocr_engine._load_cache()

        gpu_tag = "GPU" if use_gpu else "CPU"
        print(f"[Worker {worker_id}] Ready ({gpu_tag})", flush=True)

    except Exception as exc:
        print(f"[Worker {worker_id}] INIT FAILED: {exc}", flush=True)
        traceback.print_exc()
        _worker_stage3 = None


def process_single_image(task: Tuple[Path, Path, str]) -> Dict[str, Any]:
    """
    Process one image in a worker process.

    Args:
        task: (image_path, output_path, chart_type)

    Returns:
        Dict with keys: status, path, texts, elements, has_axis, error
    """
    import cv2

    global _worker_stage3, _worker_id

    image_path, output_path, chart_type = task

    if output_path.exists():
        return {"status": "skipped", "path": str(image_path)}

    if _worker_stage3 is None:
        return {
            "status": "error",
            "path": str(image_path),
            "error": "worker not initialized",
        }

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                "status": "error",
                "path": str(image_path),
                "error": "cv2.imread returned None",
            }

        chart_id = image_path.stem
        result = _worker_stage3.process_image(
            image,
            chart_id=chart_id,
            image_path=image_path,
        )

        def _role(v: Any) -> Optional[str]:
            if v is None:
                return None
            return v.value if hasattr(v, "value") else str(v)

        output_data: Dict[str, Any] = {
            "chart_id": result.chart_id,
            "chart_type": chart_type,
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "texts": [
                {
                    "text": t.text,
                    "role": _role(t.role),
                    "confidence": t.confidence,
                    "bbox": (
                        {
                            "x_min": t.bbox.x_min, "y_min": t.bbox.y_min,
                            "x_max": t.bbox.x_max, "y_max": t.bbox.y_max,
                        }
                        if t.bbox else None
                    ),
                }
                for t in result.texts
            ],
            "elements": [
                {
                    "element_type": _role(e.element_type),
                    "bbox": (
                        {
                            "x_min": e.bbox.x_min, "y_min": e.bbox.y_min,
                            "x_max": e.bbox.x_max, "y_max": e.bbox.y_max,
                        }
                        if e.bbox else None
                    ),
                    "center": (
                        {"x": e.center.x, "y": e.center.y} if e.center else None
                    ),
                    "color": (
                        {"r": e.color.r, "g": e.color.g, "b": e.color.b}
                        if e.color else None
                    ),
                    "area_pixels": e.area_pixels,
                }
                for e in result.elements
            ],
            # axis_info is now populated by the fixed process_image (gap 1 fix)
            "axis_info": result.axis_info.model_dump() if result.axis_info else None,
            "confidence": result.confidence.model_dump() if result.confidence else None,
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "path": str(image_path),
            "texts": len(output_data["texts"]),
            "elements": len(output_data["elements"]),
            "has_axis": output_data["axis_info"] is not None,
        }

    except Exception as exc:
        return {
            "status": "error",
            "path": str(image_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_tasks(
    input_dir: Path,
    output_dir: Path,
    chart_type: Optional[str],
    limit: int,
) -> List[Tuple[Path, Path, str]]:
    """Collect all pending (not yet extracted) images."""
    pending: List[Tuple[Path, Path, str]] = []
    types = [chart_type] if chart_type else CHART_TYPE_ORDER

    for ct in types:
        d = input_dir / ct
        if not d.exists():
            continue
        for img in sorted(d.glob("*.png")):
            out = output_dir / ct / f"{img.stem}.json"
            if not out.exists():
                pending.append((img, out, ct))
        for img in sorted(d.glob("*.jpg")):
            out = output_dir / ct / f"{img.stem}.json"
            if not out.exists():
                pending.append((img, out, ct))

    return pending[:limit] if limit > 0 else pending


def print_status(input_dir: Path, output_dir: Path) -> None:
    """Print per-type extraction progress."""
    print("=== Stage 3 Feature Extraction Status ===")
    total_done = total_avail = 0
    for ct in CHART_TYPE_ORDER:
        n_done = (
            len(list((output_dir / ct).glob("*.json")))
            if (output_dir / ct).exists() else 0
        )
        n_avail = (
            len(list((input_dir / ct).glob("*.png")))
            + len(list((input_dir / ct).glob("*.jpg")))
            if (input_dir / ct).exists() else 0
        )
        pct = f"{n_done / n_avail * 100:.1f}%" if n_avail > 0 else "n/a"
        bar = "#" * int(n_done / max(n_avail, 1) * 20)
        print(f"  {ct:12s} [{bar:<20s}] {n_done:5d}/{n_avail:5d} ({pct})")
        total_done += n_done
        total_avail += n_avail
    total_pct = f"{total_done / total_avail * 100:.1f}%" if total_avail > 0 else "n/a"
    print(f"  {'TOTAL':12s}  {'':22s}  {total_done:5d}/{total_avail:5d} ({total_pct})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import multiprocessing
    from multiprocessing import Pool, cpu_count

    # CRITICAL: spawn avoids CUDA context fork corruption on Windows
    # Must be set before Pool is created, but after __main__ guard
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Parallel Stage 3 feature extraction (CUDA-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "stage3_features",
    )
    parser.add_argument(
        "--chart-type", type=str, choices=list(VALID_CHART_TYPES),
        help="Process only this chart type",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of images (0 = no limit)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help=f"CPU workers (0 = auto min(8, cpu_count())). This machine: {cpu_count()} cores.",
    )
    parser.add_argument(
        "--gpu-workers", type=int, default=0,
        help="Number of workers with --gpu-workers=N enables CUDA for all N workers. "
             "Overrides --workers when > 0. Recommended: 1-2 for 6GB VRAM.",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable CUDA for all workers (CPU-only OCR)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print progress and exit (no processing)",
    )
    args = parser.parse_args()

    ocr_cache_file = PROJECT_ROOT / "data" / "cache" / "ocr_cache.json"
    if not ocr_cache_file.exists():
        print(f"ERROR: OCR cache not found: {ocr_cache_file}")
        sys.exit(1)

    if args.status:
        print_status(args.input_dir, args.output_dir)
        return

    tasks = collect_tasks(args.input_dir, args.output_dir, args.chart_type, args.limit)

    if not tasks:
        print("No pending images found.")
        print_status(args.input_dir, args.output_dir)
        return

    # Count by type
    type_counts: Dict[str, int] = {}
    for _, _, ct in tasks:
        type_counts[ct] = type_counts.get(ct, 0) + 1

    print(f"Pending: {len(tasks)} images")
    for ct in CHART_TYPE_ORDER:
        if ct in type_counts:
            print(f"  {ct}: {type_counts[ct]}")

    # Worker configuration
    use_gpu = not args.no_gpu
    if args.gpu_workers > 0:
        num_workers = args.gpu_workers
        use_gpu = True
    elif args.workers > 0:
        num_workers = args.workers
    else:
        num_workers = min(8, cpu_count())

    num_workers = min(num_workers, len(tasks))

    print(f"\nWorkers: {num_workers} | GPU: {'enabled' if use_gpu else 'disabled (CPU only)'}")
    if use_gpu and num_workers > 2:
        print(
            f"  Warning: {num_workers} GPU workers on 6GB VRAM may cause OOM. "
            "Consider --gpu-workers 2."
        )

    print(f"\nInitializing workers (each loads 588MB OCR cache)...")
    print("Expected init time: ~30-60s on first run.\n")

    start_time = time.time()
    success = errors = skipped = has_axis_count = 0
    error_examples: List[str] = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(0, ocr_cache_file, use_gpu),
    ) as pool:
        iter_ = pool.imap(process_single_image, tasks, chunksize=5)

        if use_tqdm:
            iter_ = tqdm(iter_, total=len(tasks), desc="Stage3", unit="img")

        for r in iter_:
            s = r.get("status")
            if s == "success":
                success += 1
                if r.get("has_axis"):
                    has_axis_count += 1
            elif s == "skipped":
                skipped += 1
            else:
                errors += 1
                if len(error_examples) < 10:
                    error_examples.append(f"{r['path']}: {r.get('error')}")

    elapsed = time.time() - start_time
    print()
    print("=" * 55)
    print("STAGE 3 EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"Extracted:         {success}")
    print(f"  with axis_info:  {has_axis_count} ({has_axis_count / max(success, 1) * 100:.0f}%)")
    print(f"Skipped existing:  {skipped}")
    print(f"Errors:            {errors}")
    print(f"Elapsed:           {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    if success > 0:
        print(f"Throughput:        {success / elapsed:.1f} img/s")
    print()
    print_status(args.input_dir, args.output_dir)

    if error_examples:
        print(f"\nSample errors ({len(error_examples)}):")
        for e in error_examples:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
