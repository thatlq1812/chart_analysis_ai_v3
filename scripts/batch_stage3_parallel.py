#!/usr/bin/env python3
"""
Batch Stage 3 Extraction - PARALLEL VERSION

Extract structural features using multiprocessing.
Key optimizations:
- Skip ResNet classifier (use folder name as chart_type)
- Load OCR cache once per worker (not per image)
- Pure CPU processing = perfect parallelism

Usage:
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --workers 8
    .venv/Scripts/python.exe scripts/batch_stage3_parallel.py --workers 16 --chart-type line

Output:
    data/academic_dataset/stage3_features/{chart_type}/{image_name}.json
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import functools

# Suppress sklearn warnings (spam during kmeans clustering)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Setup logging - minimal for parallel
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Global worker state (initialized once per worker process)
_worker_stage3 = None
_worker_id = None


def init_worker(worker_id: int, ocr_cache_file: Path, cache_base_dir: Path):
    """Initialize worker process with Stage3Extraction."""
    global _worker_stage3, _worker_id
    _worker_id = worker_id
    
    # Import here to avoid issues with multiprocessing
    from core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig
    
    # Configure WITHOUT any classifier (use folder name for chart_type)
    # FAST MODE: Disable slow skeletonization for now
    config = ExtractionConfig(
        ocr_engine="paddleocr",
        use_resnet_classifier=False,  # SKIP ResNet - already classified!
        use_ml_classifier=False,       # SKIP Random Forest - already classified!
        enable_classification=False,   # SKIP rule-based classifier too
        enable_vectorization=False,    # FAST: Disable skeletonization (very slow)
        enable_element_detection=True,
        enable_ocr=True,
    )
    config.ocr.engine = "paddleocr"
    config.ocr.use_gpu = True  # Use CUDA for faster OCR
    config.ocr.use_cache = True
    config.ocr.cache_file = ocr_cache_file
    config.ocr.cache_base_dir = cache_base_dir
    
    _worker_stage3 = Stage3Extraction(config)
    
    # Force load OCR cache immediately (avoid lazy loading race conditions)
    if hasattr(_worker_stage3, 'ocr_engine') and _worker_stage3.ocr_engine:
        _worker_stage3.ocr_engine._load_cache()
    
    print(f"Worker {worker_id} initialized", flush=True)


def process_single_image(task: Tuple[Path, Path, str]) -> Dict[str, Any]:
    """
    Process a single image in worker process.
    
    Args:
        task: (image_path, output_path, chart_type_from_folder)
    """
    global _worker_stage3
    
    image_path, output_path, chart_type = task
    
    try:
        # Skip if already processed
        if output_path.exists():
            return {"status": "skipped", "path": str(image_path)}
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"status": "error", "path": str(image_path), "error": "Failed to load"}
        
        # Process
        chart_id = image_path.stem
        result = _worker_stage3.process_image(
            image, 
            chart_id=chart_id, 
            image_path=image_path
        )
        
        # Use folder name as chart_type (already classified!)
        output_data = {
            "chart_id": result.chart_id,
            "chart_type": chart_type,  # From folder, not classifier
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "texts": [
                {
                    "text": t.text,
                    "role": t.role.value if hasattr(t.role, 'value') else str(t.role) if t.role else None,
                    "confidence": t.confidence,
                    "bbox": {
                        "x_min": t.bbox.x_min, "y_min": t.bbox.y_min,
                        "x_max": t.bbox.x_max, "y_max": t.bbox.y_max
                    } if t.bbox else None,
                }
                for t in result.texts
            ],
            "elements": [
                {
                    "element_type": e.element_type.value if hasattr(e.element_type, 'value') else str(e.element_type) if e.element_type else None,
                    "bbox": {
                        "x_min": e.bbox.x_min, "y_min": e.bbox.y_min,
                        "x_max": e.bbox.x_max, "y_max": e.bbox.y_max
                    } if e.bbox else None,
                    "center": {"x": e.center.x, "y": e.center.y} if e.center else None,
                    "color": {"r": e.color.r, "g": e.color.g, "b": e.color.b} if e.color else None,
                    "area_pixels": e.area_pixels,
                }
                for e in result.elements
            ],
            "axis_info": result.axis_info.model_dump() if result.axis_info else None,
            "confidence": result.confidence.model_dump() if result.confidence else None,
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "path": str(image_path),
            "texts": len(output_data["texts"]),
            "elements": len(output_data["elements"]),
        }
        
    except Exception as e:
        return {"status": "error", "path": str(image_path), "error": str(e)}


def collect_images(input_dir: Path, chart_type: Optional[str] = None) -> List[Tuple[Path, str]]:
    """Collect all classified chart images with their types."""
    images = []
    
    excluded = {"uncertain", "not_a_chart", "other", "diagram", "table"}
    
    if chart_type:
        type_dir = input_dir / chart_type
        if type_dir.exists():
            for img in type_dir.glob("*.png"):
                images.append((img, chart_type))
            for img in type_dir.glob("*.jpg"):
                images.append((img, chart_type))
    else:
        for type_dir in input_dir.iterdir():
            if type_dir.is_dir() and type_dir.name not in excluded:
                ct = type_dir.name
                for img in type_dir.glob("*.png"):
                    images.append((img, ct))
                for img in type_dir.glob("*.jpg"):
                    images.append((img, ct))
    
    return sorted(images, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Parallel Stage 3 extraction")
    parser.add_argument(
        "--input-dir", type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "stage3_features",
    )
    parser.add_argument("--chart-type", type=str, help="Process only specific chart type")
    parser.add_argument("--limit", type=int, default=0, help="Limit images (0=no limit)")
    parser.add_argument(
        "--workers", type=int, default=8,
        help=f"Number of parallel workers (default: 8, max: {cpu_count()})"
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()
    
    # Paths
    ocr_cache_file = PROJECT_ROOT / "data" / "cache" / "ocr_cache.json"
    cache_base_dir = args.input_dir
    
    if not ocr_cache_file.exists():
        print(f"ERROR: OCR cache not found: {ocr_cache_file}")
        sys.exit(1)
    
    # Collect images
    print(f"Scanning: {args.input_dir}")
    images_with_types = collect_images(args.input_dir, args.chart_type)
    print(f"Found {len(images_with_types)} images")
    
    if args.limit > 0:
        images_with_types = images_with_types[:args.limit]
        print(f"Limited to {len(images_with_types)} images")
    
    # Prepare tasks
    tasks = []
    for img_path, chart_type in images_with_types:
        output_path = args.output_dir / chart_type / f"{img_path.stem}.json"
        if args.skip_existing and output_path.exists():
            continue
        tasks.append((img_path, output_path, chart_type))
    
    skipped = len(images_with_types) - len(tasks)
    print(f"Tasks: {len(tasks)} (skipped {skipped} existing)")
    
    if not tasks:
        print("All images already processed!")
        return
    
    # Limit workers
    num_workers = min(args.workers, cpu_count(), len(tasks))
    print(f"Using {num_workers} workers (CPU cores: {cpu_count()})")
    
    # Process with multiprocessing pool
    start_time = time.time()
    success = 0
    errors = 0
    error_list = []
    
    # Create pool with initializer
    print("Initializing workers (loading OCR cache in each)...")
    
    # Use imap for progress tracking
    from tqdm import tqdm
    
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(0, ocr_cache_file, cache_base_dir),
    ) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, tasks, chunksize=10),
            total=len(tasks),
            desc="Extracting"
        ))
    
    # Count results
    for r in results:
        if r["status"] == "success":
            success += 1
        elif r["status"] == "error":
            errors += 1
            error_list.append(f"{r['path']}: {r.get('error', 'unknown')}")
    
    # Summary
    total_time = time.time() - start_time
    print("=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Success: {success}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Throughput: {len(tasks)/total_time:.1f} img/s")
    print(f"Output: {args.output_dir}")
    
    if error_list and len(error_list) <= 20:
        print("\nErrors:")
        for e in error_list:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
