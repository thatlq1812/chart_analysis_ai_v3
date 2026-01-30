#!/usr/bin/env python3
"""
Batch OCR Extraction with PaddleOCR GPU

This script runs in a separate Python environment with PaddlePaddle GPU
to avoid conflicts with PyTorch CUDA in the main environment.

Usage:
    .venv_paddle/Scripts/python.exe scripts/batch_ocr_gpu.py --input-dir data/academic_dataset/classified_charts
    .venv_paddle/Scripts/python.exe scripts/batch_ocr_gpu.py --input-dir data/academic_dataset/classified_charts --batch-size 32

Output:
    data/cache/ocr_cache.json - Cached OCR results for all images
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def init_paddleocr(use_gpu: bool = True, lang: str = "en"):
    """Initialize PaddleOCR with GPU support."""
    from paddleocr import PaddleOCR
    
    logger.info(f"Initializing PaddleOCR | use_gpu={use_gpu} | lang={lang}")
    
    ocr = PaddleOCR(
        use_angle_cls=False,  # Faster without angle classification
        lang=lang,
        use_gpu=use_gpu,
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        rec_batch_num=16,  # Batch recognition for speed
    )
    
    return ocr


def extract_ocr(ocr, image_path: Path) -> Dict[str, Any]:
    """
    Extract OCR from single image.
    
    Returns dict with:
        - texts: List of {text, bbox, confidence}
        - processing_time: float
        - error: Optional error message
    """
    start_time = time.time()
    
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                "texts": [],
                "processing_time": 0,
                "error": f"Failed to load image: {image_path}",
            }
        
        # Run OCR
        result = ocr.ocr(image, cls=False)
        
        texts = []
        if result and result[0]:
            for line in result[0]:
                bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]    # (text, confidence)
                
                # Convert polygon to bounding box
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                
                texts.append({
                    "text": text_info[0],
                    "confidence": float(text_info[1]),
                    "bbox": {
                        "x_min": int(min(xs)),
                        "y_min": int(min(ys)),
                        "x_max": int(max(xs)),
                        "y_max": int(max(ys)),
                    },
                    "polygon": [[int(p[0]), int(p[1])] for p in bbox_points],
                })
        
        processing_time = time.time() - start_time
        
        return {
            "texts": texts,
            "processing_time": round(processing_time, 4),
            "error": None,
        }
    
    except Exception as e:
        return {
            "texts": [],
            "processing_time": time.time() - start_time,
            "error": str(e),
        }


def get_relative_path(image_path: Path, base_dir: Path) -> str:
    """Get relative path for use as cache key."""
    try:
        return str(image_path.relative_to(base_dir))
    except ValueError:
        return str(image_path)


def load_existing_cache(cache_path: Path) -> Dict[str, Any]:
    """Load existing cache if available."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded existing cache | entries={len(data.get('results', {}))}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return {"results": {}, "metadata": {}}


def save_cache(cache_path: Path, cache_data: Dict[str, Any]):
    """Save cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def collect_images(input_dir: Path, extensions: List[str] = None) -> List[Path]:
    """Collect all image files from directory."""
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg"]
    
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
    
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="Batch OCR extraction with PaddleOCR GPU")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "ocr_cache.json",
        help="Output cache file path",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="OCR language (en, ch, etc.)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU (use CPU)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images already in cache",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save cache every N images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images (0 = no limit)",
    )
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Collect images
    logger.info(f"Scanning for images in: {args.input_dir}")
    images = collect_images(args.input_dir)
    logger.info(f"Found {len(images)} images")
    
    if args.limit > 0:
        images = images[:args.limit]
        logger.info(f"Limited to {len(images)} images")
    
    if not images:
        logger.warning("No images found!")
        sys.exit(0)
    
    # Load existing cache
    cache_data = load_existing_cache(args.output)
    results = cache_data.get("results", {})
    
    # Filter out already processed
    if args.skip_existing:
        to_process = []
        for img_path in images:
            key = get_relative_path(img_path, args.input_dir)
            if key not in results:
                to_process.append(img_path)
        
        logger.info(f"Already cached: {len(images) - len(to_process)}, to process: {len(to_process)}")
        images = to_process
    
    if not images:
        logger.info("All images already cached!")
        return
    
    # Initialize OCR
    use_gpu = not args.no_gpu
    ocr = init_paddleocr(use_gpu=use_gpu, lang=args.lang)
    
    # Warmup
    logger.info("Warming up OCR engine...")
    if images:
        _ = extract_ocr(ocr, images[0])
    
    # Process images
    logger.info(f"Processing {len(images)} images...")
    start_time = time.time()
    processed = 0
    errors = 0
    
    try:
        for i, img_path in enumerate(tqdm(images, desc="OCR Extraction")):
            # Extract OCR
            result = extract_ocr(ocr, img_path)
            
            # Store result
            key = get_relative_path(img_path, args.input_dir)
            results[key] = result
            
            if result.get("error"):
                errors += 1
            
            processed += 1
            
            # Periodic save
            if processed % args.save_interval == 0:
                cache_data["results"] = results
                cache_data["metadata"] = {
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_entries": len(results),
                    "ocr_engine": "paddleocr",
                    "use_gpu": use_gpu,
                    "lang": args.lang,
                }
                save_cache(args.output, cache_data)
                tqdm.write(f"Checkpoint saved: {len(results)} entries")
    
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    
    finally:
        # Final save
        cache_data["results"] = results
        cache_data["metadata"] = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_entries": len(results),
            "ocr_engine": "paddleocr",
            "use_gpu": use_gpu,
            "lang": args.lang,
        }
        save_cache(args.output, cache_data)
    
    # Stats
    total_time = time.time() - start_time
    avg_time = total_time / processed if processed > 0 else 0
    
    logger.info("=" * 50)
    logger.info("OCR EXTRACTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Processed: {processed} images")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Average: {avg_time*1000:.1f}ms/image")
    logger.info(f"Cache saved to: {args.output}")
    logger.info(f"Total cache entries: {len(results)}")


if __name__ == "__main__":
    main()
