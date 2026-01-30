#!/usr/bin/env python3
"""
Batch extract Stage 3 features from classified charts.

This script runs Stage 3 extraction on all charts in classified_charts/
and saves the extracted metadata as JSON files for SLM training.

Usage:
    python scripts/extract_stage3_features.py
    python scripts/extract_stage3_features.py --limit 100  # Test run
    python scripts/extract_stage3_features.py --chart-type bar  # Only bar charts

Output:
    data/academic_dataset/stage3_features/{chart_type}/{image_id}.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_stage3():
    """Initialize Stage 3 with optimized config."""
    from src.core_engine.stages.s3_extraction import (
        Stage3Extraction,
        ExtractionConfig,
    )
    
    config = ExtractionConfig(
        use_resnet_classifier=True,
        use_ml_classifier=False,  # ResNet is better
        enable_ocr=True,
        enable_element_detection=True,
        enable_vectorization=False,  # Skip for speed
        ocr_engine="easyocr",
    )
    
    return Stage3Extraction(config)


def extract_single_chart(stage3, image_path: Path) -> dict:
    """
    Extract features from a single chart image.
    
    Returns:
        Dict with extracted metadata
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"Failed to load image: {image_path}"}
    
    try:
        metadata = stage3.process_image(img, chart_id=image_path.stem)
        
        # Convert to serializable dict
        result = {
            "image_id": metadata.chart_id,
            "image_path": str(image_path),
            "chart_type": metadata.chart_type.value,
            "confidence": {
                "classification": metadata.confidence.classification_confidence,
                "ocr": metadata.confidence.ocr_mean_confidence,
                "overall": metadata.confidence.overall_confidence,
            },
            "texts": [
                {
                    "text": t.text,
                    "role": t.role,
                    "confidence": t.confidence,
                    "bbox": {
                        "x_min": t.bbox.x_min,
                        "y_min": t.bbox.y_min,
                        "x_max": t.bbox.x_max,
                        "y_max": t.bbox.y_max,
                    },
                }
                for t in metadata.texts
            ],
            "elements": [
                {
                    "type": e.element_type,
                    "center": {"x": e.center.x, "y": e.center.y},
                    "color": {"r": e.color.r, "g": e.color.g, "b": e.color.b}
                    if e.color else None,
                    "area_pixels": e.area_pixels,
                }
                for e in metadata.elements
            ],
            "axis_info": None,  # Can be added if needed
            "extracted_at": datetime.now().isoformat(),
        }
        
        # Add axis info if available
        if metadata.axis_info:
            result["axis_info"] = {
                "x_axis_detected": metadata.axis_info.x_axis_detected,
                "y_axis_detected": metadata.axis_info.y_axis_detected,
                "x_range": [metadata.axis_info.x_min, metadata.axis_info.x_max],
                "y_range": [metadata.axis_info.y_min, metadata.axis_info.y_max],
            }
        
        return result
        
    except Exception as e:
        return {
            "image_id": image_path.stem,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Extract Stage 3 features from charts")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/academic_dataset/classified_charts"),
        help="Input directory with classified charts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/academic_dataset/stage3_features"),
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--chart-type",
        type=str,
        default=None,
        help="Process only specific chart type (e.g., 'bar', 'line')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have extracted features",
    )
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get chart types to process
    if args.chart_type:
        chart_types = [args.chart_type]
    else:
        chart_types = [
            d.name for d in args.input_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    
    logger.info(f"Chart types to process: {chart_types}")
    
    # Initialize Stage 3
    logger.info("Initializing Stage 3 extraction...")
    stage3 = setup_stage3()
    
    # Collect all images
    all_images = []
    for chart_type in chart_types:
        type_dir = args.input_dir / chart_type
        if not type_dir.exists():
            logger.warning(f"Directory not found: {type_dir}")
            continue
        
        images = list(type_dir.glob("*.png")) + list(type_dir.glob("*.jpg"))
        all_images.extend([(chart_type, img) for img in images])
    
    logger.info(f"Total images found: {len(all_images)}")
    
    # Apply limit if specified
    if args.limit:
        all_images = all_images[:args.limit]
        logger.info(f"Limited to {len(all_images)} images")
    
    # Process images
    stats = {
        "total": len(all_images),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "by_type": {},
    }
    
    start_time = time.time()
    
    for chart_type, image_path in tqdm(all_images, desc="Extracting features"):
        # Prepare output path
        output_type_dir = args.output_dir / chart_type
        output_type_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_type_dir / f"{image_path.stem}.json"
        
        # Skip if exists
        if args.skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue
        
        # Extract features
        result = extract_single_chart(stage3, image_path)
        
        # Track stats
        if "error" in result:
            stats["errors"] += 1
            logger.debug(f"Error: {result['error']} | {image_path.name}")
        else:
            stats["processed"] += 1
            stats["by_type"][chart_type] = stats["by_type"].get(chart_type, 0) + 1
        
        # Save result
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images: {stats['total']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped (existing): {stats['skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Time elapsed: {elapsed:.1f}s ({elapsed/max(1,stats['processed']):.2f}s/image)")
    logger.info(f"Output directory: {args.output_dir}")
    
    if stats["by_type"]:
        logger.info("\nBy chart type:")
        for ct, count in sorted(stats["by_type"].items()):
            logger.info(f"  {ct}: {count}")
    
    # Save summary
    summary_path = args.output_dir / "extraction_summary.json"
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "stats": stats,
        "elapsed_seconds": elapsed,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
