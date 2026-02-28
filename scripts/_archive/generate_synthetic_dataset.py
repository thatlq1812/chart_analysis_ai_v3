#!/usr/bin/env python3
"""
Generate Synthetic Document Dataset for Chart Detection

This script creates a training dataset by:
1. Taking chart images from ChartQA/academic datasets
2. Pasting them onto document-like backgrounds (text-only PDF pages)
3. Generating accurate YOLO bounding box labels
4. Including NEGATIVE SAMPLES (pages without charts)

CRITICAL IMPROVEMENTS:
- Grayscale output (color has no value for chart detection)
- Negative samples to prevent false positives on blank/text pages
- Proper bbox labels (not full-page boxes)
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from tools.data_factory.services.page_synthesizer import (
    PageSynthesizer,
    SynthesizerConfig,
    create_dataset_yaml,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic document pages with charts for YOLO training"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.20,
        help="Ratio of negative samples - pages WITHOUT charts (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--min-charts",
        type=int,
        default=1,
        help="Minimum charts per positive page (default: 1)",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=3,
        help="Maximum charts per positive page (default: 3)",
    )
    parser.add_argument(
        "--chart-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "images" / "huggingface",
        help="Directory containing chart images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "training",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic_source" / "backgrounds",
        help="Directory containing background images (from extract_backgrounds.py)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Train/val split ratio (default: 0.85)",
    )
    parser.add_argument(
        "--page-width",
        type=int,
        default=1654,
        help="Page width (default: 1654 = A4 at 150 DPI)",
    )
    parser.add_argument(
        "--page-height",
        type=int,
        default=2339,
        help="Page height (default: 2339 = A4 at 150 DPI)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        default=True,
        help="Output grayscale images (default: True, recommended)",
    )
    parser.add_argument(
        "--no-grayscale",
        action="store_false",
        dest="grayscale",
        help="Output RGB images instead of grayscale",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean output directory before generating",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Synthetic Document Dataset Generator")
    logger.info("=" * 60)

    # Validate chart directory
    if not args.chart_dir.exists():
        logger.error(f"Chart directory not found | path={args.chart_dir}")
        logger.info("Looking for alternative chart directories...")
        
        # Try alternative paths
        alternatives = [
            PROJECT_ROOT / "data" / "academic_dataset" / "images",
            PROJECT_ROOT / "data" / "academic_dataset" / "images" / "arxiv",
        ]
        for alt in alternatives:
            if alt.exists():
                args.chart_dir = alt
                logger.info(f"Using alternative chart dir | path={alt}")
                break
        else:
            logger.error("No chart images found. Run HuggingFace download first.")
            sys.exit(1)

    # Count charts
    chart_count = sum(
        1 for ext in [".png", ".jpg", ".jpeg"]
        for _ in args.chart_dir.glob(f"**/*{ext}")
    )
    logger.info(f"Chart images found | count={chart_count}")

    if chart_count == 0:
        logger.error("No chart images found!")
        sys.exit(1)

    # Check backgrounds
    bg_count = 0
    if args.background_dir.exists():
        bg_count = sum(
            1 for ext in [".png", ".jpg"]
            for _ in args.background_dir.glob(f"*{ext}")
        )

    if bg_count == 0:
        logger.error("No background images found!")
        logger.error("Run extract_backgrounds.py first:")
        logger.error("  python scripts/extract_backgrounds.py --max-backgrounds 2000")
        sys.exit(1)

    logger.info(f"Background images found | count={bg_count}")

    # Clean if requested
    if args.clean and args.output_dir.exists():
        logger.warning(f"Cleaning output directory | path={args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Create config
    config = SynthesizerConfig(
        min_charts_per_page=args.min_charts,
        max_charts_per_page=args.max_charts,
        page_width=args.page_width,
        page_height=args.page_height,
        negative_sample_ratio=args.negative_ratio,
        use_grayscale=args.grayscale,
        min_chart_width_ratio=0.30,
        max_chart_width_ratio=0.85,
        margin_top=60,
        margin_bottom=60,
        margin_left=50,
        margin_right=50,
        apply_noise=True,
        noise_intensity=0.015,
        random_brightness=True,
    )

    logger.info("Configuration:")
    logger.info(f"  Total samples:     {args.num_samples}")
    logger.info(f"  Negative ratio:    {args.negative_ratio:.0%}")
    logger.info(f"  Charts per page:   {config.min_charts_per_page}-{config.max_charts_per_page}")
    logger.info(f"  Page size:         {config.page_width}x{config.page_height}")
    logger.info(f"  Grayscale:         {config.use_grayscale}")
    logger.info(f"  Train/Val ratio:   {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    logger.info("-" * 60)

    # Initialize synthesizer
    synthesizer = PageSynthesizer(
        chart_dir=args.chart_dir,
        output_dir=args.output_dir,
        background_dir=args.background_dir,
        config=config,
    )

    # Generate dataset
    stats = synthesizer.generate_dataset(
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        prefix="doc",
    )

    # Create dataset.yaml
    yaml_path = create_dataset_yaml(args.output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Dataset Generation Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory:    {args.output_dir}")
    logger.info(f"Dataset YAML:        {yaml_path}")
    logger.info(f"Total samples:       {stats['total_samples']}")
    logger.info(f"Positive samples:    {stats['positive_samples']}")
    logger.info(f"Negative samples:    {stats['negative_samples']}")
    logger.info(f"Train samples:       {stats['train_samples']}")
    logger.info(f"Val samples:         {stats['val_samples']}")
    logger.info(f"Total charts placed: {stats['total_charts_placed']}")
    logger.info("")
    logger.info("Next step: Train YOLO")
    logger.info("  python scripts/train_yolo.py --workers 8 --epochs 100")


if __name__ == "__main__":
    main()
