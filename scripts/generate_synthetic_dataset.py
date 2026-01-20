#!/usr/bin/env python3
"""
Generate Synthetic Document Dataset for Chart Detection

This script creates a training dataset by:
1. Taking chart images from ChartQA/HuggingFace datasets
2. Pasting them onto document-like backgrounds
3. Generating accurate YOLO bounding box labels

The resulting model will detect charts within full document pages,
not just classify isolated chart images.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
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
        help="Number of synthetic pages to generate (default: 10000)",
    )
    parser.add_argument(
        "--min-charts",
        type=int,
        default=1,
        help="Minimum charts per page (default: 1)",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=3,
        help="Maximum charts per page (default: 3)",
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
        default=PROJECT_ROOT / "data" / "training_synthetic",
        help="Output directory for synthetic dataset",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Optional: Directory containing background images",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--page-width",
        type=int,
        default=1654,
        help="Page width in pixels (default: 1654 = A4 at 150 DPI)",
    )
    parser.add_argument(
        "--page-height",
        type=int,
        default=2339,
        help="Page height in pixels (default: 2339 = A4 at 150 DPI)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.info("=" * 60)
    logger.info("Synthetic Document Dataset Generator")
    logger.info("=" * 60)
    
    # Validate chart directory
    if not args.chart_dir.exists():
        logger.error(f"Chart directory not found | path={args.chart_dir}")
        sys.exit(1)
    
    # Count available charts
    chart_count = len(list(args.chart_dir.glob("**/*.png"))) + len(list(args.chart_dir.glob("**/*.jpg")))
    logger.info(f"Source charts found | count={chart_count} | dir={args.chart_dir}")
    
    if chart_count == 0:
        logger.error("No chart images found in the specified directory")
        sys.exit(1)
    
    # Create configuration
    config = SynthesizerConfig(
        min_charts_per_page=args.min_charts,
        max_charts_per_page=args.max_charts,
        page_width=args.page_width,
        page_height=args.page_height,
        min_chart_width_ratio=0.35,
        max_chart_width_ratio=0.85,
        margin_top=80,
        margin_bottom=80,
        margin_left=60,
        margin_right=60,
        apply_noise=True,
        noise_intensity=0.015,
    )
    
    logger.info(f"Configuration:")
    logger.info(f"  - Charts per page: {config.min_charts_per_page}-{config.max_charts_per_page}")
    logger.info(f"  - Page size: {config.page_width}x{config.page_height}")
    logger.info(f"  - Chart width ratio: {config.min_chart_width_ratio:.0%}-{config.max_chart_width_ratio:.0%}")
    
    # Clean output directory if exists
    if args.output_dir.exists():
        logger.warning(f"Output directory exists, cleaning | path={args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Initialize synthesizer
    synthesizer = PageSynthesizer(
        chart_dir=args.chart_dir,
        output_dir=args.output_dir,
        background_dir=args.background_dir,
        config=config,
    )
    
    # Generate dataset
    logger.info(f"Generating {args.num_samples} synthetic pages...")
    stats = synthesizer.generate_dataset(
        num_samples=args.num_samples,
        prefix="synth_doc",
    )
    
    logger.info(f"Generation complete | stats={stats}")
    
    # Create train/val split and dataset.yaml
    logger.info(f"Creating train/val split with ratio {args.train_ratio}...")
    yaml_path = create_dataset_yaml(args.output_dir, train_ratio=args.train_ratio)
    
    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"  - Output: {args.output_dir}")
    logger.info(f"  - Dataset YAML: {yaml_path}")
    logger.info(f"  - Total pages: {stats['total_samples']}")
    logger.info(f"  - Total charts: {stats['total_charts']}")
    logger.info(f"  - Avg charts/page: {stats.get('avg_charts_per_page', 0):.2f}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step: Train YOLO with Early Stopping")
    logger.info("  yolo train model=yolov8n.pt data=data/training_synthetic/dataset.yaml \\")
    logger.info("       epochs=100 patience=15 device=0 project=results/training_runs \\")
    logger.info("       name=chart_detector_synthetic")


if __name__ == "__main__":
    main()
