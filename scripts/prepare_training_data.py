"""
Data Preparation Script for ResNet-18 Training

Splits academic dataset into train/val/test (70/15/15)
Creates manifest files for efficient loading
Analyzes class distribution and reports statistics
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import logging
import shutil

logger = logging.getLogger(__name__)


def load_metadata(metadata_dir: Path, images_dir: Path) -> List[Dict]:
    """Load metadata only for images that exist on disk."""
    # Step 1: Get all existing image files
    existing_images = {img.stem for img in images_dir.glob("*.png")}
    logger.info(f"Found {len(existing_images)} existing images in {images_dir}")
    
    # Step 2: Load metadata only for existing images
    metadata_files = list(metadata_dir.glob("*.json"))
    logger.info(f"Found {len(metadata_files)} metadata files")
    
    all_charts = []
    skipped = 0
    for meta_file in metadata_files:
        try:
            with open(meta_file, encoding="utf-8") as f:
                chart_data = json.load(f)
                image_id = chart_data.get("image_id", "")
                
                # Only keep metadata if corresponding image exists
                if image_id in existing_images:
                    all_charts.append(chart_data)
                else:
                    skipped += 1
        except Exception as e:
            logger.warning(f"Failed to load {meta_file.name}: {e}")
    
    logger.info(f"Loaded {len(all_charts)} charts (skipped {skipped} without images)")
    return all_charts


def filter_valid_charts(charts: List[Dict], images_dir: Path) -> List[Dict]:
    """Keep charts with valid types (exclude unknown/other)."""
    valid_charts = []
    skipped = 0
    
    for chart in charts:
        chart_type = chart.get("chart_type", "").lower()
        
        # Skip unknown and other categories
        if chart_type in ["unknown", "other", "", "none"]:
            skipped += 1
            continue
        
        # Construct image path (relative to images_dir)
        image_id = chart.get("image_id", "")
        chart["image_path"] = f"{image_id}.png"
        valid_charts.append(chart)
    
    logger.info(f"Valid charts: {len(valid_charts)}/{len(charts)} (skipped {skipped} unknown/other)")
    return valid_charts


def analyze_distribution(charts: List[Dict]) -> Dict[str, int]:
    """Analyze class distribution."""
    types = [chart["chart_type"] for chart in charts]
    distribution = dict(Counter(types))
    
    logger.info("Class Distribution:")
    total = len(charts)
    for chart_type, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / total
        logger.info(f"  {chart_type:15s}: {count:4d} ({percentage:5.2f}%)")
    
    return distribution


def stratified_split(
    charts: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split charts into train/val/test with stratification.
    
    Ensures each split has similar class distribution.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    
    # Group by chart type
    charts_by_type = {}
    for chart in charts:
        chart_type = chart["chart_type"]
        if chart_type not in charts_by_type:
            charts_by_type[chart_type] = []
        charts_by_type[chart_type].append(chart)
    
    train_set = []
    val_set = []
    test_set = []
    
    # Split each class
    for chart_type, type_charts in charts_by_type.items():
        random.shuffle(type_charts)
        
        n = len(type_charts)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_set.extend(type_charts[:n_train])
        val_set.extend(type_charts[n_train:n_train+n_val])
        test_set.extend(type_charts[n_train+n_val:])
        
        logger.info(
            f"{chart_type:15s}: train={n_train:4d}, val={n_val:4d}, "
            f"test={n - n_train - n_val:4d}"
        )
    
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    logger.info(f"\nTotal splits: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    return train_set, val_set, test_set


def save_manifest(charts: List[Dict], output_path: Path):
    """Save manifest JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(charts, f, indent=2)
    
    logger.info(f"Saved manifest: {output_path} ({len(charts)} samples)")


def generate_summary(
    train_set: List[Dict],
    val_set: List[Dict],
    test_set: List[Dict],
    output_path: Path,
):
    """Generate summary report."""
    summary = {
        "total_samples": len(train_set) + len(val_set) + len(test_set),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "train_distribution": dict(Counter(c["chart_type"] for c in train_set)),
        "val_distribution": dict(Counter(c["chart_type"] for c in val_set)),
        "test_distribution": dict(Counter(c["chart_type"] for c in test_set)),
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved: {output_path}")
    
    # Also print to console
    logger.info("\n" + "=" * 60)
    logger.info("DATA SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {summary['total_samples']}")
    logger.info(f"  Train: {summary['train_samples']} ({100*summary['train_samples']/summary['total_samples']:.1f}%)")
    logger.info(f"  Val:   {summary['val_samples']} ({100*summary['val_samples']/summary['total_samples']:.1f}%)")
    logger.info(f"  Test:  {summary['test_samples']} ({100*summary['test_samples']/summary['total_samples']:.1f}%)")


def main():
    """Main data preparation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data splits")
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="data/academic_dataset/metadata",
        help="Path to metadata directory"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/academic_dataset/images",
        help="Path to images directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/academic_dataset/manifests",
        help="Output directory for manifests"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Load and validate data
    metadata_dir = Path(args.metadata_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("Loading metadata...")
    charts = load_metadata(metadata_dir, images_dir)
    
    logger.info("Validating image files...")
    charts = filter_valid_charts(charts, images_dir)
    
    if not charts:
        logger.error("No valid charts found. Exiting.")
        return
    
    # Analyze distribution
    logger.info("\n" + "=" * 60)
    logger.info("ORIGINAL DISTRIBUTION")
    logger.info("=" * 60)
    analyze_distribution(charts)
    
    # Perform stratified split
    logger.info("\n" + "=" * 60)
    logger.info("STRATIFIED SPLIT")
    logger.info("=" * 60)
    train_set, val_set, test_set = stratified_split(
        charts,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    # Save manifests
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MANIFESTS")
    logger.info("=" * 60)
    save_manifest(train_set, output_dir / "train_manifest.json")
    save_manifest(val_set, output_dir / "val_manifest.json")
    save_manifest(test_set, output_dir / "test_manifest.json")
    
    # Generate summary
    generate_summary(train_set, val_set, test_set, output_dir / "split_summary.json")
    
    logger.info("\n[SUCCESS] Data preparation complete!")
    logger.info(f"Manifest files saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Review split_summary.json")
    logger.info("  2. Run training: python scripts/train_resnet18_classifier.py --data-dir data/academic_dataset/images")


if __name__ == "__main__":
    main()
