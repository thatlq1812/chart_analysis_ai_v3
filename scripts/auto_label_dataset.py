"""
Auto-Label Dataset for YOLO Training

This script generates YOLO format labels for the training dataset.
Since all images are already cropped charts, we use full-image bounding boxes
with a small random margin to simulate realistic detection.

Usage:
    python scripts/auto_label_dataset.py
    python scripts/auto_label_dataset.py --margin 0.02
    python scripts/auto_label_dataset.py --dry-run
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_yolo_label(
    margin: float = 0.0,
    class_id: int = 0,
    random_margin: bool = True,
) -> str:
    """
    Generate YOLO format label for full-image bounding box.
    
    YOLO format: class_id x_center y_center width height
    All values normalized to 0-1
    
    Args:
        margin: Fixed margin from edges (0-0.1)
        class_id: Class ID (0 for single-class detection)
        random_margin: If True, add random variation to margin
        
    Returns:
        YOLO format label string
    """
    if random_margin:
        # Random margin between 0.01 and 0.05
        m = random.uniform(0.01, 0.05)
    else:
        m = margin
    
    # Center is always 0.5, 0.5
    x_center = 0.5
    y_center = 0.5
    
    # Width and height with margin
    width = 1.0 - (2 * m)
    height = 1.0 - (2 * m)
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_split(
    split_dir: Path,
    labels_dir: Path,
    margin: float = 0.0,
    random_margin: bool = True,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Process a split directory and generate labels.
    
    Args:
        split_dir: Directory containing images
        labels_dir: Directory to write labels
        margin: Fixed margin
        random_margin: Use random margin variation
        dry_run: If True, only count without writing
        
    Returns:
        Tuple of (processed_count, skipped_count)
    """
    processed = 0
    skipped = 0
    
    # Supported image extensions
    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    
    for ext in extensions:
        for img_path in split_dir.glob(f"*{ext}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Check if label already has content
            if label_path.exists():
                content = label_path.read_text().strip()
                if content:
                    skipped += 1
                    continue
            
            if not dry_run:
                # Generate and write label
                label = generate_yolo_label(
                    margin=margin,
                    class_id=0,  # Single class: chart
                    random_margin=random_margin,
                )
                label_path.write_text(label + "\n")
            
            processed += 1
    
    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate YOLO labels for chart dataset"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Fixed margin from edges (default: 0.02)"
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Disable random margin variation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files, don't write labels"
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        default=None,
        help="Path to training directory"
    )
    
    args = parser.parse_args()
    
    # Paths
    training_dir = Path(args.training_dir) if args.training_dir else project_root / "data" / "training"
    
    if not training_dir.exists():
        print(f"ERROR: Training directory not found: {training_dir}")
        print("Run prepare_dataset_splits.py first")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  AUTO-LABEL DATASET FOR YOLO")
    print("=" * 60)
    print(f"  Training dir: {training_dir}")
    print(f"  Margin: {args.margin}")
    print(f"  Random margin: {not args.no_random}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    total_processed = 0
    total_skipped = 0
    
    # Process each split
    for split in ["train", "val", "test"]:
        images_dir = training_dir / "images" / split
        labels_dir = training_dir / "labels" / split
        
        if not images_dir.exists():
            print(f"\n[{split}] Directory not found, skipping")
            continue
        
        print(f"\n[{split}] Processing...")
        
        processed, skipped = process_split(
            split_dir=images_dir,
            labels_dir=labels_dir,
            margin=args.margin,
            random_margin=not args.no_random,
            dry_run=args.dry_run,
        )
        
        total_processed += processed
        total_skipped += skipped
        
        print(f"    Labeled: {processed:,}")
        print(f"    Skipped (already labeled): {skipped:,}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total labeled: {total_processed:,}")
    print(f"  Total skipped: {total_skipped:,}")
    
    if args.dry_run:
        print("\n  [DRY RUN] No files were modified")
    else:
        print(f"\n  Labels written to: {training_dir / 'labels'}")
    
    # Verify a sample
    if not args.dry_run and total_processed > 0:
        print("\n  Sample label content:")
        sample_label = next((training_dir / "labels" / "train").glob("*.txt"), None)
        if sample_label:
            print(f"    File: {sample_label.name}")
            print(f"    Content: {sample_label.read_text().strip()}")
    
    print("=" * 60 + "\n")
    
    # Next steps
    print("Next steps:")
    print("  1. Verify labels: Check a few images with their labels")
    print("  2. Train YOLO:")
    print(f"     yolo train model=yolov8n.pt data={training_dir / 'dataset.yaml'} epochs=100 imgsz=640")
    print("")


if __name__ == "__main__":
    main()
