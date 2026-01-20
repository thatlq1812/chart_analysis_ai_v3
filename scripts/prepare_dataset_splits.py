"""
Dataset Splitter - Prepare data for YOLO training

This script splits the combined dataset (HuggingFace + ArXiv) into
train/val/test sets following YOLO directory structure.

Usage:
    python scripts/prepare_dataset_splits.py
    python scripts/prepare_dataset_splits.py --train-ratio 0.8 --val-ratio 0.1
    python scripts/prepare_dataset_splits.py --dry-run

Directory structure created:
    data/training/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def collect_all_images(data_root: Path) -> List[Dict]:
    """
    Collect all images from different sources.
    
    Returns:
        List of dicts with image info (path, source, metadata_path)
    """
    images = []
    
    # Source 1: HuggingFace ChartQA
    hf_chartqa_dir = data_root / "academic_dataset" / "images" / "huggingface" / "chartqa"
    if hf_chartqa_dir.exists():
        for img_path in hf_chartqa_dir.glob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                images.append({
                    "path": img_path,
                    "source": "huggingface_chartqa",
                    "metadata_path": None,  # HF images may not have separate metadata
                })
    
    # Source 2: ArXiv mined images
    arxiv_images_dir = data_root / "academic_dataset" / "images"
    arxiv_metadata_dir = data_root / "academic_dataset" / "metadata"
    
    if arxiv_images_dir.exists():
        for img_path in arxiv_images_dir.glob("*.png"):
            # Skip if in subdirectory (like huggingface/)
            if img_path.parent == arxiv_images_dir:
                # Look for corresponding metadata
                metadata_path = arxiv_metadata_dir / f"{img_path.stem}.json"
                images.append({
                    "path": img_path,
                    "source": "arxiv_mined",
                    "metadata_path": metadata_path if metadata_path.exists() else None,
                })
        
        # Also check for jpg
        for img_path in arxiv_images_dir.glob("*.jpg"):
            if img_path.parent == arxiv_images_dir:
                metadata_path = arxiv_metadata_dir / f"{img_path.stem}.json"
                images.append({
                    "path": img_path,
                    "source": "arxiv_mined", 
                    "metadata_path": metadata_path if metadata_path.exists() else None,
                })
    
    return images


def split_dataset(
    images: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split images into train/val/test sets.
    
    Args:
        images: List of image info dicts
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1.0"
    
    # Shuffle with seed
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def create_yolo_structure(output_dir: Path) -> Dict[str, Path]:
    """Create YOLO directory structure."""
    dirs = {
        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "test_images": output_dir / "images" / "test",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
        "test_labels": output_dir / "labels" / "test",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def copy_images_to_split(
    images: List[Dict],
    dest_dir: Path,
    labels_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """
    Copy images to destination directory.
    
    Args:
        images: List of image info dicts
        dest_dir: Destination directory for images
        labels_dir: Optional directory for label files
        dry_run: If True, only print what would be done
        
    Returns:
        Number of images copied
    """
    copied = 0
    
    for img_info in images:
        src_path = img_info["path"]
        dest_path = dest_dir / src_path.name
        
        if dry_run:
            print(f"  Would copy: {src_path.name}")
        else:
            # Copy image
            shutil.copy2(src_path, dest_path)
            
            # Create empty label file (for YOLO - will be populated during annotation)
            if labels_dir:
                label_path = labels_dir / f"{src_path.stem}.txt"
                if not label_path.exists():
                    label_path.touch()
        
        copied += 1
    
    return copied


def create_dataset_yaml(output_dir: Path, class_names: List[str] = None):
    """Create YOLO dataset.yaml configuration file."""
    if class_names is None:
        # Default chart types
        class_names = [
            "bar_chart",
            "line_chart", 
            "pie_chart",
            "scatter_plot",
            "area_chart",
            "other_chart",
        ]
    
    yaml_content = f"""# Geo-SLM Chart Analysis Dataset
# Auto-generated by prepare_dataset_splits.py

path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path


def create_split_manifest(
    output_dir: Path,
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
):
    """Create manifest JSON with split information."""
    manifest = {
        "created_at": str(Path(__file__).stat().st_mtime),
        "total_images": len(train) + len(val) + len(test),
        "splits": {
            "train": {
                "count": len(train),
                "sources": {},
            },
            "val": {
                "count": len(val),
                "sources": {},
            },
            "test": {
                "count": len(test),
                "sources": {},
            },
        },
    }
    
    # Count by source
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for img in split_data:
            source = img["source"]
            if source not in manifest["splits"][split_name]["sources"]:
                manifest["splits"][split_name]["sources"][source] = 0
            manifest["splits"][split_name]["sources"][source] += 1
    
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test for YOLO training"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio for training set (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio for test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/training)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't copy files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split directories"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("ERROR: Ratios must sum to 1.0")
        sys.exit(1)
    
    data_root = project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "training"
    
    print("\n" + "=" * 60)
    print("  DATASET SPLIT PREPARATION")
    print("=" * 60)
    
    # Collect images
    print("\n[1] Collecting images from all sources...")
    images = collect_all_images(data_root)
    print(f"    Found {len(images):,} total images")
    
    if len(images) == 0:
        print("\nERROR: No images found. Run mining first:")
        print("  python -m tools.data_factory.main mine")
        sys.exit(1)
    
    # Count by source
    source_counts = {}
    for img in images:
        source = img["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("    By source:")
    for source, count in sorted(source_counts.items()):
        print(f"      - {source}: {count:,}")
    
    # Split
    print(f"\n[2] Splitting dataset (seed={args.seed})...")
    print(f"    Ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    
    train, val, test = split_dataset(
        images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    print(f"    Train: {len(train):,} images")
    print(f"    Val:   {len(val):,} images")
    print(f"    Test:  {len(test):,} images")
    
    if args.dry_run:
        print("\n[DRY RUN] Would create directories and copy files to:")
        print(f"    {output_dir}")
        print("\n    Sample files to copy:")
        for img in train[:5]:
            print(f"      train/ <- {img['path'].name}")
        return
    
    # Check if output exists
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            print(f"\nWARNING: Output directory not empty: {output_dir}")
            print("Use --force to overwrite")
            sys.exit(1)
        else:
            print(f"\n[3] Cleaning existing directory...")
            for subdir in ["images", "labels"]:
                subpath = output_dir / subdir
                if subpath.exists():
                    shutil.rmtree(subpath)
    
    # Create structure
    print(f"\n[3] Creating YOLO directory structure...")
    dirs = create_yolo_structure(output_dir)
    print(f"    Created: {output_dir}")
    
    # Copy files
    print(f"\n[4] Copying images...")
    
    train_copied = copy_images_to_split(
        train, dirs["train_images"], dirs["train_labels"]
    )
    print(f"    Train: {train_copied:,} images copied")
    
    val_copied = copy_images_to_split(
        val, dirs["val_images"], dirs["val_labels"]
    )
    print(f"    Val:   {val_copied:,} images copied")
    
    test_copied = copy_images_to_split(
        test, dirs["test_images"], dirs["test_labels"]
    )
    print(f"    Test:  {test_copied:,} images copied")
    
    # Create config files
    print(f"\n[5] Creating configuration files...")
    
    yaml_path = create_dataset_yaml(output_dir)
    print(f"    Created: {yaml_path.name}")
    
    manifest_path = create_split_manifest(output_dir, train, val, test)
    print(f"    Created: {manifest_path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SPLIT COMPLETE")
    print("=" * 60)
    total = train_copied + val_copied + test_copied
    print(f"  Total images: {total:,}")
    print(f"  Output: {output_dir}")
    print(f"\n  Next steps:")
    print(f"  1. Annotate images using LabelImg or CVAT")
    print(f"  2. Train YOLO: yolo train data={yaml_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
