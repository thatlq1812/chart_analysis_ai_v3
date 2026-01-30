#!/usr/bin/env python3
"""
Train YOLO for Chart Detection (Binary: chart vs non-chart).

This script prepares data and trains YOLOv8 to detect chart regions in images.
The model outputs bounding boxes for detected charts.

Data Source: classified_charts/ directory with 32,364 chart images
- Chart types: area, bar, box, heatmap, histogram, line, pie, scatter
- Non-chart: diagram, not_a_chart, other, table (used as negative samples)
- Excluded: uncertain (14,466 images - ambiguous labels)

Usage:
    python scripts/train_yolo_chart_detector.py --epochs 50
    python scripts/train_yolo_chart_detector.py --epochs 100 --batch 16 --prepare-only
"""

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ACADEMIC_DATASET = DATA_DIR / "academic_dataset"
CLASSIFIED_CHARTS_DIR = ACADEMIC_DATASET / "classified_charts"
YOLO_DATASET_DIR = DATA_DIR / "yolo_chart_detection"

# Chart types to use as positive samples
CHART_TYPES = ["area", "bar", "box", "heatmap", "histogram", "line", "pie", "scatter"]

# Non-chart types to use as negative samples
NON_CHART_TYPES = ["diagram", "not_a_chart", "other", "table"]

# Excluded (ambiguous labels)
EXCLUDED_TYPES = ["uncertain"]


def load_classified_images() -> Tuple[List[Path], List[Path]]:
    """
    Load images from classified_charts directory.
    
    Returns:
        Tuple of (chart_images, non_chart_images)
    """
    chart_images = []
    non_chart_images = []
    
    for chart_type in CHART_TYPES:
        type_dir = CLASSIFIED_CHARTS_DIR / chart_type
        if type_dir.exists():
            images = list(type_dir.glob("*.png"))
            chart_images.extend(images)
            print(f"  {chart_type}: {len(images)} images")
    
    for non_chart_type in NON_CHART_TYPES:
        type_dir = CLASSIFIED_CHARTS_DIR / non_chart_type
        if type_dir.exists():
            images = list(type_dir.glob("*.png"))
            non_chart_images.extend(images)
            print(f"  {non_chart_type}: {len(images)} images (negative)")
    
    return chart_images, non_chart_images


def split_data(
    images: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split images into train/val/test sets.
    
    Args:
        images: List of image paths
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.15)
        
    Returns:
        Tuple of (train, val, test) lists
    """
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test


def create_yolo_label(image_path: Path, is_chart: bool = True) -> str:
    """
    Create YOLO format label.
    
    For charts: class_id x_center y_center width height (normalized)
    For non-charts: empty file (no objects)
    
    Args:
        image_path: Path to image
        is_chart: Whether image contains a chart
        
    Returns:
        YOLO format label string
    """
    if not is_chart:
        return ""  # Empty label = no objects
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    
    # Full image is the chart (since these are already cropped chart images)
    # YOLO format: class_id x_center y_center width height (normalized 0-1)
    # Class 0 = chart, full image = center at 0.5, 0.5 with width/height = 1.0
    return "0 0.500000 0.500000 1.000000 1.000000"


def prepare_yolo_dataset(
    chart_images: List[Path],
    non_chart_images: List[Path],
    neg_ratio: float = 0.3,
    output_dir: Path = YOLO_DATASET_DIR,
) -> Path:
    """
    Prepare YOLO format dataset.
    
    Args:
        chart_images: List of chart image paths
        non_chart_images: List of non-chart image paths
        neg_ratio: Ratio of negative samples relative to positive (0.3 = 30%)
        output_dir: Output directory
        
    Returns:
        Path to dataset.yaml
    """
    print("\n" + "=" * 60)
    print("PREPARING YOLO DATASET")
    print("=" * 60)
    
    # Clear existing dataset
    if output_dir.exists():
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Split chart images (70/15/15)
    print(f"\nSplitting {len(chart_images)} chart images...")
    train_charts, val_charts, test_charts = split_data(chart_images)
    print(f"  Train: {len(train_charts)}")
    print(f"  Val:   {len(val_charts)}")
    print(f"  Test:  {len(test_charts)}")
    
    # Calculate and split negative samples
    n_neg_needed = int(len(chart_images) * neg_ratio)
    n_neg_available = len(non_chart_images)
    n_neg_used = min(n_neg_needed, n_neg_available)
    
    print(f"\nUsing {n_neg_used} negative samples (available: {n_neg_available})")
    
    if n_neg_used > 0:
        random.shuffle(non_chart_images)
        neg_samples = non_chart_images[:n_neg_used]
        train_neg, val_neg, test_neg = split_data(neg_samples)
    else:
        train_neg, val_neg, test_neg = [], [], []
    
    def process_split(
        split_name: str,
        chart_items: List[Path],
        neg_items: List[Path],
    ) -> int:
        """Process a single split."""
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        count = 0
        errors = 0
        
        # Process chart images
        for src_path in tqdm(chart_items, desc=f"{split_name} charts"):
            try:
                if not src_path.exists():
                    errors += 1
                    continue
                
                # Copy image
                dst_path = images_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                
                # Create label (full image = chart bbox)
                label = create_yolo_label(src_path, is_chart=True)
                label_path = labels_dir / f"{src_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write(label)
                
                count += 1
                
            except Exception as e:
                errors += 1
        
        # Process negative images
        for src_path in tqdm(neg_items, desc=f"{split_name} non-charts"):
            try:
                if not src_path.exists():
                    errors += 1
                    continue
                
                # Copy image
                dst_path = images_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                
                # Create empty label (no chart)
                label_path = labels_dir / f"{src_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write("")  # Empty = no objects
                
                count += 1
                
            except Exception as e:
                errors += 1
        
        print(f"  {split_name}: {count} images, {errors} errors")
        return count
    
    # Process all splits
    train_count = process_split("train", train_charts, train_neg)
    val_count = process_split("val", val_charts, val_neg)
    test_count = process_split("test", test_charts, test_neg)
    
    # Create dataset.yaml
    dataset_config = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,  # Number of classes
        "names": ["chart"],  # Class names
    }
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("\n" + "=" * 60)
    print("DATASET PREPARED")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Test:  {test_count} images")
    print(f"  Config: {yaml_path}")
    print("=" * 60)
    
    return yaml_path


def train_yolo(
    dataset_yaml: Path,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    model_base: str = "yolov8n.pt",
    project_name: str = "chart_detector",
) -> Path:
    """
    Train YOLO model for chart detection.
    
    Args:
        dataset_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        model_base: Base model to finetune
        project_name: Project name for runs
        
    Returns:
        Path to best weights
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("TRAINING YOLO CHART DETECTOR")
    print("=" * 60)
    print(f"  Base model: {model_base}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print("=" * 60 + "\n")
    
    # Load base model
    model = YOLO(model_base)
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=project_name,
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
        workers=0,  # Avoid multiprocessing issues on Windows
    )
    
    # Find best weights
    runs_dir = PROJECT_ROOT / "runs" / "detect" / project_name
    best_weights = runs_dir / "weights" / "best.pt"
    
    if best_weights.exists():
        # Copy to models/weights
        dest_path = PROJECT_ROOT / "models" / "weights" / "yolo_chart_detector.pt"
        shutil.copy2(best_weights, dest_path)
        print(f"\nBest model saved to: {dest_path}")
        return dest_path
    
    return best_weights


def evaluate_model(model_path: Path, dataset_yaml: Path):
    """Evaluate trained model on test set."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    model = YOLO(str(model_path))
    
    # Validate on test set
    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for chart detection")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--neg-ratio", type=float, default=0.3, 
                        help="Ratio of negative samples (0.3 = 30%%)")
    parser.add_argument("--model", type=str, default="models/weights/yolov8n.pt",
                        help="Base model path")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare dataset, don't train")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation, use existing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO CHART DETECTOR TRAINING (v2)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load images from classified_charts
    print("\nLoading classified images...")
    chart_images, non_chart_images = load_classified_images()
    print(f"\nTotal charts: {len(chart_images)}")
    print(f"Total non-charts: {len(non_chart_images)}")
    
    # Prepare dataset
    if not args.skip_prepare:
        dataset_yaml = prepare_yolo_dataset(
            chart_images=chart_images,
            non_chart_images=non_chart_images,
            neg_ratio=args.neg_ratio,
        )
    else:
        dataset_yaml = YOLO_DATASET_DIR / "dataset.yaml"
        if not dataset_yaml.exists():
            print(f"ERROR: Dataset not found at {dataset_yaml}")
            print("Run without --skip-prepare first.")
            return
    
    if args.prepare_only:
        print("\nDataset preparation complete. Skipping training.")
        return
    
    # Train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = train_yolo(
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model_base=args.model,
        project_name=f"chart_detector_{timestamp}",
    )
    
    # Evaluate
    if model_path.exists():
        evaluate_model(model_path, dataset_yaml)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
