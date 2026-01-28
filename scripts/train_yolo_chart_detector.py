#!/usr/bin/env python3
"""
Train YOLO for Chart Detection (Binary: chart vs non-chart).

This script prepares data and trains YOLOv8 to detect chart regions in images.
The model outputs bounding boxes for detected charts.

Usage:
    python scripts/train_yolo_chart_detector.py --epochs 50
    python scripts/train_yolo_chart_detector.py --epochs 100 --batch 16
"""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ACADEMIC_DATASET = DATA_DIR / "academic_dataset"
IMAGES_DIR = ACADEMIC_DATASET / "images"
MANIFESTS_DIR = ACADEMIC_DATASET / "manifests"
YOLO_DATASET_DIR = DATA_DIR / "yolo_chart_detection"


def load_chart_manifest() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load train/val/test manifests."""
    train_manifest = MANIFESTS_DIR / "train_manifest.json"
    val_manifest = MANIFESTS_DIR / "val_manifest.json"
    test_manifest = MANIFESTS_DIR / "test_manifest.json"
    
    with open(train_manifest) as f:
        train_data = json.load(f)
    with open(val_manifest) as f:
        val_data = json.load(f)
    with open(test_manifest) as f:
        test_data = json.load(f)
    
    return train_data, val_data, test_data


def get_non_chart_images(chart_images: set, max_samples: int = 3000) -> List[Path]:
    """
    Get non-chart images from the dataset.
    
    Strategy: Images not in manifests are likely non-charts
    (diagrams, photos, tables, equations, etc.)
    """
    all_images = set(p.stem for p in IMAGES_DIR.glob("*.png"))
    non_chart_stems = all_images - chart_images
    
    # Sample randomly
    non_chart_list = list(non_chart_stems)
    random.shuffle(non_chart_list)
    
    sampled = non_chart_list[:max_samples]
    return [IMAGES_DIR / f"{stem}.png" for stem in sampled]


def create_yolo_label(image_path: Path, bbox: Dict = None, is_chart: bool = True) -> str:
    """
    Create YOLO format label.
    
    For charts: class_id x_center y_center width height (normalized)
    For non-charts: empty file (no objects)
    
    Args:
        image_path: Path to image
        bbox: Bounding box dict with x_min, y_min, x_max, y_max
        is_chart: Whether image contains a chart
        
    Returns:
        YOLO format label string
    """
    if not is_chart:
        return ""  # Empty label = no objects
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    
    if bbox:
        # Use provided bbox
        x_min = bbox.get("x_min", 0)
        y_min = bbox.get("y_min", 0)
        x_max = bbox.get("x_max", img_w)
        y_max = bbox.get("y_max", img_h)
    else:
        # Full image is the chart
        x_min, y_min = 0, 0
        x_max, y_max = img_w, img_h
    
    # Convert to YOLO format (normalized center + wh)
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    
    # Clamp values
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0.01, min(1, width))
    height = max(0.01, min(1, height))
    
    # Class 0 = chart
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def prepare_yolo_dataset(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    neg_ratio: float = 0.5,
    output_dir: Path = YOLO_DATASET_DIR,
) -> Path:
    """
    Prepare YOLO format dataset.
    
    Args:
        train_data: Training chart samples
        val_data: Validation chart samples
        test_data: Test chart samples
        neg_ratio: Ratio of negative samples (0.5 = 50% negatives)
        output_dir: Output directory
        
    Returns:
        Path to dataset.yaml
    """
    print("=" * 60)
    print("PREPARING YOLO DATASET")
    print("=" * 60)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Collect all chart image stems
    all_chart_stems = set()
    for item in train_data + val_data + test_data:
        all_chart_stems.add(Path(item["image_path"]).stem)
    
    print(f"Total chart samples: {len(all_chart_stems)}")
    
    # Calculate negative samples needed
    total_charts = len(train_data) + len(val_data) + len(test_data)
    neg_samples_needed = int(total_charts * neg_ratio)
    
    # Get negative samples
    print(f"Collecting {neg_samples_needed} negative samples...")
    non_chart_images = get_non_chart_images(all_chart_stems, neg_samples_needed)
    print(f"Found {len(non_chart_images)} non-chart images")
    
    # Split negatives proportionally
    n_train_neg = int(len(non_chart_images) * 0.7)
    n_val_neg = int(len(non_chart_images) * 0.15)
    
    random.shuffle(non_chart_images)
    train_neg = non_chart_images[:n_train_neg]
    val_neg = non_chart_images[n_train_neg:n_train_neg + n_val_neg]
    test_neg = non_chart_images[n_train_neg + n_val_neg:]
    
    def process_split(
        split_name: str,
        chart_items: List[Dict],
        neg_images: List[Path],
    ):
        """Process a single split."""
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        count = 0
        errors = 0
        
        # Process chart images
        for item in tqdm(chart_items, desc=f"{split_name} charts"):
            try:
                src_path = IMAGES_DIR / item["image_path"]
                if not src_path.exists():
                    errors += 1
                    continue
                
                # Copy image
                dst_path = images_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                
                # Create label (full image = chart bbox)
                # Use bbox from manifest if available, else full image
                bbox = item.get("bbox")
                label = create_yolo_label(src_path, bbox=None, is_chart=True)  # Full image
                
                label_path = labels_dir / f"{src_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write(label)
                
                count += 1
                
            except Exception as e:
                errors += 1
        
        # Process negative images
        for src_path in tqdm(neg_images, desc=f"{split_name} non-charts"):
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
    train_count = process_split("train", train_data, train_neg)
    val_count = process_split("val", val_data, val_neg)
    test_count = process_split("test", test_data, test_neg)
    
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
    parser.add_argument("--neg-ratio", type=float, default=0.5, 
                        help="Ratio of negative samples")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare dataset, don't train")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation, use existing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO CHART DETECTOR TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load manifests
    print("\nLoading manifests...")
    train_data, val_data, test_data = load_chart_manifest()
    print(f"  Train: {len(train_data)} charts")
    print(f"  Val:   {len(val_data)} charts")
    print(f"  Test:  {len(test_data)} charts")
    
    # Prepare dataset
    if not args.skip_prepare:
        dataset_yaml = prepare_yolo_dataset(
            train_data, val_data, test_data,
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
