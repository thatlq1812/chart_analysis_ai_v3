#!/usr/bin/env python3
"""
Train YOLO Model for Chart Detection

Optimized for RTX 3060 Laptop (6GB VRAM):
- batch size auto-tuned
- workers=8 for good data loading
- grayscale input supported
- early stopping enabled
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for chart detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "training" / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto, default: -1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data loading workers (default: 8, optimized for 6GB VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 0 for GPU, cpu for CPU (default: 0)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "results" / "training_runs",
        help="Output project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: auto-generated)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )

    args = parser.parse_args()

    # Validate dataset
    if not args.data.exists():
        logger.error(f"Dataset not found | path={args.data}")
        logger.info("Generate dataset first:")
        logger.info("  1. python scripts/extract_backgrounds.py")
        logger.info("  2. python scripts/generate_synthetic_dataset.py")
        sys.exit(1)

    # Generate run name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"chart_detector_{timestamp}"

    logger.info("=" * 60)
    logger.info("YOLO Chart Detector Training")
    logger.info("=" * 60)
    logger.info(f"Model:      {args.model}")
    logger.info(f"Dataset:    {args.data}")
    logger.info(f"Epochs:     {args.epochs}")
    logger.info(f"Patience:   {args.patience}")
    logger.info(f"Batch:      {'auto' if args.batch == -1 else args.batch}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Workers:    {args.workers}")
    logger.info(f"Device:     {args.device}")
    logger.info(f"Output:     {args.project / args.name}")
    logger.info("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load model
    if args.resume:
        logger.info(f"Resuming from checkpoint | path={args.resume}")
        model = YOLO(str(args.resume))
    else:
        logger.info(f"Loading base model | model={args.model}")
        model = YOLO(args.model)

    # Training
    logger.info("Starting training...")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        # Augmentation settings
        hsv_h=0.0,  # No hue augmentation (grayscale)
        hsv_s=0.0,  # No saturation augmentation
        hsv_v=0.2,  # Value augmentation
        degrees=5.0,  # Small rotation
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.0,  # No vertical flip (documents are always upright)
        fliplr=0.0,  # No horizontal flip
        mosaic=0.5,  # Mosaic augmentation (helps with multi-chart detection)
        mixup=0.0,
        # Saving
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    # Show results
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info(f"Final Metrics:")
        logger.info(f"  mAP50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"  mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        logger.info(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")

    # Best weights path
    best_weights = args.project / args.name / "weights" / "best.pt"
    if best_weights.exists():
        logger.info(f"Best weights: {best_weights}")
        logger.info("")
        logger.info("Test on PDF:")
        logger.info(f"  python scripts/test_detection.py --weights {best_weights}")


if __name__ == "__main__":
    main()
