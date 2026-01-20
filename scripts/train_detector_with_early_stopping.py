#!/usr/bin/env python3
"""
Train YOLO Chart Detector with Early Stopping

This script trains a YOLOv8 model for chart detection on synthetic documents
with proper early stopping to prevent overfitting and save training time.

Early Stopping Logic:
- Monitor: mAP50 (validation metric)
- Patience: Number of epochs without improvement before stopping
- Best weights are automatically saved as 'best.pt'
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO chart detector with early stopping"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model (default: yolov8n.pt for RTX 3060 6GB)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "training_synthetic" / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience - epochs without improvement (default: 15)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce if OOM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 0 for GPU, cpu for CPU (default: 0)",
    )
    
    # Output arguments
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "results" / "training_runs",
        help="Project directory for saving results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated with timestamp)",
    )
    
    # Advanced arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer (default: auto)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate ratio (default: 0.01)",
    )
    
    # Augmentation
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Apply augmentation (default: True)",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.5,
        help="Mosaic augmentation probability (default: 0.5)",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.1,
        help="MixUp augmentation probability (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"chart_detector_{timestamp}"
    
    # Configure logging
    logger.info("=" * 70)
    logger.info("YOLO Chart Detector Training with Early Stopping")
    logger.info("=" * 70)
    
    # Validate dataset
    if not args.data.exists():
        logger.error(f"Dataset YAML not found | path={args.data}")
        logger.info("Run generate_synthetic_dataset.py first to create the dataset")
        sys.exit(1)
    
    logger.info(f"Configuration:")
    logger.info(f"  Model:           {args.model}")
    logger.info(f"  Dataset:         {args.data}")
    logger.info(f"  Max Epochs:      {args.epochs}")
    logger.info(f"  Early Stopping:  patience={args.patience} epochs")
    logger.info(f"  Batch Size:      {args.batch}")
    logger.info(f"  Image Size:      {args.imgsz}")
    logger.info(f"  Device:          {args.device}")
    logger.info(f"  Output:          {args.project}/{args.name}")
    logger.info("-" * 70)
    
    # Initialize model
    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)
    
    # Print early stopping explanation
    logger.info("")
    logger.info("Early Stopping Explanation:")
    logger.info(f"  - Training will stop if mAP50 doesn't improve for {args.patience} epochs")
    logger.info(f"  - Best weights saved automatically to 'best.pt'")
    logger.info(f"  - Last weights saved to 'last.pt' (for resume)")
    logger.info("")
    
    # Start training
    logger.info("Starting training...")
    logger.info("-" * 70)
    
    try:
        results = model.train(
            # Dataset
            data=str(args.data),
            
            # Training duration
            epochs=args.epochs,
            patience=args.patience,  # EARLY STOPPING!
            
            # Batch and image size
            batch=args.batch,
            imgsz=args.imgsz,
            
            # Device
            device=args.device,
            workers=args.workers,
            
            # Output
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            
            # Optimizer
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            
            # Augmentation
            augment=args.augment,
            mosaic=args.mosaic,
            mixup=args.mixup,
            
            # Other settings
            resume=args.resume,
            verbose=True,
            plots=True,
            save=True,
            save_period=-1,  # Only save best.pt and last.pt
            
            # Callbacks for progress
            close_mosaic=10,  # Disable mosaic last 10 epochs
        )
        
        # Training complete
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        
        # Get results path
        results_path = args.project / args.name
        best_weights = results_path / "weights" / "best.pt"
        
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Best weights: {best_weights}")
        
        if best_weights.exists():
            logger.info("")
            logger.info("To validate the model:")
            logger.info(f"  yolo val model={best_weights} data={args.data}")
            logger.info("")
            logger.info("To run inference on images:")
            logger.info(f"  yolo predict model={best_weights} source=path/to/images")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed | error={e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
