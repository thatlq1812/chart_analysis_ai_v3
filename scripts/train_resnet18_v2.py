"""
ResNet-18 Chart Classifier Training Pipeline v2

Improvements over v1:
- Uses Gemini-corrected labels (classified_charts/)
- Stronger data augmentation (grayscale-first for academic charts)
- Class-balanced training with weighted loss
- 60 epochs with cosine annealing
- Mixed precision training (faster on GPU)
- Fast loading: resize large images immediately, grayscale conversion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import PIL.Image
# Increase limit for large academic paper images
PIL.Image.MAX_IMAGE_PIXELS = 200_000_000

import json
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Chart types to train (8 valid chart types only)
CHART_TYPES = ["area", "bar", "box", "heatmap", "histogram", "line", "pie", "scatter"]


class ChartDatasetV2(Dataset):
    """
    Dataset for chart classification using classified_charts folder.
    Supports train/val/test splits with stratified sampling.
    """
    
    def __init__(
        self,
        classified_dir: Path,
        split: str = "train",
        transform=None,
        class_mapping: Dict[str, int] = None,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Args:
            classified_dir: Path to classified_charts/
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
            class_mapping: Dict mapping class names to indices
            split_ratio: (train, val, test) ratios
            seed: Random seed for reproducibility
            max_samples_per_class: Cap samples per class (for undersampling)
        """
        self.classified_dir = Path(classified_dir)
        self.split = split
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        
        # Build class mapping
        if class_mapping is None:
            self.class_mapping = {cls: idx for idx, cls in enumerate(CHART_TYPES)}
        else:
            self.class_mapping = class_mapping
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Load all images with stratified split
        self.samples = self._load_samples(split_ratio, seed)
        
        logger.info(
            f"ChartDatasetV2 | split={split} | samples={len(self.samples)} | "
            f"classes={len(self.class_mapping)}"
        )
    
    def _load_samples(
        self,
        split_ratio: Tuple[float, float, float],
        seed: int
    ) -> List[Tuple[Path, str]]:
        """Load and split samples."""
        random.seed(seed)
        np.random.seed(seed)
        
        all_samples = []
        
        for chart_type in CHART_TYPES:
            type_dir = self.classified_dir / chart_type
            if not type_dir.exists():
                continue
            
            images = list(type_dir.glob("*.png"))
            random.shuffle(images)
            
            # Apply undersampling if specified
            if self.max_samples_per_class and len(images) > self.max_samples_per_class:
                images = images[:self.max_samples_per_class]
            
            # Split
            n = len(images)
            train_end = int(n * split_ratio[0])
            val_end = train_end + int(n * split_ratio[1])
            
            if self.split == "train":
                selected = images[:train_end]
            elif self.split == "val":
                selected = images[train_end:val_end]
            else:  # test
                selected = images[val_end:]
            
            for img_path in selected:
                all_samples.append((img_path, chart_type))
        
        random.shuffle(all_samples)
        return all_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, chart_type = self.samples[idx]
        
        # Load preprocessed grayscale image (already 256x256)
        image = Image.open(img_path).convert("RGB")  # Convert L to RGB for ResNet
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_mapping[chart_type]
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes."""
        types = [t for _, t in self.samples]
        return dict(Counter(types))
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted loss."""
        dist = self.get_class_distribution()
        total = sum(dist.values())
        
        weights = []
        for chart_type in CHART_TYPES:
            count = dist.get(chart_type, 1)
            # Inverse frequency weighting
            weight = total / (len(CHART_TYPES) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def create_transforms(image_size: int = 224, is_training: bool = True):
    """Create transforms for preprocessed 256x256 grayscale images."""
    
    # Grayscale normalization (same value repeated for 3 channels)
    normalize = transforms.Normalize(
        mean=[0.485, 0.485, 0.485],
        std=[0.229, 0.229, 0.229]
    )
    
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(image_size),  # 256 -> 224
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(image_size),  # 256 -> 224
            transforms.ToTensor(),
            normalize,
        ])


def create_dataloaders(
    classified_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], torch.Tensor]:
    """Create train/val/test dataloaders."""
    
    train_transform = create_transforms(is_training=True)
    eval_transform = create_transforms(is_training=False)
    
    train_dataset = ChartDatasetV2(
        classified_dir,
        split="train",
        transform=train_transform,
        max_samples_per_class=max_samples_per_class,
    )
    
    val_dataset = ChartDatasetV2(
        classified_dir,
        split="val",
        transform=eval_transform,
        class_mapping=train_dataset.class_mapping,
    )
    
    test_dataset = ChartDatasetV2(
        classified_dir,
        split="test",
        transform=eval_transform,
        class_mapping=train_dataset.class_mapping,
    )
    
    # Log distributions
    logger.info(f"Train distribution: {train_dataset.get_class_distribution()}")
    logger.info(f"Val distribution: {val_dataset.get_class_distribution()}")
    logger.info(f"Test distribution: {test_dataset.get_class_distribution()}")
    
    # Get class weights for weighted loss
    class_weights = train_dataset.get_class_weights()
    logger.info(f"Class weights: {dict(zip(CHART_TYPES, class_weights.tolist()))}")
    
    # WeightedRandomSampler for balanced batches
    dist = train_dataset.get_class_distribution()
    sample_weights = []
    for img_path, chart_type in train_dataset.samples:
        weight = 1.0 / dist[chart_type]
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_mapping, class_weights


class ResNetChartClassifierV2(nn.Module):
    """ResNet-18 classifier with dropout for regularization."""
    
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace final FC with dropout + FC
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
        logger.info(f"ResNetChartClassifierV2 | classes={num_classes} | dropout={dropout}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze all layers except final FC."""
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.resnet.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{100. * correct / total:.1f}%"
        })
    
    return running_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """Validate model and return per-class accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = Counter()
    class_total = Counter()
    
    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Per-class
        for pred, label in zip(predicted, labels):
            class_total[label.item()] += 1
            if pred == label:
                class_correct[label.item()] += 1
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for idx, chart_type in enumerate(CHART_TYPES):
        if class_total[idx] > 0:
            per_class_acc[chart_type] = 100. * class_correct[idx] / class_total[idx]
        else:
            per_class_acc[chart_type] = 0.0
    
    return running_loss / len(dataloader), 100. * correct / total, per_class_acc


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ResNet-18 Chart Classifier v2")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/academic_dataset/classified_charts_preprocessed",
        help="Path to preprocessed classified_charts directory"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per class (for undersampling)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Output directory"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    use_amp = not args.no_amp and device.type == "cuda"
    logger.info(f"Mixed precision: {use_amp}")
    
    # Create dataloaders
    classified_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader, class_mapping, class_weights = create_dataloaders(
        classified_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_samples_per_class=args.max_samples,
    )
    
    # Create model
    num_classes = len(class_mapping)
    model = ResNetChartClassifierV2(num_classes=num_classes, dropout=0.3).to(device)
    
    # Loss with class weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "per_class_acc": [], "lr": []
    }
    
    best_val_acc = 0.0
    start_time = datetime.now()
    patience_counter = 0
    
    # ============================================================
    # PHASE 1: Warm-up (frozen backbone) - 5 epochs
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Warm-up (frozen backbone) - 5 epochs")
    logger.info("=" * 60)
    
    model.freeze_backbone()
    warmup_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    for epoch in range(5):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, warmup_optimizer, device, scaler, use_amp
        )
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)
        
        logger.info(
            f"Epoch {epoch+1}/5 | "
            f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
            f"Val: {val_loss:.4f} / {val_acc:.1f}%"
        )
    
    # ============================================================
    # PHASE 2: Full training - N epochs
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Full training - {args.epochs} epochs")
    logger.info("=" * 60)
    
    model.unfreeze_backbone()
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["per_class_acc"].append(per_class)
        history["lr"].append(current_lr)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f} | "
            f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
            f"Val: {val_loss:.4f} / {val_acc:.1f}%"
        )
        
        # Per-class accuracy (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            logger.info("  Per-class accuracy:")
            for chart_type, acc in sorted(per_class.items()):
                logger.info(f"    {chart_type:12}: {acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_mapping": class_mapping,
                "val_acc": val_acc,
                "epoch": epoch,
                "history": history,
            }
            save_path = output_dir / "resnet18_chart_classifier_v2_best.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"  [SAVED] Best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  [EARLY STOP] No improvement for {args.patience} epochs")
                break
    
    # ============================================================
    # FINAL TEST
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST")
    logger.info("=" * 60)
    
    # Load best model
    checkpoint = torch.load(output_dir / "resnet18_chart_classifier_v2_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_per_class = validate(model, test_loader, criterion, device)
    
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info("Per-class accuracy:")
    for chart_type, acc in sorted(test_per_class.items()):
        logger.info(f"  {chart_type:12}: {acc:.1f}%")
    
    # Save final results
    elapsed = datetime.now() - start_time
    results = {
        "test_acc": test_acc,
        "test_per_class": test_per_class,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "training_time": str(elapsed),
        "class_mapping": class_mapping,
    }
    
    with open(output_dir / "resnet18_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTraining completed in {elapsed}")
    logger.info(f"Best model saved to: {output_dir / 'resnet18_chart_classifier_v2_best.pt'}")


if __name__ == "__main__":
    main()
