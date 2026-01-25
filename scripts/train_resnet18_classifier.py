"""
ResNet-18 Chart Classifier Training Pipeline

Based on Gemini 3 Pro recommendations:
- Transfer learning from ImageNet pretrained ResNet-18
- Target accuracy: >80% on real academic charts
- Explainable via Grad-CAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Tuple
from collections import Counter
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChartDataset(Dataset):
    """
    Dataset for chart classification.
    
    Expected structure:
        data/academic_dataset/images/
            bar/
            line/
            pie/
            scatter/
            ...
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        transform=None,
        class_mapping: Dict[str, int] = None
    ):
        """
        Args:
            root_dir: Path to images directory
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
            class_mapping: Dict mapping class names to indices
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load dataset manifest
        manifest_path = self.root_dir.parent / "manifests" / f"{split}_manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        # Build class mapping if not provided
        if class_mapping is None:
            all_types = set(item["chart_type"] for item in self.manifest)
            self.class_mapping = {cls: idx for idx, cls in enumerate(sorted(all_types))}
        else:
            self.class_mapping = class_mapping
        
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        logger.info(
            f"ChartDataset loaded | split={split} | "
            f"samples={len(self.manifest)} | classes={len(self.class_mapping)}"
        )
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.manifest[idx]
        
        # Load image
        img_path = self.root_dir / item["image_path"]
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_mapping[item["chart_type"]]
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset."""
        types = [item["chart_type"] for item in self.manifest]
        return dict(Counter(types))


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train/val/test dataloaders with proper augmentation.
    
    Args:
        data_dir: Path to data/academic_dataset/images/
        batch_size: Batch size for training
        num_workers: Number of worker processes
        image_size: Input image size (224 for ResNet)
    
    Returns:
        train_loader, val_loader, test_loader, class_mapping
    """
    
    # ImageNet normalization (ResNet pretrained on ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Data augmentation per Gemini recommendations
        transforms.RandomApply([
            transforms.Grayscale(num_output_channels=3)  # 20% grayscale (academic papers)
        ], p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=5, shear=5),  # Slight rotation/shear
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Create datasets
    train_dataset = ChartDataset(data_dir, split="train", transform=train_transform)
    val_dataset = ChartDataset(
        data_dir, split="val", transform=eval_transform,
        class_mapping=train_dataset.class_mapping
    )
    test_dataset = ChartDataset(
        data_dir, split="test", transform=eval_transform,
        class_mapping=train_dataset.class_mapping
    )
    
    # Handle class imbalance with WeightedRandomSampler
    class_counts = train_dataset.get_class_distribution()
    logger.info(f"Training class distribution: {class_counts}")
    
    # Calculate sample weights (inverse frequency)
    weights = []
    for item in train_dataset.manifest:
        chart_type = item["chart_type"]
        weight = 1.0 / class_counts[chart_type]
        weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
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
    
    return train_loader, val_loader, test_loader, train_dataset.class_mapping


class ResNetChartClassifier(nn.Module):
    """
    ResNet-18 based chart classifier.
    
    Architecture:
    - Pretrained ResNet-18 backbone (ImageNet weights)
    - Custom classification head
    - Explainable via Grad-CAM
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace final FC layer
        in_features = self.resnet.fc.in_features  # 512 for ResNet-18
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
        logger.info(
            f"ResNetChartClassifier initialized | "
            f"num_classes={num_classes} | pretrained={pretrained}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze early layers for initial training."""
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:  # Don't freeze final layer
                param.requires_grad = False
        logger.info("Backbone frozen (only training final layer)")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen (training all layers)")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
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
            "acc": f"{100. * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ResNet-18 Chart Classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/weights", help="Output directory")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    data_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader, class_mapping = create_dataloaders(
        data_dir, batch_size=args.batch_size
    )
    
    # Create model
    num_classes = len(class_mapping)
    model = ResNetChartClassifier(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    
    # Phase 1: Train only final layer (5 epochs)
    logger.info("=" * 60)
    logger.info("PHASE 1: Training final layer only (frozen backbone)")
    logger.info("=" * 60)
    model.freeze_backbone()
    
    for epoch in range(5):
        logger.info(f"\nEpoch {epoch+1}/5")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
    
    # Phase 2: Fine-tune entire network
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Fine-tuning entire network")
    logger.info("=" * 60)
    model.unfreeze_backbone()
    
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_mapping": class_mapping,
                "val_acc": val_acc,
                "epoch": epoch,
            }
            torch.save(checkpoint, output_dir / "resnet18_chart_classifier_best.pt")
            logger.info(f"[SAVED] Best model (val_acc={val_acc:.2f}%)")
    
    # Final test
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST")
    logger.info("=" * 60)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
