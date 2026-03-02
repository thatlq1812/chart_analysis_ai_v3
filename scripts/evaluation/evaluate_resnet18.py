"""
Evaluate ResNet-18 Chart Classifier on Test Set

Generates:
- Per-class accuracy metrics
- Confusion matrix (heatmap)
- Classification report
- Misclassified examples
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============ Model Definition ============

class ResNetChartClassifier(nn.Module):
    """ResNet-18 based chart classifier"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# ============ Dataset ============

class ChartDataset(Dataset):
    """Dataset for chart images"""
    
    def __init__(self, manifest_path: Path, images_dir: Path, transform=None):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        self.images_dir = images_dir
        self.transform = transform
        
        # Build class mapping
        self.classes = sorted(set(s['chart_type'] for s in self.samples))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        logger.info(f"ChartDataset loaded | samples={len(self.samples)} | classes={len(self.classes)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.images_dir / sample['image_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[sample['chart_type']]
        
        return image, label, sample['image_path']


# ============ Evaluation ============

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Evaluate model on test set
    
    Returns:
        Dict with metrics, predictions, confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, labels, image_paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_image_paths.extend(image_paths)
    
    # Compute metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_correct = np.sum(
                (np.array(all_preds) == i) & class_mask
            )
            per_class_acc[class_name] = class_correct / class_mask.sum()
        else:
            per_class_acc[class_name] = 0.0
    
    # Find misclassified samples
    misclassified = []
    for pred, label, path in zip(all_preds, all_labels, all_image_paths):
        if pred != label:
            misclassified.append({
                'image_path': path,
                'true_label': class_names[label],
                'predicted_label': class_names[pred]
            })
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'misclassified': misclassified,
        'class_names': class_names
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    plt.title('ResNet-18 Chart Classifier - Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved: {output_path}")
    plt.close()


def plot_per_class_accuracy(per_class_acc: Dict[str, float], output_path: Path):
    """Plot per-class accuracy bar chart"""
    classes = list(per_class_acc.keys())
    accuracies = [per_class_acc[c] * 100 for c in classes]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes, accuracies, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='100%')
    plt.axhline(y=94.4, color='red', linestyle='--', alpha=0.5, label='Overall (94.4%)')
    
    plt.title('ResNet-18 Per-Class Accuracy', fontsize=16, pad=20)
    plt.xlabel('Chart Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Per-class accuracy plot saved: {output_path}")
    plt.close()


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "academic_dataset"
    images_dir = data_dir / "images"
    test_manifest = data_dir / "manifests" / "test_manifest.json"
    model_path = project_root / "models" / "weights" / "resnet18_chart_classifier_best.pt"
    
    output_dir = project_root / "models" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Transforms (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    test_dataset = ChartDataset(test_manifest, images_dir, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    num_classes = len(test_dataset.classes)
    model = ResNetChartClassifier(num_classes, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded from: {model_path}")
    if 'val_accuracy' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']} | Val Acc: {checkpoint['val_accuracy']:.2f}%")
    else:
        logger.info(f"Checkpoint loaded (best model)")
    
    # Evaluate
    logger.info("Evaluating on test set...")
    results = evaluate_model(model, test_loader, device, test_dataset.classes)
    
    # Print results
    logger.info("=" * 60)
    logger.info(f"TEST ACCURACY: {results['accuracy'] * 100:.2f}%")
    logger.info("=" * 60)
    logger.info("\nPer-Class Accuracy:")
    for class_name, acc in results['per_class_accuracy'].items():
        logger.info(f"  {class_name:12s}: {acc * 100:6.2f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("Classification Report:")
    logger.info("=" * 60)
    print(results['classification_report'])
    
    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        results['confusion_matrix'],
        results['class_names'],
        cm_path
    )
    
    # Plot per-class accuracy
    acc_path = output_dir / "per_class_accuracy.png"
    plot_per_class_accuracy(results['per_class_accuracy'], acc_path)
    
    # Save results to JSON
    results_json = {
        'overall_accuracy': results['accuracy'],
        'per_class_accuracy': results['per_class_accuracy'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'class_names': results['class_names'],
        'total_samples': len(test_dataset),
        'num_misclassified': len(results['misclassified']),
        'misclassified_samples': results['misclassified'][:20]  # First 20
    }
    
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Total misclassified: {len(results['misclassified'])} / {len(test_dataset)}")


if __name__ == "__main__":
    main()
