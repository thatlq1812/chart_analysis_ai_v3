"""
Generate Grad-CAM Heatmaps for ResNet-18 Chart Classifier

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions
the model focuses on when making predictions.

Usage:
    python scripts/generate_gradcam.py --num-samples 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


# ============ Grad-CAM Implementation ============

class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks
    https://arxiv.org/abs/1610.02391
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate CAM heatmap for target class
        
        Args:
            image: Input tensor [1, C, H, W]
            target_class: Target class index
        
        Returns:
            CAM heatmap [H, W] normalized to [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(image)
        
        # Backward pass for target class
        self.model.zero_grad()
        target = output[:, target_class]
        target.backward()
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image [H, W, 3] in range [0, 255]
            heatmap: CAM heatmap [H, W] in range [0, 1]
            alpha: Blending factor
        
        Returns:
            Overlayed image [H, W, 3]
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap (jet)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlayed = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
        
        return overlayed


# ============ Visualization ============

def plot_gradcam_grid(
    samples: List[dict],
    output_path: Path
):
    """
    Plot grid of Grad-CAM visualizations
    
    Args:
        samples: List of {image, heatmap, overlayed, label, prediction, confidence}
        output_path: Output PNG path
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Original image
        axes[i, 0].imshow(sample['image'])
        axes[i, 0].set_title(f"Original\nTrue: {sample['label']}")
        axes[i, 0].axis('off')
        
        # Heatmap
        axes[i, 1].imshow(sample['heatmap'], cmap='jet')
        axes[i, 1].set_title(f"Grad-CAM Heatmap")
        axes[i, 1].axis('off')
        
        # Overlayed
        axes[i, 2].imshow(sample['overlayed'])
        title = f"Overlay\nPred: {sample['prediction']} ({sample['confidence']:.1f}%)"
        if sample['label'] != sample['prediction']:
            title += " ❌"
        else:
            title += " ✓"
        axes[i, 2].set_title(title)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Grad-CAM visualization saved: {output_path}")
    plt.close()


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples per class (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: models/explainability)'
    )
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "academic_dataset"
    images_dir = data_dir / "images"
    test_manifest = data_dir / "manifests" / "test_manifest.json"
    model_path = project_root / "models" / "weights" / "resnet18_chart_classifier_best.pt"
    
    output_dir = args.output_dir or project_root / "models" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test samples
    with open(test_manifest, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    # Group by chart type
    samples_by_type = {}
    for sample in test_samples:
        chart_type = sample['chart_type']
        if chart_type not in samples_by_type:
            samples_by_type[chart_type] = []
        samples_by_type[chart_type].append(sample)
    
    class_names = sorted(samples_by_type.keys())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    
    # Load model
    num_classes = len(class_names)
    model = ResNetChartClassifier(num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {model_path}")
    
    # Get target layer (last conv layer in ResNet-18)
    target_layer = model.resnet.layer4[-1].conv2
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Process samples for each class
    logger.info(f"Generating Grad-CAM for {args.num_samples} samples per class...")
    
    for chart_type in class_names:
        logger.info(f"Processing: {chart_type}")
        
        # Select random samples
        type_samples = samples_by_type[chart_type]
        selected = np.random.choice(
            len(type_samples),
            min(args.num_samples, len(type_samples)),
            replace=False
        )
        
        results = []
        
        for idx in selected:
            sample = type_samples[idx]
            
            # Load image
            image_path = images_dir / sample['image_path']
            image_pil = Image.open(image_path).convert('RGB')
            image_np = np.array(image_pil)
            
            # Transform for model
            image_tensor = transform(image_pil).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item() * 100
            
            # Generate Grad-CAM for predicted class
            heatmap = gradcam.generate_cam(image_tensor, pred_idx)
            
            # Overlay heatmap
            overlayed = gradcam.overlay_heatmap(image_np, heatmap, alpha=0.4)
            
            results.append({
                'image': image_np,
                'heatmap': heatmap,
                'overlayed': overlayed,
                'label': chart_type,
                'prediction': class_names[pred_idx],
                'confidence': confidence
            })
        
        # Save visualization
        output_path = output_dir / f"gradcam_{chart_type}.png"
        plot_gradcam_grid(results, output_path)
    
    # Generate combined summary
    logger.info("Generating combined summary...")
    
    all_results = []
    for chart_type in class_names:
        type_samples = samples_by_type[chart_type]
        sample = type_samples[0]  # Take first sample
        
        # Load and process
        image_path = images_dir / sample['image_path']
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item() * 100
        
        # Grad-CAM
        heatmap = gradcam.generate_cam(image_tensor, pred_idx)
        overlayed = gradcam.overlay_heatmap(image_np, heatmap, alpha=0.4)
        
        all_results.append({
            'image': image_np,
            'heatmap': heatmap,
            'overlayed': overlayed,
            'label': chart_type,
            'prediction': class_names[pred_idx],
            'confidence': confidence
        })
    
    summary_path = output_dir / "gradcam_summary_all_classes.png"
    plot_gradcam_grid(all_results, summary_path)
    
    logger.info("=" * 60)
    logger.info(f"Grad-CAM visualizations complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated {len(class_names)} per-class visualizations")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
