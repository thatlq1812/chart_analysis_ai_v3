"""
ResNet-18 Chart Classifier for Stage 3

Replaces SimpleChartClassifier with deep learning model.
Achieves 94.66% accuracy vs 37.5% baseline.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class ResNet18Classifier:
    """
    ResNet-18 based chart classifier
    
    Features:
    - Transfer learning from ImageNet
    - 94.66% test accuracy (vs 37.5% baseline)
    - Supports 8 chart types
    - GPU acceleration when available
    
    Usage:
        classifier = ResNet18Classifier(model_path, device='auto')
        chart_type = classifier.predict(image_path)
        chart_type, confidence = classifier.predict_with_confidence(image_path)
    """
    
    def __init__(
        self,
        model_path: Path,
        device: str = 'auto',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize ResNet-18 classifier
        
        Args:
            model_path: Path to trained model weights (.pt file)
            device: Device to use ('auto', 'cuda', 'cpu', 'mps')
            confidence_threshold: Minimum confidence for valid prediction
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"ResNet18Classifier | device={self.device} | model={model_path.name}")
        
        # Load model
        self._load_model()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get class names from checkpoint
        if 'class_mapping' in checkpoint:
            # class_mapping is {class_name: index}
            mapping = checkpoint['class_mapping']
            # Sort by index to get ordered list
            self.class_names = sorted(mapping.keys(), key=lambda x: mapping[x])
        elif 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            # Default order (alphabetical)
            self.class_names = [
                'area', 'bar', 'box', 'heatmap',
                'histogram', 'line', 'pie', 'scatter'
            ]
        
        self.num_classes = len(self.class_names)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        # Build model wrapper (matches training structure with Sequential FC)
        class ResNetWrapper(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.resnet = models.resnet18(pretrained=False)
                in_features = self.resnet.fc.in_features
                # Use Sequential to match training structure: resnet.fc.1.weight
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, num_classes)
                )
            
            def forward(self, x):
                return self.resnet(x)
        
        model = ResNetWrapper(self.num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        
        logger.info(f"Model loaded | classes={self.num_classes} | {self.class_names}")
    
    def predict(self, image_path: Path) -> str:
        """
        Predict chart type from image
        
        Args:
            image_path: Path to chart image
        
        Returns:
            Chart type string (e.g., 'line', 'bar', 'scatter')
        """
        chart_type, _ = self.predict_with_confidence(image_path)
        return chart_type
    
    def predict_with_confidence(self, image_input) -> tuple[str, float]:
        """
        Predict chart type with confidence score
        
        Args:
            image_input: Path to chart image OR BGR numpy array
        
        Returns:
            Tuple of (chart_type, confidence)
            If confidence < threshold, returns ('unknown', confidence)
        """
        try:
            # Handle different input types
            if isinstance(image_input, (Path, str)):
                # Load from path
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Convert BGR numpy array to PIL RGB
                import cv2
                rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_array)
            else:
                raise ValueError(f"Unsupported input type: {type(image_input)}")
            
            # Transform
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence = confidence.item()
                pred_idx = pred_idx.item()
            
            # Get chart type
            chart_type = self.class_names[pred_idx]
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.warning(
                    f"Low confidence prediction | "
                    f"pred={chart_type} | "
                    f"conf={confidence:.3f} < threshold={self.confidence_threshold}"
                )
                return 'unknown', confidence
            
            return chart_type, confidence
        
        except Exception as e:
            logger.error(f"Prediction failed | error={e}")
            return 'unknown', 0.0
    
    def predict_batch(self, image_paths: list[Path]) -> list[tuple[str, float]]:
        """
        Predict chart types for batch of images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of (chart_type, confidence) tuples
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_with_confidence(image_path)
            results.append(result)
        
        return results
    
    def get_class_probabilities(self, image_path: Path) -> dict[str, float]:
        """
        Get probability distribution over all classes
        
        Args:
            image_path: Path to chart image
        
        Returns:
            Dict mapping class names to probabilities
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                probs = probs.cpu().numpy()[0]
            
            # Create dict
            prob_dict = {
                class_name: float(probs[idx])
                for class_name, idx in self.class_to_idx.items()
            }
            
            return prob_dict
        
        except Exception as e:
            logger.error(f"Probability computation failed | image={image_path} | error={e}")
            return {cls: 0.0 for cls in self.class_names}


# ============ Factory Function ============

def create_resnet_classifier(
    model_path: Optional[Path] = None,
    device: str = 'auto',
    confidence_threshold: float = 0.5
) -> ResNet18Classifier:
    """
    Factory function to create ResNet-18 classifier
    
    Args:
        model_path: Path to model weights (default: auto-detect)
        device: Device to use
        confidence_threshold: Minimum confidence
    
    Returns:
        Initialized ResNet18Classifier
    """
    if model_path is None:
        # Auto-detect model path - try v2 first (94.14% accuracy)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        v2_path = project_root / "models" / "weights" / "resnet18_chart_classifier_v2_best.pt"
        v1_path = project_root / "models" / "weights" / "resnet18_chart_classifier_best.pt"
        
        if v2_path.exists():
            model_path = v2_path
        elif v1_path.exists():
            model_path = v1_path
        else:
            raise FileNotFoundError(f"No ResNet model found at {v2_path} or {v1_path}")
    
    return ResNet18Classifier(model_path, device, confidence_threshold)
