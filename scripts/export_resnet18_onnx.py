"""
Export ResNet-18 Chart Classifier to ONNX Format

ONNX (Open Neural Network Exchange) enables cross-platform deployment.

Usage:
    python scripts/export_resnet18_onnx.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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


# ============ ONNX Export ============

def export_to_onnx(
    model: nn.Module,
    onnx_path: Path,
    device: torch.device,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 11
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        onnx_path: Output ONNX file path
        device: Device model is on
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version
    """
    # Move model to CPU for ONNX export
    model = model.cpu()
    model.eval()
    
    # Create dummy input on CPU
    dummy_input = torch.randn(input_shape)
    
    # Export
    logger.info(f"Exporting to ONNX | input_shape={input_shape} | opset={opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"ONNX model saved: {onnx_path}")
    logger.info(f"File size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")


# ============ ONNX Validation ============

def validate_onnx(
    onnx_path: Path,
    pytorch_model: nn.Module,
    test_image_path: Path,
    class_names: list,
    device: torch.device
):
    """
    Validate ONNX model by comparing with PyTorch model
    
    Args:
        onnx_path: ONNX model path
        pytorch_model: Original PyTorch model
        test_image_path: Test image for validation
        class_names: List of class names
        device: PyTorch device
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnx or onnxruntime not installed. Skipping validation.")
        logger.warning("Install with: pip install onnx onnxruntime")
        return
    
    logger.info("Validating ONNX model...")
    
    # Check ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid")
    
    # Load test image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor.to(device))
        pytorch_probs = torch.nn.functional.softmax(pytorch_output, dim=1)
        pytorch_pred = torch.argmax(pytorch_probs, dim=1).item()
        pytorch_conf = pytorch_probs[0, pytorch_pred].item()
    
    # ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_input = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    onnx_probs = torch.nn.functional.softmax(torch.from_numpy(onnx_output), dim=1)
    onnx_pred = torch.argmax(onnx_probs, dim=1).item()
    onnx_conf = onnx_probs[0, onnx_pred].item()
    
    # Compare
    logger.info("=" * 60)
    logger.info("Inference Comparison:")
    logger.info(f"  PyTorch: {class_names[pytorch_pred]} ({pytorch_conf * 100:.2f}%)")
    logger.info(f"  ONNX:    {class_names[onnx_pred]} ({onnx_conf * 100:.2f}%)")
    
    if pytorch_pred == onnx_pred:
        logger.info("✓ Predictions match!")
    else:
        logger.warning("✗ Predictions differ!")
    
    # Check numerical difference
    diff = np.abs(pytorch_output.cpu().numpy() - onnx_output).max()
    logger.info(f"  Max output difference: {diff:.6f}")
    
    if diff < 1e-5:
        logger.info("✓ Outputs are numerically close")
    else:
        logger.warning(f"✗ Large output difference: {diff}")
    
    logger.info("=" * 60)


# ============ Benchmark ============

def benchmark_inference(
    onnx_path: Path,
    num_iterations: int = 100
):
    """
    Benchmark ONNX model inference speed
    
    Args:
        onnx_path: ONNX model path
        num_iterations: Number of iterations for benchmarking
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed. Skipping benchmark.")
        return
    
    logger.info(f"Benchmarking ONNX inference ({num_iterations} iterations)...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input}
    
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, onnx_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = ort_session.run(None, onnx_input)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    times = np.array(times)
    
    logger.info("=" * 60)
    logger.info("Inference Speed (ONNX Runtime - CPU):")
    logger.info(f"  Mean:   {times.mean():.2f} ms")
    logger.info(f"  Median: {np.median(times):.2f} ms")
    logger.info(f"  Min:    {times.min():.2f} ms")
    logger.info(f"  Max:    {times.max():.2f} ms")
    logger.info(f"  Throughput: {1000 / times.mean():.1f} images/sec")
    logger.info("=" * 60)


# ============ Main ============

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "academic_dataset"
    images_dir = data_dir / "images"
    test_manifest = data_dir / "manifests" / "test_manifest.json"
    model_path = project_root / "models" / "weights" / "resnet18_chart_classifier_best.pt"
    
    onnx_dir = project_root / "models" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "resnet18_chart_classifier.onnx"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load class names
    with open(test_manifest, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    class_names = sorted(set(s['chart_type'] for s in test_samples))
    num_classes = len(class_names)
    
    # Load PyTorch model
    model = ResNetChartClassifier(num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"PyTorch model loaded: {model_path}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Export to ONNX
    export_to_onnx(model, onnx_path, device, input_shape=(1, 3, 224, 224), opset_version=11)
    
    # Move model back to original device for validation
    model = model.to(device)
    
    # Get test image
    test_image_path = images_dir / test_samples[0]['image_path']
    
    # Validate ONNX model
    validate_onnx(onnx_path, model, test_image_path, class_names, device)
    
    # Benchmark
    benchmark_inference(onnx_path, num_iterations=100)
    
    # Save metadata
    metadata = {
        'model_name': 'ResNet18ChartClassifier',
        'input_shape': [1, 3, 224, 224],
        'output_shape': [1, num_classes],
        'class_names': class_names,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'opset_version': 11,
        'framework': 'PyTorch 2.0+',
        'exported_from': str(model_path)
    }
    
    metadata_path = onnx_dir / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadata saved: {metadata_path}")
    logger.info("=" * 60)
    logger.info("ONNX export complete!")
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Metadata:   {metadata_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
