"""
Integration Test: ResNet-18 Classifier in Stage 3 Pipeline

Tests end-to-end chart analysis with new ResNet-18 classifier.

Comparison:
- OLD: ChartClassifier (rule-based) → 37.5% accuracy
- NEW: ResNet18Classifier (deep learning) → 94.66% accuracy

Usage:
    python scripts/test_resnet_integration.py --num-samples 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core_engine.stages.s3_extraction.resnet_classifier import ResNet18Classifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_test_samples(manifest_path: Path, num_samples: int) -> List[dict]:
    """Load test samples from manifest"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Sample stratified by chart type
    samples_by_type = {}
    for sample in samples:
        chart_type = sample['chart_type']
        if chart_type not in samples_by_type:
            samples_by_type[chart_type] = []
        samples_by_type[chart_type].append(sample)
    
    # Take samples from each type
    selected = []
    per_type = max(1, num_samples // len(samples_by_type))
    
    for chart_type, type_samples in samples_by_type.items():
        selected.extend(type_samples[:per_type])
    
    return selected[:num_samples]


def test_single_image(
    classifier: ResNet18Classifier,
    image_path: Path,
    ground_truth: str
) -> dict:
    """Test classification on single image"""
    try:
        # Predict
        predicted_type, confidence = classifier.predict_with_confidence(image_path)
        
        # Get all probabilities
        probs = classifier.get_class_probabilities(image_path)
        top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Check correctness
        is_correct = predicted_type == ground_truth
        
        return {
            'image': image_path.name,
            'ground_truth': ground_truth,
            'predicted': predicted_type,
            'confidence': confidence,
            'correct': is_correct,
            'top_3_probs': {k: f"{v:.3f}" for k, v in top_3}
        }
    
    except Exception as e:
        logger.error(f"Failed to classify {image_path.name}: {e}")
        return {
            'image': image_path.name,
            'ground_truth': ground_truth,
            'predicted': 'error',
            'confidence': 0.0,
            'correct': False,
            'error': str(e)
        }


def visualize_results(
    results: List[dict],
    images_dir: Path,
    output_dir: Path
):
    """Create visualization grid of results"""
    import matplotlib.pyplot as plt
    from matplotlib import patches
    
    # Select 8 samples (4 correct, 4 incorrect if available)
    correct = [r for r in results if r['correct']]
    incorrect = [r for r in results if not r['correct'] and 'error' not in r]
    
    selected = correct[:4] + incorrect[:4]
    
    if len(selected) == 0:
        logger.warning("No samples to visualize")
        return
    
    n_samples = len(selected)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(selected):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Load image
        image_path = images_dir / result['image']
        if not image_path.exists():
            ax.axis('off')
            continue
        
        image = Image.open(image_path).convert('RGB')
        ax.imshow(image)
        
        # Title
        is_correct = result['correct']
        status = "✓" if is_correct else "✗"
        color = 'green' if is_correct else 'red'
        
        title = f"{status} GT: {result['ground_truth']}\nPred: {result['predicted']} ({result['confidence']:.2f})"
        ax.set_title(title, fontsize=12, color=color, fontweight='bold')
        ax.axis('off')
        
        # Add border
        border_color = 'green' if is_correct else 'red'
        rect = patches.Rectangle(
            (0, 0), image.width, image.height,
            linewidth=5, edgecolor=border_color, facecolor='none',
            transform=ax.transData
        )
        ax.add_patch(rect)
    
    # Hide unused subplots
    for i in range(len(selected), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    output_path = output_dir / "resnet_integration_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test ResNet-18 integration")
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of test samples (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: data/output)'
    )
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "academic_dataset"
    images_dir = data_dir / "images"
    test_manifest = data_dir / "manifests" / "test_manifest.json"
    model_path = project_root / "models" / "weights" / "resnet18_chart_classifier_best.pt"
    
    output_dir = args.output_dir or project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check files exist
    if not test_manifest.exists():
        logger.error(f"Test manifest not found: {test_manifest}")
        return
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Initialize ResNet-18 classifier
    logger.info("Initializing ResNet-18 classifier...")
    classifier = ResNet18Classifier(
        model_path=model_path,
        device='auto',
        confidence_threshold=0.5
    )
    
    # Load test samples
    logger.info(f"Loading {args.num_samples} test samples...")
    samples = load_test_samples(test_manifest, args.num_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Test each sample
    logger.info("Running classification tests...")
    results = []
    
    for i, sample in enumerate(samples, 1):
        image_path = images_dir / sample['image_path']
        ground_truth = sample['chart_type']
        
        logger.info(f"[{i}/{len(samples)}] Testing: {image_path.name}")
        
        result = test_single_image(classifier, image_path, ground_truth)
        results.append(result)
        
        # Log result
        status = "✓" if result['correct'] else "✗"
        logger.info(
            f"  {status} GT: {result['ground_truth']} | "
            f"Pred: {result['predicted']} ({result['confidence']:.3f})"
        )
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Per-class accuracy
    per_class = {}
    for result in results:
        gt = result['ground_truth']
        if gt not in per_class:
            per_class[gt] = {'correct': 0, 'total': 0}
        per_class[gt]['total'] += 1
        if result['correct']:
            per_class[gt]['correct'] += 1
    
    per_class_acc = {
        k: (v['correct'] / v['total'] * 100)
        for k, v in per_class.items()
    }
    
    # Print summary
    logger.info("=" * 70)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total Samples:   {total}")
    logger.info(f"Correct:         {correct}")
    logger.info(f"Incorrect:       {total - correct}")
    logger.info(f"Accuracy:        {accuracy:.2f}%")
    logger.info("")
    logger.info("Per-Class Accuracy:")
    for chart_type in sorted(per_class_acc.keys()):
        acc = per_class_acc[chart_type]
        count = per_class[chart_type]['total']
        logger.info(f"  {chart_type:12s}: {acc:6.2f}% ({per_class[chart_type]['correct']}/{count})")
    logger.info("=" * 70)
    
    # Save detailed results
    results_json = {
        'summary': {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        },
        'detailed_results': results
    }
    
    results_path = output_dir / "resnet_integration_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved: {results_path}")
    
    # Create visualization
    logger.info("Creating visualization...")
    visualize_results(results, images_dir, output_dir)
    
    logger.info("=" * 70)
    logger.info("INTEGRATION TEST COMPLETE")
    logger.info("=" * 70)
    
    # Return exit code based on accuracy threshold
    if accuracy >= 90:
        logger.info("✓ PASSED: Accuracy >= 90%")
        return 0
    else:
        logger.warning(f"✗ FAILED: Accuracy {accuracy:.2f}% < 90%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
