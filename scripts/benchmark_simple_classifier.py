#!/usr/bin/env python3
"""
Benchmark for Simple Chart Classifier.

Tests the image-based classifier against ground truth labels.

Usage:
    python scripts/benchmark_simple_classifier.py
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core_engine.stages.s3_extraction.simple_classifier import (
    SimpleChartClassifier,
    SimpleClassifierConfig,
)

logging.basicConfig(level=logging.WARNING)


def load_ground_truth(dataset_path: Path) -> Dict[str, str]:
    """Load ground truth labels from dataset.json."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ground_truth = {}
    for sample in data["samples"]:
        image_id = sample["image_id"]
        chart_type = sample["chart_type"]
        ground_truth[image_id] = chart_type
    
    return ground_truth


def process_image(
    image_path: Path,
    classifier: SimpleChartClassifier,
) -> Tuple[str, float, Dict]:
    """Process single image and return classification."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return "error", 0.0, {"error": "Failed to load image"}
        
        # Resize large images
        h, w = image.shape[:2]
        max_dim = 600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        result = classifier.classify(image, chart_id="benchmark")
        
        return (
            result.chart_type.value,
            result.confidence,
            result.features,
        )
    
    except Exception as e:
        return "error", 0.0, {"error": str(e)}


def main():
    """Run benchmark."""
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data/academic_dataset/chart_qa/dataset.json"
    images_dir = project_root / "data/academic_dataset/images"
    
    print("=" * 70)
    print("SIMPLE CHART CLASSIFIER BENCHMARK")
    print("=" * 70)
    
    # Load ground truth
    print("\n[1/4] Loading ground truth...")
    ground_truth = load_ground_truth(dataset_path)
    print(f"      Loaded {len(ground_truth)} labels")
    
    # Initialize classifier
    print("\n[2/4] Initializing classifier...")
    classifier = SimpleChartClassifier(SimpleClassifierConfig())
    print("      Classifier initialized")
    
    # Sample images by type
    print("\n[3/4] Running benchmark...")
    
    type_samples: Dict[str, List[str]] = defaultdict(list)
    for image_id, chart_type in ground_truth.items():
        type_samples[chart_type].append(image_id)
    
    # Sample
    max_per_type = 30
    sampled_ids = []
    for chart_type, ids in type_samples.items():
        np.random.seed(42)
        n_sample = min(len(ids), max_per_type)
        sampled = np.random.choice(ids, n_sample, replace=False)
        sampled_ids.extend([(s, chart_type) for s in sampled])
        print(f"      {chart_type}: sampling {n_sample}/{len(ids)} images")
    
    # Run classification
    results = []
    confusion = defaultdict(lambda: defaultdict(int))
    
    print(f"\n      Processing {len(sampled_ids)} images...")
    
    for i, (image_id, actual_type) in enumerate(sampled_ids):
        image_path = images_dir / f"{image_id}.png"
        
        predicted, confidence, features = process_image(image_path, classifier)
        
        results.append({
            "image_id": image_id,
            "actual": actual_type,
            "predicted": predicted,
            "confidence": confidence,
            "features": features,
        })
        
        confusion[actual_type][predicted] += 1
        
        if (i + 1) % 50 == 0:
            print(f"      Processed {i + 1}/{len(sampled_ids)} images...")
    
    # Analyze results
    print("\n[4/4] Analyzing results...")
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["actual"] == r["predicted"])
    total = len(results)
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for r in results:
        class_total[r["actual"]] += 1
        if r["actual"] == r["predicted"]:
            class_correct[r["actual"]] += 1
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    for chart_type in sorted(class_total.keys()):
        acc = 100 * class_correct[chart_type] / class_total[chart_type]
        print(f"  {chart_type:12}: {class_correct[chart_type]:3}/{class_total[chart_type]:3} ({acc:5.1f}%)")
    
    # Confusion matrix
    print("\nConfusion Matrix (Actual -> Predicted):")
    print("-" * 70)
    
    # Get all types
    all_types = sorted(set(confusion.keys()) | set(
        pred for actual_preds in confusion.values() for pred in actual_preds
    ))
    
    # Header
    header = "Actual\\Pred".ljust(12) + "".join(f"{t[:7]:>8}" for t in all_types)
    print(header)
    
    # Rows
    for actual in sorted(confusion.keys()):
        row = f"{actual:12}"
        for predicted in all_types:
            count = confusion[actual].get(predicted, 0)
            row += f"{count:8}"
        print(row)
    
    # Feature analysis
    print("\nFeature Analysis (Average by Actual Type):")
    print("-" * 70)
    
    feature_names = ["circularity", "n_rectangles", "n_markers", "grid_score"]
    header = "Type".ljust(12) + "".join(f"{f:>14}" for f in feature_names)
    print(header)
    print("-" * 70)
    
    for chart_type in sorted(class_total.keys()):
        type_results = [r for r in results if r["actual"] == chart_type and "error" not in r["features"]]
        if not type_results:
            continue
        
        row = f"{chart_type:12}"
        for feat in feature_names:
            values = [r["features"].get(feat, 0) for r in type_results]
            avg = sum(values) / len(values) if values else 0
            row += f"{avg:14.2f}"
        print(row)
    
    # Save results
    output_dir = project_root / "docs/reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "simple_classifier_benchmark.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": correct / total,
            "total_images": total,
            "per_class_accuracy": {k: class_correct[k] / class_total[k] for k in class_total},
            "confusion_matrix": dict(confusion),
            "results": results,
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
