#!/usr/bin/env python
"""
Benchmark Chart Classifier on Academic Dataset

This script tests the current classifier against ground truth labels
and identifies misclassifications for improvement.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core_engine.stages.s3_extraction.preprocessor import (
    ImagePreprocessor,
    PreprocessConfig,
)
from core_engine.stages.s3_extraction.skeletonizer import (
    Skeletonizer,
    SkeletonConfig,
)
from core_engine.stages.s3_extraction.vectorizer import (
    Vectorizer,
    VectorizeConfig,
)
from core_engine.stages.s3_extraction.element_detector import (
    ElementDetector,
    ElementDetectorConfig,
)
from core_engine.stages.s3_extraction.classifier import (
    ChartClassifier,
    ClassifierConfig,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_ground_truth(dataset_path: Path) -> Dict[str, str]:
    """Load ground truth chart types from dataset.json."""
    with open(dataset_path) as f:
        data = json.load(f)
    
    ground_truth = {}
    for sample in data["samples"]:
        image_id = sample["image_id"]
        chart_type = sample["chart_type"]
        ground_truth[image_id] = chart_type
    
    return ground_truth


def process_image(
    image_path: Path,
    preprocessor: ImagePreprocessor,
    skeletonizer: Skeletonizer,
    vectorizer: Vectorizer,
    element_detector: ElementDetector,
    classifier: ChartClassifier,
) -> Tuple[str, float, Dict]:
    """Process single image and return classification."""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return "error", 0.0, {"error": "Failed to load image"}
        
        # Resize large images to speed up processing
        h, w = image.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            h, w = new_h, new_w
        
        # Step 1: Preprocessing
        preprocess_result = preprocessor.process(image)
        
        # Step 2: Skeletonization
        skeleton_result = skeletonizer.process(
            preprocess_result.binary_image,
            chart_id="benchmark",
        )
        
        # Step 3: Trace paths from skeleton
        paths = skeletonizer.trace_paths(
            skeleton_result.skeleton,
            skeleton_result.keypoints,
        )
        
        # Step 4: Vectorization
        vector_result = vectorizer.process(
            paths,
            stroke_width_map=skeleton_result.stroke_width_map,
            grayscale_image=preprocess_result.grayscale_image,
            chart_id="benchmark",
        )
        
        # Step 5: Element detection
        element_result = element_detector.detect(
            preprocess_result.binary_image,
            color_image=image,
            chart_id="benchmark",
        )
        
        # Classification
        classification = classifier.classify(
            bars=element_result.bars,
            polylines=vector_result.polylines,
            markers=element_result.markers,
            slices=element_result.slices,
            texts=[],  # Skip OCR for speed
            image_shape=(h, w),
            chart_id="benchmark",
        )
        
        return (
            classification.chart_type.value,
            classification.confidence,
            classification.features,
        )
    
    except Exception as e:
        return "error", 0.0, {"error": str(e)}


def main():
    """Run benchmark."""
    # Paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data/academic_dataset/chart_qa/dataset.json"
    images_dir = project_root / "data/academic_dataset/images"
    
    print("=" * 70)
    print("CHART CLASSIFIER BENCHMARK")
    print("=" * 70)
    
    # Load ground truth
    print("\n[1/4] Loading ground truth...")
    ground_truth = load_ground_truth(dataset_path)
    print(f"      Loaded {len(ground_truth)} labels")
    
    # Initialize modules
    print("\n[2/4] Initializing modules...")
    preprocessor = ImagePreprocessor(PreprocessConfig())
    skeletonizer = Skeletonizer(SkeletonConfig())
    vectorizer = Vectorizer(VectorizeConfig())
    element_detector = ElementDetector(ElementDetectorConfig())
    classifier = ChartClassifier(ClassifierConfig())
    print("      All modules initialized")
    
    # Run benchmark
    print("\n[3/4] Running benchmark...")
    
    # Sample images by type
    type_samples: Dict[str, List[str]] = defaultdict(list)
    for image_id, chart_type in ground_truth.items():
        type_samples[chart_type].append(image_id)
    
    # Sample up to N per type
    max_per_type = 10  # Small sample for initial testing
    sampled_ids = []
    for chart_type, ids in type_samples.items():
        np.random.seed(42)  # Reproducible
        n_sample = min(len(ids), max_per_type)
        sampled = np.random.choice(ids, n_sample, replace=False)
        sampled_ids.extend(sampled)
        print(f"      {chart_type}: sampling {n_sample}/{len(ids)} images")
    
    # Run classification
    results = []
    confusion = defaultdict(lambda: defaultdict(int))
    
    print(f"\n      Processing {len(sampled_ids)} images...")
    
    for i, image_id in enumerate(sampled_ids):
        image_path = images_dir / f"{image_id}.png"
        
        if not image_path.exists():
            continue
        
        predicted, confidence, features = process_image(
            image_path,
            preprocessor,
            skeletonizer,
            vectorizer,
            element_detector,
            classifier,
        )
        
        actual = ground_truth[image_id]
        
        results.append({
            "image_id": image_id,
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence,
            "features": features,
            "correct": actual == predicted,
        })
        
        confusion[actual][predicted] += 1
        
        if (i + 1) % 50 == 0:
            print(f"      Processed {i + 1}/{len(sampled_ids)} images...")
    
    # Analysis
    print("\n[4/4] Analyzing results...")
    
    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        class_stats[r["actual"]]["total"] += 1
        if r["correct"]:
            class_stats[r["actual"]]["correct"] += 1
    
    for chart_type in sorted(class_stats.keys()):
        stats = class_stats[chart_type]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {chart_type:12s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")
    
    # Confusion matrix
    print("\nConfusion Matrix (Actual -> Predicted):")
    print("-" * 70)
    
    all_types = sorted(set(list(confusion.keys()) + 
                          [p for d in confusion.values() for p in d.keys()]))
    
    # Header
    header = "Actual\\Pred"
    for t in all_types:
        header += f" {t[:7]:>7s}"
    print(header)
    
    for actual in all_types:
        row = f"{actual:12s}"
        for predicted in all_types:
            count = confusion[actual][predicted]
            row += f" {count:7d}"
        print(row)
    
    # Most common errors
    print("\nMost Common Errors:")
    print("-" * 50)
    
    errors = [(r["actual"], r["predicted"], r["image_id"]) 
              for r in results if not r["correct"]]
    
    error_patterns = defaultdict(list)
    for actual, predicted, image_id in errors:
        error_patterns[(actual, predicted)].append(image_id)
    
    for (actual, predicted), ids in sorted(
        error_patterns.items(), key=lambda x: -len(x[1])
    )[:10]:
        print(f"  {actual} -> {predicted}: {len(ids)} errors")
        print(f"    Examples: {ids[:3]}")
    
    # Feature analysis for errors
    print("\nFeature Analysis (Average by Actual Type):")
    print("-" * 70)
    
    type_features = defaultdict(lambda: defaultdict(list))
    for r in results:
        for feat, val in r["features"].items():
            if isinstance(val, (int, float)):
                type_features[r["actual"]][feat].append(val)
    
    key_features = ["num_bars", "num_polylines", "num_markers", "num_slices"]
    print(f"{'Type':12s} | " + " | ".join(f"{f:12s}" for f in key_features))
    print("-" * 70)
    
    for chart_type in sorted(type_features.keys()):
        row = f"{chart_type:12s} |"
        for feat in key_features:
            vals = type_features[chart_type][feat]
            avg = np.mean(vals) if vals else 0
            row += f" {avg:12.2f} |"
        print(row)
    
    # Save detailed results
    output_path = project_root / "docs/reports/classifier_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "class_stats": {k: dict(v) for k, v in class_stats.items()},
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "errors": [r for r in results if not r["correct"]],
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
