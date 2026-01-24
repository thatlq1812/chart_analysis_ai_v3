#!/usr/bin/env python3
"""
Train ML-based Chart Classifier.

Extracts features from labeled images and trains a Random Forest classifier.

Usage:
    python scripts/train_classifier.py

Author: That Le
Date: 2025-01-21
"""

import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core_engine.stages.s3_extraction.simple_classifier import (
    SimpleChartClassifier,
    SimpleClassifierConfig,
)

logging.basicConfig(level=logging.WARNING)


# Target classes for this project
TARGET_CLASSES = {"bar", "line", "pie"}


def map_label(original_label: str) -> str:
    """Map original label to target classes (bar, line, pie, other)."""
    if original_label in TARGET_CLASSES:
        return original_label
    return "other"


def load_dataset(dataset_path: Path) -> Dict[str, str]:
    """Load dataset labels and map to target classes."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    labels = {}
    for sample in data["samples"]:
        image_id = sample["image_id"]
        chart_type = sample["chart_type"]
        # Map to target classes
        labels[image_id] = map_label(chart_type)
    
    return labels


def extract_features(image: np.ndarray, classifier: SimpleChartClassifier) -> List[float]:
    """Extract feature vector from image."""
    result = classifier.classify(image, chart_id="feature_extraction")
    
    # Feature vector
    features = [
        result.features["h_edge_ratio"],
        result.features["v_edge_ratio"],
        result.features["d_edge_ratio"],
        result.features["n_colors"],
        result.features["color_coverage"],
        result.features["circularity"],
        result.features["n_circles"],
        result.features["grid_score"],
        result.features["n_markers"],
        result.features["n_rectangles"],
        result.features["rect_coverage"],
    ]
    
    return features


def main():
    """Train classifier."""
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data/academic_dataset/chart_qa/dataset.json"
    images_dir = project_root / "data/academic_dataset/images"
    output_dir = project_root / "models/weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CHART CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    labels = load_dataset(dataset_path)
    print(f"      Total images: {len(labels)}")
    
    # Count per type
    type_counts = Counter(labels.values())
    for chart_type, count in sorted(type_counts.items()):
        print(f"      {chart_type}: {count}")
    
    # Initialize feature extractor
    print("\n[2/5] Initializing feature extractor...")
    classifier = SimpleChartClassifier(SimpleClassifierConfig())
    
    # Extract features (sample for speed)
    print("\n[3/5] Extracting features...")
    
    # Sample from each type
    max_per_type = 100  # Max images per type
    sampled = []
    for chart_type, ids in type_counts.items():
        type_ids = [img_id for img_id, t in labels.items() if t == chart_type]
        np.random.seed(42)
        n_sample = min(len(type_ids), max_per_type)
        sampled_ids = np.random.choice(type_ids, n_sample, replace=False)
        sampled.extend([(img_id, chart_type) for img_id in sampled_ids])
    
    print(f"      Sampling {len(sampled)} images...")
    
    X = []
    y = []
    errors = 0
    
    for i, (image_id, chart_type) in enumerate(sampled):
        image_path = images_dir / f"{image_id}.png"
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                errors += 1
                continue
            
            # Resize
            h, w = image.shape[:2]
            max_dim = 600
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
            
            features = extract_features(image, classifier)
            X.append(features)
            y.append(chart_type)
        
        except Exception as e:
            errors += 1
            continue
        
        if (i + 1) % 100 == 0:
            print(f"      Processed {i + 1}/{len(sampled)} images (errors: {errors})")
    
    print(f"      Extracted features from {len(X)} images (errors: {errors})")
    
    # Convert to numpy
    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    print("\n[4/5] Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"      Train set: {len(X_train)}")
    print(f"      Test set: {len(X_test)}")
    
    # Train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate
    print("\n[5/5] Evaluating...")
    y_pred = rf_classifier.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    ))
    
    # Feature importance
    feature_names = [
        "h_edge_ratio", "v_edge_ratio", "d_edge_ratio",
        "n_colors", "color_coverage", "circularity",
        "n_circles", "grid_score", "n_markers",
        "n_rectangles", "rect_coverage"
    ]
    
    print("\nFeature Importance:")
    print("-" * 40)
    importances = rf_classifier.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name:20}: {imp:.4f}")
    
    # Save model
    model_path = output_dir / "chart_classifier_rf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": rf_classifier,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
        }, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Calculate overall accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\nOverall Test Accuracy: {accuracy:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
