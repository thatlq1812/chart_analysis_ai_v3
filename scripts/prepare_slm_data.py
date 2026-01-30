#!/usr/bin/env python3
"""
Prepare SLM training data by merging Stage 3 features with QA pairs.

This script:
1. Loads Stage 3 extracted features (OCR, elements, chart type)
2. Loads QA pairs from chart_qa_v2/generated/
3. Merges them into ChatML format for Qwen fine-tuning

Usage:
    python scripts/prepare_slm_data.py
    python scripts/prepare_slm_data.py --limit 1000  # Test run

Output:
    data/slm_training/
        train.json
        val.json
        test.json
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# System prompt for Chart QA
SYSTEM_PROMPT = """You are a chart analysis expert. Answer questions about charts accurately based on the provided metadata. Be concise and specific.

For numerical questions, provide exact values when possible.
For descriptive questions, be brief but informative.
When uncertain, indicate your confidence level."""


def load_qa_pairs(qa_dir: Path) -> Dict[str, dict]:
    """
    Load QA pairs from individual JSON files.
    
    Returns:
        Dict mapping image_id to QA data
    """
    qa_data = {}
    chart_types = ["area", "bar", "box", "heatmap", "histogram", "line", "pie", "scatter"]
    
    for chart_type in chart_types:
        type_dir = qa_dir / chart_type
        if not type_dir.exists():
            continue
        
        for qa_file in type_dir.glob("*.json"):
            try:
                with open(qa_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Use image_id as key
                image_id = data.get("image_id", qa_file.stem)
                qa_data[image_id] = data
                
            except Exception as e:
                logger.warning(f"Failed to load {qa_file}: {e}")
    
    return qa_data


def load_stage3_features(features_dir: Path) -> Dict[str, dict]:
    """
    Load Stage 3 extracted features.
    
    Returns:
        Dict mapping image_id to feature data
    """
    features = {}
    
    for type_dir in features_dir.iterdir():
        if not type_dir.is_dir():
            continue
        
        for feature_file in type_dir.glob("*.json"):
            try:
                with open(feature_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if "error" in data:
                    continue
                
                image_id = data.get("image_id", feature_file.stem)
                features[image_id] = data
                
            except Exception as e:
                logger.warning(f"Failed to load {feature_file}: {e}")
    
    return features


def format_chart_context(features: dict) -> str:
    """
    Format Stage 3 features into a context string for the prompt.
    """
    lines = [
        f"Chart Type: {features['chart_type']}",
    ]
    
    # Add extracted texts
    texts = features.get("texts", [])
    if texts:
        # Group by role
        titles = [t["text"] for t in texts if t.get("role") == "title"]
        labels = [t["text"] for t in texts if t.get("role") in ["x_label", "y_label", "axis_label"]]
        data_labels = [t["text"] for t in texts if t.get("role") == "data_label"]
        other_texts = [t["text"] for t in texts if t.get("role") in ["unknown", None, "legend"]]
        
        if titles:
            lines.append(f"Title: {titles[0]}")
        if labels:
            lines.append(f"Axis Labels: {', '.join(labels[:5])}")
        if data_labels:
            lines.append(f"Data Labels: {', '.join(data_labels[:10])}")
        if other_texts:
            lines.append(f"Other Text: {', '.join(other_texts[:10])}")
    
    # Add element info
    elements = features.get("elements", [])
    if elements:
        element_types = Counter(e["type"] for e in elements)
        lines.append(f"Elements: {dict(element_types)}")
    
    # Add axis info
    axis_info = features.get("axis_info")
    if axis_info and axis_info.get("y_axis_detected"):
        y_range = axis_info.get("y_range", [None, None])
        if y_range[0] is not None:
            lines.append(f"Y-Axis Range: {y_range[0]} to {y_range[1]}")
    
    return "\n".join(lines)


def create_conversation(features: dict, qa_pair: dict) -> dict:
    """
    Create a single conversation in ChatML format.
    """
    context = format_chart_context(features)
    question = qa_pair["question"]
    answer = qa_pair["answer"]
    
    # Format user message
    user_message = f"{context}\n\nQuestion: {question}"
    
    # Format assistant response
    # Add confidence indicator based on inference data
    confidence = qa_pair.get("inference", {}).get("confidence_level", "high")
    assistant_message = f"{answer}"
    
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "metadata": {
            "image_id": features.get("image_id"),
            "chart_type": features.get("chart_type"),
            "question_type": qa_pair.get("question_type"),
            "difficulty": qa_pair.get("difficulty"),
        }
    }


def prepare_dataset(
    qa_data: Dict[str, dict],
    features: Dict[str, dict],
    limit: Optional[int] = None,
    use_qa_context: bool = False,
) -> List[dict]:
    """
    Prepare full dataset by merging QA pairs with Stage 3 features.
    
    Args:
        qa_data: Dict of QA pairs per image
        features: Dict of Stage 3 features per image
        limit: Max conversations to generate
        use_qa_context: If True, use QA inference context instead of Stage 3 features
    """
    dataset = []
    
    # Find matching image_ids
    matched_ids = set(qa_data.keys()) & set(features.keys())
    logger.info(f"Matched image IDs: {len(matched_ids)}")
    logger.info(f"Using QA context mode: {use_qa_context}")
    
    # Process all QA data
    for image_id, qa in qa_data.items():
        # Get Stage 3 features if available
        feat = features.get(image_id)
        
        # Create features from QA inference context if no Stage 3 features
        if feat is None or use_qa_context:
            # Use inference context from QA generation
            inference = qa.get("qa_pairs", [{}])[0].get("inference", {})
            feat = {
                "image_id": image_id,
                "chart_type": qa.get("chart_type", "unknown"),
                "texts": _extract_texts_from_inference(inference),
                "elements": [],
                "axis_info": _extract_axis_from_inference(inference),
            }
        
        # Create conversations for each QA pair
        for qa_pair in qa.get("qa_pairs", []):
            if not qa_pair.get("is_answerable", True):
                continue
            
            conv = create_conversation(feat, qa_pair)
            dataset.append(conv)
        
        if limit and len(dataset) >= limit:
            break
    
    return dataset


def _extract_texts_from_inference(inference: dict) -> List[dict]:
    """Extract text elements from QA inference context."""
    texts = []
    
    # Title
    if inference.get("title"):
        texts.append({"text": inference["title"], "role": "title", "confidence": 0.9})
    
    # X-axis label
    if inference.get("x_axis_label"):
        texts.append({"text": inference["x_axis_label"], "role": "x_label", "confidence": 0.9})
    
    # Y-axis label
    if inference.get("y_axis_label"):
        texts.append({"text": inference["y_axis_label"], "role": "y_label", "confidence": 0.9})
    
    # Data labels from x_categories
    for cat in inference.get("x_categories", []):
        texts.append({"text": str(cat), "role": "data_label", "confidence": 0.9})
    
    # Legend items
    for item in inference.get("legend_items", []):
        texts.append({"text": item, "role": "legend", "confidence": 0.9})
    
    return texts


def _extract_axis_from_inference(inference: dict) -> Optional[dict]:
    """Extract axis info from QA inference context."""
    y_range = inference.get("y_range", [None, None])
    if y_range and y_range[0] is not None:
        return {
            "y_axis_detected": True,
            "y_range": y_range,
        }
    return None


def split_dataset(
    dataset: List[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train/val/test.
    """
    random.seed(seed)
    random.shuffle(dataset)
    
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare SLM training data")
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path("data/academic_dataset/chart_qa_v2/generated"),
        help="Directory with QA pairs",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/academic_dataset/stage3_features"),
        help="Directory with Stage 3 features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/slm_training"),
        help="Output directory for training data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total conversations (for testing)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--use-qa-context",
        action="store_true",
        help="Use QA inference context instead of Stage 3 features",
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading QA pairs...")
    qa_data = load_qa_pairs(args.qa_dir)
    logger.info(f"Loaded {len(qa_data)} charts with QA pairs")
    
    logger.info("Loading Stage 3 features...")
    features = load_stage3_features(args.features_dir)
    logger.info(f"Loaded {len(features)} charts with features")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(
        qa_data, features, 
        limit=args.limit,
        use_qa_context=args.use_qa_context,
    )
    logger.info(f"Total conversations: {len(dataset)}")
    
    if not dataset:
        logger.error("No data to process!")
        return
    
    # Split
    train, val, test = split_dataset(dataset, train_ratio=args.train_ratio)
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Save
    for name, data in [("train", train), ("val", val), ("test", test)]:
        output_path = args.output_dir / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {output_path}")
    
    # Save summary
    summary = {
        "created_at": datetime.now().isoformat(),
        "qa_dir": str(args.qa_dir),
        "features_dir": str(args.features_dir),
        "total_conversations": len(dataset),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "chart_types": dict(Counter(d["metadata"]["chart_type"] for d in dataset)),
        "question_types": dict(Counter(d["metadata"]["question_type"] for d in dataset)),
    }
    
    summary_path = args.output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
