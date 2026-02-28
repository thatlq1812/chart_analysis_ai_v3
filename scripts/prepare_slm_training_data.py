#!/usr/bin/env python3
"""
Prepare SLM Training Data

Merge QA pairs with Stage 3 features to create training dataset for Qwen-2.5-1.5B.
Follows curriculum learning approach from chatlog3.md:
- Stage 1: Structure grounding
- Stage 2: Numeric grounding  
- Stage 3: Reasoning & trend
- Stage 4: Robustness

Usage:
    .venv/Scripts/python.exe scripts/prepare_slm_training_data.py
    .venv/Scripts/python.exe scripts/prepare_slm_training_data.py --output-format alpaca
    .venv/Scripts/python.exe scripts/prepare_slm_training_data.py --curriculum stage2

Output:
    data/slm_training/
        train.jsonl
        val.jsonl
        test.jsonl
        dataset_info.json
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_qa_pairs(qa_dir: Path) -> Dict[str, List[Dict]]:
    """Load all QA pairs grouped by chart_id."""
    qa_data = defaultdict(list)
    
    for chart_type_dir in qa_dir.iterdir():
        if not chart_type_dir.is_dir():
            continue
        if chart_type_dir.name in ["not_a_chart", "other", "diagram", "table"]:
            continue
            
        for qa_file in chart_type_dir.glob("*.json"):
            try:
                with open(qa_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                chart_id = qa_file.stem
                qa_data[chart_id].append({
                    "chart_type": chart_type_dir.name,
                    "qa_file": str(qa_file),
                    "data": data
                })
            except Exception as e:
                logger.warning(f"Failed to load {qa_file}: {e}")
    
    return dict(qa_data)


def load_stage3_features(features_dir: Path) -> Dict[str, Dict]:
    """Load all Stage 3 features grouped by chart_id."""
    features = {}
    
    for chart_type_dir in features_dir.iterdir():
        if not chart_type_dir.is_dir():
            continue
            
        for feature_file in chart_type_dir.glob("*.json"):
            try:
                with open(feature_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                chart_id = feature_file.stem
                features[chart_id] = data
            except Exception as e:
                logger.warning(f"Failed to load {feature_file}: {e}")
    
    return features


def load_ocr_cache(cache_file: Path) -> Dict[str, Dict]:
    """Load OCR cache."""
    logger.info(f"Loading OCR cache from {cache_file}...")
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data.get('results', {}))} OCR entries")
    return data.get("results", {})


def format_chart_context(
    features: Optional[Dict],
    ocr_data: Optional[Dict],
    chart_type: str,
    format_style: str = "structured"
) -> str:
    """
    Format chart context for model input.
    
    Following chatlog3.md recommendation:
    Use semi-structured text, not raw OCR.
    """
    lines = []
    
    # Chart type
    lines.append(f"[CHART_TYPE]: {chart_type.upper()}")
    
    # From Stage 3 features
    if features:
        # Axis info
        axis_info = features.get("axis_info", {})
        if axis_info:
            y_range = axis_info.get("y_range", {})
            x_range = axis_info.get("x_range", {})
            if y_range:
                lines.append(f"[AXIS_Y]: min={y_range.get('min', 'N/A')} max={y_range.get('max', 'N/A')}")
            if x_range:
                lines.append(f"[AXIS_X]: min={x_range.get('min', 'N/A')} max={x_range.get('max', 'N/A')}")
        
        # Elements summary
        elements = features.get("elements", [])
        if elements:
            element_counts = defaultdict(int)
            for e in elements:
                element_counts[e.get("type", "unknown")] += 1
            elem_str = ", ".join(f"{k}={v}" for k, v in element_counts.items())
            lines.append(f"[ELEMENTS]: {elem_str}")
        
        # Text labels
        texts = features.get("texts", [])
        if texts:
            # Group by role
            by_role = defaultdict(list)
            for t in texts:
                role = t.get("role", "unknown") or "unknown"
                by_role[role].append(t.get("text", ""))
            
            for role, values in by_role.items():
                if values and role != "unknown":
                    lines.append(f"[{role.upper()}]: {', '.join(values[:10])}")
    
    # From OCR cache (fallback)
    elif ocr_data:
        texts = ocr_data.get("texts", [])
        if texts:
            text_values = [t.get("text", "") for t in texts[:20]]
            lines.append(f"[OCR_TEXT]: {', '.join(text_values)}")
    
    return "\n".join(lines)


def create_conversation(
    question: str,
    answer: str,
    context: str,
    question_type: str = "unknown",
    curriculum_stage: int = 2,
) -> Dict:
    """Create a conversation in chat format."""
    
    # System prompt based on curriculum stage
    if curriculum_stage == 1:
        system = (
            "You are a chart structure expert. Describe the chart's visual structure "
            "accurately based on the provided metadata. Focus on layout, not values."
        )
    elif curriculum_stage == 2:
        system = (
            "You are a chart analysis expert. Answer questions about charts accurately "
            "based on the provided metadata. Provide exact numerical values when possible."
        )
    elif curriculum_stage == 3:
        system = (
            "You are a chart reasoning expert. Analyze trends, make comparisons, and "
            "draw insights from the chart data. Be analytical and precise."
        )
    else:
        system = (
            "You are a chart analysis expert. Answer questions about charts accurately "
            "based on the provided metadata. Be concise and specific."
        )
    
    return {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n[QUESTION]: {question}"},
            {"role": "assistant", "content": answer}
        ],
        "metadata": {
            "question_type": question_type,
            "curriculum_stage": curriculum_stage,
        }
    }


def classify_question_curriculum(question_type: str) -> int:
    """Map question type to curriculum stage."""
    stage1_types = {"structural", "layout", "element_count"}
    stage2_types = {"extraction", "range", "threshold", "max", "min", "value"}
    stage3_types = {"trend", "comparison", "why_reasoning", "interpolation", 
                    "percentage_change", "multi_hop", "prediction"}
    
    if question_type in stage1_types:
        return 1
    elif question_type in stage2_types:
        return 2
    elif question_type in stage3_types:
        return 3
    return 2  # Default


def process_qa_pair(
    qa_entry: Dict,
    features: Optional[Dict],
    ocr_data: Optional[Dict],
    chart_type: str,
) -> List[Dict]:
    """Process a single QA entry into training samples."""
    samples = []

    data = qa_entry.get("data", {})
    qa_pairs = data.get("qa_pairs", [])

    # Format context — pass QA record caption/context when stage3 features absent
    caption = data.get("caption") or ""
    context_text = data.get("context_text") or ""
    context = format_chart_context(features, ocr_data, chart_type)
    if caption:
        context += f"\n[CAPTION]: {caption}"
    if context_text:
        context += f"\n[CONTEXT]: {context_text}"
    
    for qa in qa_pairs:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        q_type = qa.get("question_type", qa.get("type", "unknown"))
        
        if not question or not answer:
            continue
        
        # Determine curriculum stage
        stage = classify_question_curriculum(q_type)
        
        # Create conversation
        conv = create_conversation(
            question=question,
            answer=str(answer),
            context=context,
            question_type=q_type,
            curriculum_stage=stage,
        )
        conv["metadata"]["chart_type"] = chart_type
        conv["metadata"]["image_id"] = data.get("image_id", "")
        conv["metadata"]["difficulty"] = qa.get("difficulty", 3)
        conv["metadata"]["source"] = data.get("generator_model", "gemini")
        
        samples.append(conv)
    
    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/val/test."""
    random.seed(seed)
    random.shuffle(samples)
    
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def save_jsonl(samples: List[Dict], output_path: Path):
    """Save samples as JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_json(samples: List[Dict], output_path: Path):
    """Save samples as JSON array (compatible with train_slm_lora.py)."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare SLM training data")
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "chart_qa_v2" / "generated",
        help="Directory with QA pairs",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "academic_dataset" / "stage3_features",
        help="Directory with Stage 3 features",
    )
    parser.add_argument(
        "--ocr-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "ocr_cache.json",
        help="OCR cache file (fallback if no features)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "slm_training",
        help="Output directory",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help="Filter by curriculum stage",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per chart type (0 = no limit)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance samples across chart types",
    )
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading QA pairs...")
    qa_data = load_qa_pairs(args.qa_dir)
    logger.info(f"Loaded QA for {len(qa_data)} charts")
    
    logger.info("Loading Stage 3 features...")
    features_data = load_stage3_features(args.features_dir)
    logger.info(f"Loaded features for {len(features_data)} charts")
    
    # Load OCR cache
    ocr_cache = {}
    if args.ocr_cache.exists():
        ocr_cache = load_ocr_cache(args.ocr_cache)
    
    # Process all QA pairs
    all_samples = []
    stats = defaultdict(lambda: defaultdict(int))
    
    for chart_id, qa_entries in qa_data.items():
        for qa_entry in qa_entries:
            chart_type = qa_entry["chart_type"]
            
            # Get features
            features = features_data.get(chart_id)
            
            # Get OCR from cache
            ocr_key = f"{chart_type}\\{chart_id}.png"
            ocr_data = ocr_cache.get(ocr_key) or ocr_cache.get(ocr_key.replace("\\", "/"))
            
            # Process
            samples = process_qa_pair(qa_entry, features, ocr_data, chart_type)
            
            for sample in samples:
                stage = sample["metadata"]["curriculum_stage"]
                q_type = sample["metadata"]["question_type"]
                
                # Filter by curriculum if specified
                if args.curriculum != "all":
                    target_stage = int(args.curriculum[-1])
                    if stage != target_stage:
                        continue
                
                all_samples.append(sample)
                stats[chart_type]["total"] += 1
                stats[chart_type][f"stage{stage}"] += 1
                stats["_global"][q_type] += 1
    
    logger.info(f"Total samples before filtering: {len(all_samples)}")
    
    # Balance if requested
    if args.balance:
        type_counts = defaultdict(list)
        for sample in all_samples:
            ct = sample["metadata"]["chart_type"]
            type_counts[ct].append(sample)
        
        min_count = min(len(v) for v in type_counts.values())
        if args.max_samples > 0:
            min_count = min(min_count, args.max_samples)
        
        balanced = []
        for ct, samples in type_counts.items():
            random.shuffle(samples)
            balanced.extend(samples[:min_count])
        
        all_samples = balanced
        logger.info(f"Balanced to {len(all_samples)} samples ({min_count} per type)")
    
    elif args.max_samples > 0:
        # Limit per chart type
        type_counts = defaultdict(list)
        for sample in all_samples:
            ct = sample["metadata"]["chart_type"]
            type_counts[ct].append(sample)
        
        limited = []
        for ct, samples in type_counts.items():
            random.shuffle(samples)
            limited.extend(samples[:args.max_samples])
        
        all_samples = limited
        logger.info(f"Limited to {len(all_samples)} samples")
    
    # Split
    train, val, test = split_dataset(all_samples)
    
    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(train, args.output_dir / "train.jsonl")
    save_jsonl(val, args.output_dir / "val.jsonl")
    save_jsonl(test, args.output_dir / "test.jsonl")
    # Also save as .json arrays for train_slm_lora.py
    save_json(train, args.output_dir / "train.json")
    save_json(val, args.output_dir / "val.json")
    save_json(test, args.output_dir / "test.json")
    
    # Save dataset info
    info = {
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "qa_dir": str(args.qa_dir),
        "features_dir": str(args.features_dir),
        "total_samples": len(all_samples),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "curriculum_filter": args.curriculum,
        "chart_type_stats": {k: dict(v) for k, v in stats.items() if k != "_global"},
        "question_type_stats": dict(stats["_global"]),
        "features_coverage": f"{len(features_data)}/{len(qa_data)} ({100*len(features_data)/max(1,len(qa_data)):.1f}%)",
    }
    
    with open(args.output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("DATASET PREPARED")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Train: {len(train)}")
    logger.info(f"Val: {len(val)}")
    logger.info(f"Test: {len(test)}")
    logger.info(f"Output: {args.output_dir}")
    
    logger.info("\nChart type distribution:")
    for ct, ct_stats in sorted(stats.items()):
        if ct != "_global":
            logger.info(f"  {ct}: {ct_stats['total']}")
    
    logger.info("\nQuestion type distribution:")
    for qt, count in sorted(stats["_global"].items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
