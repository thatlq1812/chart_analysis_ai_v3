#!/usr/bin/env python3
"""
Prepare SLM Training Data — v3

Improvements over v2:
  - Axis info correctly read from flat keys (x_min/x_max/y_min/y_max)
  - Calibration confidence gating (only include axis if conf > AXIS_CONF_THRESHOLD)
  - OCR texts grouped by role: [TITLE], [LEGEND], [X_TICKS], [Y_TICKS], [DATA_LABELS]
  - Element counts by type: [ELEMENTS]: bar=24 point=0
  - [OCR_QUALITY]: low marker for zero-text charts
  - All 8 chart types including line (0 in v2)
  - Metadata enriched: has_stage3, axis_conf, features_used
  - Consistent single schema (no CAPTION/CONTEXT mix)
  - Dry-run mode: print stats without writing files

Usage:
    .venv/Scripts/python.exe scripts/prepare_slm_training_v3.py
    .venv/Scripts/python.exe scripts/prepare_slm_training_v3.py --dry-run
    .venv/Scripts/python.exe scripts/prepare_slm_training_v3.py --max-per-type 3000
    .venv/Scripts/python.exe scripts/prepare_slm_training_v3.py --curriculum stage3

Output:
    data/slm_training_v3/
        train.json
        val.json
        test.json
        dataset_info.json
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# ── Constants ────────────────────────────────────────────────────────────────
QA_DIR = PROJECT_ROOT / "data" / "academic_dataset" / "chart_qa_v2" / "generated"
FEATURES_DIR = PROJECT_ROOT / "data" / "academic_dataset" / "stage3_features"
OUTPUT_DIR = PROJECT_ROOT / "data" / "slm_training_v3"

# Minimum calibration confidence to include axis range in prompt
AXIS_CONF_THRESHOLD = 0.30

# Chart types used for training (skip non-chart categories)
VALID_CHART_TYPES = {"bar", "line", "scatter", "heatmap", "histogram", "box", "pie", "area"}

# Curriculum stage mapping
STAGE1_TYPES = {"structural", "layout", "element_count"}
STAGE2_TYPES = {"extraction", "range", "threshold", "max", "min", "value"}
STAGE3_TYPES = {"trend", "comparison", "why_reasoning", "interpolation",
                "percentage_change", "multi_hop", "prediction"}

SYSTEM_PROMPTS = {
    1: (
        "You are a chart structure expert. Describe the chart's visual structure "
        "accurately based on the provided metadata. Focus on layout, not values."
    ),
    2: (
        "You are a chart analysis expert. Answer questions about charts accurately "
        "based on the provided metadata. Provide exact numerical values when possible."
    ),
    3: (
        "You are a chart reasoning expert. Analyze trends, make comparisons, and "
        "draw insights from the chart data. Be analytical and precise."
    ),
}


# ── Data Loaders ─────────────────────────────────────────────────────────────

def load_qa_pairs(qa_dir: Path) -> Dict[str, Dict]:
    """
    Load all QA files. Returns {chart_id: {"chart_type": str, "data": dict}}.
    """
    qa_map: Dict[str, Dict] = {}
    for ct_dir in qa_dir.iterdir():
        if not ct_dir.is_dir():
            continue
        if ct_dir.name not in VALID_CHART_TYPES:
            continue
        for qa_file in ct_dir.glob("*.json"):
            try:
                data = json.loads(qa_file.read_text(encoding="utf-8", errors="replace"))
                qa_map[qa_file.stem] = {"chart_type": ct_dir.name, "data": data}
            except Exception as e:
                logger.warning(f"QA load failed | file={qa_file.name} | error={e}")
    logger.info(f"Loaded QA for {len(qa_map)} charts")
    return qa_map


def load_stage3_features(features_dir: Path) -> Dict[str, Dict]:
    """
    Load all Stage3 feature files. Returns {chart_id: feature_dict}.
    """
    features: Dict[str, Dict] = {}
    for ct_dir in features_dir.iterdir():
        if not ct_dir.is_dir():
            continue
        for feat_file in ct_dir.glob("*.json"):
            try:
                features[feat_file.stem] = json.loads(feat_file.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                logger.warning(f"Feature load failed | file={feat_file.name} | error={e}")
    logger.info(f"Loaded Stage3 features for {len(features)} charts")
    return features


# ── Context Formatting ───────────────────────────────────────────────────────

def format_context(
    chart_type: str,
    features: Optional[Dict],
    qa_data: Optional[Dict],
) -> Tuple[str, List[str]]:
    """
    Build the structured context string for the model prompt.

    Returns:
        (context_str, features_used_list)

    Example output:
        [CHART_TYPE]: LINE
        [TITLE]: Training loss comparison
        [LEGEND]: Adam, SGD, RMSProp
        [X_TICKS]: 0, 10, 20, 30, 40, 50
        [Y_TICKS]: 0.1, 0.3, 0.5, 0.7
        [DATA_LABELS]: 0.12, 0.45
        [AXIS_INFO]: x=[0.0, 50.0] conf=0.89 | y=[0.1, 1.0] conf=0.92
        [ELEMENTS]: point=45
    """
    lines: List[str] = []
    features_used: List[str] = []

    lines.append(f"[CHART_TYPE]: {chart_type.upper()}")

    if features:
        features_used.append("stage3")

        # ── Text roles ────────────────────────────────────────────────────
        texts = features.get("texts") or []
        by_role: Dict[str, List[str]] = defaultdict(list)
        for t in texts:
            role = (t.get("role") or "unknown").lower()
            text = (t.get("text") or "").strip()
            if text:
                by_role[role].append(text)

        role_tag_map = {
            "title": "TITLE",
            "subtitle": "TITLE",
            "x_axis_label": "X_LABEL",
            "y_axis_label": "Y_LABEL",
            "legend": "LEGEND",
            "x_tick": "X_TICKS",
            "y_tick": "Y_TICKS",
            "data_label": "DATA_LABELS",
        }

        for role, tag in role_tag_map.items():
            vals = by_role.get(role, [])
            if vals:
                # De-duplicate while preserving order, cap at 12
                seen: set = set()
                unique_vals = [v for v in vals if not (v in seen or seen.add(v))][:12]
                lines.append(f"[{tag}]: {', '.join(unique_vals)}")
                features_used.append(f"text_{role}")

        # Unknown-role texts as fallback OCR (max 10 tokens)
        unknowns = by_role.get("unknown", [])[:10]
        if unknowns and not any(role in by_role for role in role_tag_map):
            lines.append(f"[OCR_TEXT]: {', '.join(unknowns)}")
            features_used.append("ocr_fallback")

        if not texts:
            lines.append("[OCR_QUALITY]: low")

        # ── Axis calibration ──────────────────────────────────────────────
        ai = features.get("axis_info") or {}
        x_min = ai.get("x_min")
        x_max = ai.get("x_max")
        y_min = ai.get("y_min")
        y_max = ai.get("y_max")
        x_conf = ai.get("x_calibration_confidence", 0.0)
        y_conf = ai.get("y_calibration_confidence", 0.0)

        axis_parts: List[str] = []
        if x_min is not None and x_max is not None and x_conf >= AXIS_CONF_THRESHOLD:
            axis_parts.append(f"x=[{x_min}, {x_max}] conf={x_conf:.2f}")
            features_used.append("axis_x")
        if y_min is not None and y_max is not None and y_conf >= AXIS_CONF_THRESHOLD:
            axis_parts.append(f"y=[{y_min}, {y_max}] conf={y_conf:.2f}")
            features_used.append("axis_y")

        if axis_parts:
            lines.append(f"[AXIS_INFO]: {' | '.join(axis_parts)}")

        # ── Elements breakdown ────────────────────────────────────────────
        elements = features.get("elements") or []
        if elements:
            elem_counts: Counter = Counter()
            for e in elements:
                et = e.get("element_type") or "unknown"
                elem_counts[et] += 1
            parts = [f"{k}={v}" for k, v in elem_counts.most_common()]
            lines.append(f"[ELEMENTS]: {', '.join(parts)}")
            features_used.append("elements")

    else:
        # Fallback: use QA caption/context if no Stage3 feature
        if qa_data:
            caption = qa_data.get("caption") or ""
            context_text = qa_data.get("context_text") or ""
            if caption:
                lines.append(f"[CAPTION]: {caption[:200]}")
                features_used.append("caption")
            if context_text:
                lines.append(f"[CONTEXT]: {context_text[:300]}")
                features_used.append("context_text")
        if not features_used:
            lines.append("[OCR_QUALITY]: unavailable")

    return "\n".join(lines), features_used


# ── Curriculum Classification ────────────────────────────────────────────────

def classify_curriculum(question_type: str) -> int:
    """Map question_type string to curriculum stage 1–3."""
    qt = (question_type or "").lower()
    if qt in STAGE1_TYPES:
        return 1
    if qt in STAGE3_TYPES:
        return 3
    return 2  # default: numeric extraction


# ── Conversation Builder ──────────────────────────────────────────────────────

def build_sample(
    question: str,
    answer: str,
    context: str,
    question_type: str,
    chart_type: str,
    chart_id: str,
    difficulty: int,
    source: str,
    features_used: List[str],
) -> Dict:
    """Build a single training sample in ChatML conversations format."""
    stage = classify_curriculum(question_type)
    system = SYSTEM_PROMPTS.get(stage, SYSTEM_PROMPTS[2])

    has_axis = any("axis" in fu for fu in features_used)
    has_stage3 = "stage3" in features_used

    return {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n[QUESTION]: {question}"},
            {"role": "assistant", "content": str(answer)},
        ],
        "metadata": {
            "question_type": question_type,
            "curriculum_stage": stage,
            "chart_type": chart_type,
            "chart_id": chart_id,
            "difficulty": difficulty,
            "source": source,
            "has_stage3": has_stage3,
            "has_axis_info": has_axis,
            "features_used": features_used,
        },
    }


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_dataset(
    qa_map: Dict[str, Dict],
    feat_map: Dict[str, Dict],
    curriculum_filter: Optional[int],
    max_per_type: int,
) -> Tuple[List[Dict], Dict]:
    """
    Process all QA pairs and build the complete sample list.
    Returns (samples_list, stats_dict).
    """
    all_samples: List[Dict] = []
    stats: Dict[str, Any] = defaultdict(lambda: defaultdict(int))

    no_qa_pairs = 0
    no_features = 0

    for chart_id, qa_entry in qa_map.items():
        chart_type = qa_entry["chart_type"]
        qa_data = qa_entry["data"]

        features = feat_map.get(chart_id)
        if features is None:
            no_features += 1

        qa_pairs = qa_data.get("qa_pairs") or []
        if not qa_pairs:
            no_qa_pairs += 1
            continue

        context, features_used = format_context(chart_type, features, qa_data)
        source = qa_data.get("generator_model") or "gemini"

        for qa in qa_pairs:
            question = (qa.get("question") or "").strip()
            answer_raw = qa.get("answer")
            answer = str(answer_raw).strip() if answer_raw is not None else ""
            q_type = qa.get("question_type") or qa.get("type") or "unknown"
            difficulty = int(qa.get("difficulty") or 3)

            if not question or not answer:
                continue

            stage = classify_curriculum(q_type)
            if curriculum_filter is not None and stage != curriculum_filter:
                continue

            sample = build_sample(
                question=question,
                answer=answer,
                context=context,
                question_type=q_type,
                chart_type=chart_type,
                chart_id=chart_id,
                difficulty=difficulty,
                source=source,
                features_used=list(features_used),
            )
            all_samples.append(sample)
            stats["by_type"][chart_type] += 1
            stats["by_stage"][f"stage{stage}"] += 1
            stats["by_qtype"][q_type] += 1
            stats["has_stage3"]["yes" if features else "no"] += 1
            stats["has_axis"]["yes" if "axis_x" in features_used or "axis_y" in features_used else "no"] += 1

    logger.info(f"Built {len(all_samples)} samples total")
    logger.info(f"  Charts with no QA pairs: {no_qa_pairs}")
    logger.info(f"  Charts with no Stage3 features: {no_features}")

    # Per-type cap
    if max_per_type > 0:
        by_type: Dict[str, List] = defaultdict(list)
        for s in all_samples:
            by_type[s["metadata"]["chart_type"]].append(s)
        capped: List[Dict] = []
        for ct, slist in by_type.items():
            random.shuffle(slist)
            capped.extend(slist[:max_per_type])
        all_samples = capped
        logger.info(f"After per-type cap ({max_per_type}): {len(all_samples)} samples")

    return all_samples, dict(stats)


# ── Train/Val/Test Split ──────────────────────────────────────────────────────

def split_by_chart_id(
    samples: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split by chart_id to prevent data leakage (all QA from one chart go to same split).
    """
    rng = random.Random(seed)

    # Group by chart_id
    by_id: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_id[s["metadata"]["chart_id"]].append(s)

    chart_ids = list(by_id.keys())
    rng.shuffle(chart_ids)

    n = len(chart_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_ids = set(chart_ids[:train_end])
    val_ids = set(chart_ids[train_end:val_end])

    train, val, test = [], [], []
    for cid, slist in by_id.items():
        if cid in train_ids:
            train.extend(slist)
        elif cid in val_ids:
            val.extend(slist)
        else:
            test.extend(slist)

    # Shuffle within splits
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_json(samples: List[Dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SLM training dataset v3 from Stage3 features + QA pairs"
    )
    parser.add_argument("--qa-dir", type=Path, default=QA_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--curriculum",
        choices=["all", "stage1", "stage2", "stage3"],
        default="all",
        help="Include only a specific curriculum stage",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=0,
        help="Max samples per chart type (0 = no limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing files",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Validate inputs
    if not args.qa_dir.exists():
        logger.error(f"QA dir not found: {args.qa_dir}")
        sys.exit(1)
    if not args.features_dir.exists():
        logger.error(f"Features dir not found: {args.features_dir}")
        sys.exit(1)

    curriculum_filter = None
    if args.curriculum != "all":
        curriculum_filter = int(args.curriculum[-1])
        logger.info(f"Curriculum filter: stage {curriculum_filter}")

    # Load
    logger.info("Loading QA pairs...")
    qa_map = load_qa_pairs(args.qa_dir)

    logger.info("Loading Stage3 features...")
    feat_map = load_stage3_features(args.features_dir)

    coverage = sum(1 for cid in qa_map if cid in feat_map)
    logger.info(
        f"Stage3 coverage: {coverage}/{len(qa_map)} QA charts "
        f"({100*coverage/max(1,len(qa_map)):.1f}%)"
    )

    # Build
    logger.info("Building samples...")
    all_samples, stats = build_dataset(
        qa_map=qa_map,
        feat_map=feat_map,
        curriculum_filter=curriculum_filter,
        max_per_type=args.max_per_type,
    )

    # Split (by chart_id to prevent leakage)
    train, val, test = split_by_chart_id(all_samples, seed=args.seed)

    # ── Report ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DATASET v3 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples : {len(all_samples)}")
    logger.info(f"  Train       : {len(train)}")
    logger.info(f"  Val         : {len(val)}")
    logger.info(f"  Test        : {len(test)}")
    logger.info("")
    logger.info("Chart type distribution:")
    for ct, count in sorted(stats.get("by_type", {}).items(), key=lambda x: -x[1]):
        logger.info(f"  {ct:<12} {count:>6}")
    logger.info("")
    logger.info("Curriculum stage distribution:")
    for s, count in sorted(stats.get("by_stage", {}).items()):
        logger.info(f"  {s}: {count}")
    logger.info("")
    logger.info("Has axis info:")
    for k, v in stats.get("has_axis", {}).items():
        logger.info(f"  {k}: {v} ({100*v/max(1,len(all_samples)):.1f}%)")
    logger.info("")
    logger.info("Has Stage3 features:")
    for k, v in stats.get("has_stage3", {}).items():
        logger.info(f"  {k}: {v} ({100*v/max(1,len(all_samples)):.1f}%)")

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — no files written.")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nWriting to {args.output_dir} ...")
    save_json(train, args.output_dir / "train.json")
    save_json(val,   args.output_dir / "val.json")
    save_json(test,  args.output_dir / "test.json")

    info = {
        "created_at": datetime.now().isoformat(),
        "version": "v3",
        "qa_dir": str(args.qa_dir),
        "features_dir": str(args.features_dir),
        "curriculum_filter": args.curriculum,
        "max_per_type": args.max_per_type,
        "seed": args.seed,
        "axis_conf_threshold": AXIS_CONF_THRESHOLD,
        "total_samples": len(all_samples),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "stage3_coverage": f"{coverage}/{len(qa_map)} ({100*coverage/max(1,len(qa_map)):.1f}%)",
        "stats": {k: dict(v) for k, v in stats.items()},
    }
    save_json(info, args.output_dir / "dataset_info.json")

    logger.info(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
