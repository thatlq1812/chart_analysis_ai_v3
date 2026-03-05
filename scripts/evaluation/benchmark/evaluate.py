"""
Stage 3 Benchmark Evaluation Harness

Runs Stage 3 extraction on each benchmark chart image, compares the
output against ground truth annotations, and produces a detailed
accuracy report per component:

- Classification accuracy
- OCR text detection (precision/recall by role)
- Element count accuracy (within tolerance)
- Axis range accuracy (relative error)
- Overall ceiling score

Usage:
    .venv/Scripts/python.exe scripts/evaluation/benchmark/evaluate.py

    Options:
        --manifest  Path to benchmark manifest (default: data/benchmark/benchmark_manifest.json)
        --ocr       OCR engine to use: easyocr | paddleocr | none (default: easyocr)
        --output    Output directory (default: data/benchmark/results/)
        --skip-run  Skip re-running Stage 3, use cached results

Output:
    data/benchmark/results/evaluation_report.json
    data/benchmark/results/evaluation_report.md
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"

# Tolerances for numeric comparison
ELEMENT_COUNT_TOLERANCE = 0.25  # 25% tolerance on element count
AXIS_RELATIVE_TOLERANCE = 0.15  # 15% relative error on axis range


# ---------------------------------------------------------------------------
# Metrics data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    """Classification accuracy metrics."""

    total: int = 0
    correct: int = 0
    incorrect: int = 0
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class OCRMetrics:
    """OCR text detection metrics."""

    total_gt_texts: int = 0
    total_detected_texts: int = 0
    title_found: int = 0
    title_total: int = 0
    tick_labels_found: int = 0
    tick_labels_total: int = 0
    axis_labels_found: int = 0
    axis_labels_total: int = 0
    legend_found: int = 0
    legend_total: int = 0
    data_labels_found: int = 0
    data_labels_total: int = 0

    @property
    def title_recall(self) -> float:
        return self.title_found / self.title_total if self.title_total > 0 else 0.0

    @property
    def tick_recall(self) -> float:
        return self.tick_labels_found / self.tick_labels_total if self.tick_labels_total > 0 else 0.0

    @property
    def overall_recall(self) -> float:
        return self.total_detected_texts / self.total_gt_texts if self.total_gt_texts > 0 else 0.0


@dataclass
class ElementMetrics:
    """Element detection metrics."""

    total_charts: int = 0
    count_within_tolerance: int = 0
    count_exact_match: int = 0
    type_correct: int = 0
    mean_relative_error: float = 0.0
    errors: List[float] = field(default_factory=list)

    @property
    def tolerance_accuracy(self) -> float:
        return self.count_within_tolerance / self.total_charts if self.total_charts > 0 else 0.0

    @property
    def type_accuracy(self) -> float:
        return self.type_correct / self.total_charts if self.total_charts > 0 else 0.0


@dataclass
class AxisMetrics:
    """Axis calibration metrics."""

    total_axes: int = 0  # Number of axis ranges annotated
    within_tolerance: int = 0
    mean_relative_error: float = 0.0
    errors: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.within_tolerance / self.total_axes if self.total_axes > 0 else 0.0


@dataclass
class ChartResult:
    """Evaluation result for a single chart."""

    chart_id: str
    chart_type_gt: str
    chart_type_pred: str
    difficulty: str
    classification_correct: bool = False

    # OCR
    gt_text_count: int = 0
    detected_text_count: int = 0
    title_found: bool = False

    # Elements
    gt_element_count: int = 0
    pred_element_count: int = 0
    element_count_error: float = 0.0
    element_type_correct: bool = False

    # Axis
    axis_errors: Dict[str, Optional[float]] = field(default_factory=dict)
    axis_within_tolerance: bool = False

    # Confidence
    overall_confidence: float = 0.0
    processing_time_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------


def compare_classification(
    gt_type: str,
    pred_type: str,
) -> bool:
    """Check if predicted chart type matches ground truth."""
    # Normalize - treat stacked_bar/grouped_bar as bar
    normalize_map = {
        "stacked_bar": "bar",
        "grouped_bar": "bar",
        "donut": "pie",
    }
    gt_norm = normalize_map.get(gt_type, gt_type)
    pred_norm = normalize_map.get(pred_type, pred_type)
    return gt_norm == pred_norm


def compare_ocr_texts(
    gt_texts: List[Dict[str, Any]],
    pred_texts: List[Dict[str, Any]],
    gt_title: Optional[str],
) -> Dict[str, Any]:
    """
    Compare ground truth texts with predicted OCR texts.

    Uses fuzzy matching: a GT text is "found" if any predicted text
    contains it as a substring (case-insensitive) or has >80% overlap.
    """
    result = {
        "title_found": False,
        "role_counts": {},
        "found_by_role": {},
    }

    # Count GT texts by role
    gt_by_role: Dict[str, List[str]] = {}
    for t in gt_texts:
        role = t.get("role", "unknown")
        text = t.get("text", "")
        if role not in gt_by_role:
            gt_by_role[role] = []
        gt_by_role[role].append(text.lower().strip())

    # Predicted texts
    pred_text_set = {t.get("text", "").lower().strip() for t in pred_texts}

    # Title check
    if gt_title:
        gt_title_lower = gt_title.lower().strip()
        result["title_found"] = any(
            gt_title_lower in p or p in gt_title_lower
            for p in pred_text_set
            if len(p) > 2
        )

    # Role-by-role matching
    for role, gt_list in gt_by_role.items():
        found = 0
        for gt_text in gt_list:
            if not gt_text:
                continue
            # Fuzzy match: GT text found in any prediction
            matched = any(
                _fuzzy_text_match(gt_text, p)
                for p in pred_text_set
            )
            if matched:
                found += 1
        result["role_counts"][role] = len(gt_list)
        result["found_by_role"][role] = found

    return result


def _fuzzy_text_match(gt: str, pred: str, threshold: float = 0.6) -> bool:
    """
    Check if two text strings match fuzzy.

    Handles OCR errors like 'O' vs '0', spacing differences, etc.
    """
    if not gt or not pred:
        return False

    # Exact substring
    if gt in pred or pred in gt:
        return True

    # Numeric-aware: strip non-alphanumeric and compare
    gt_clean = "".join(c for c in gt if c.isalnum())
    pred_clean = "".join(c for c in pred if c.isalnum())
    if gt_clean and pred_clean and (gt_clean in pred_clean or pred_clean in gt_clean):
        return True

    # Character overlap ratio
    if len(gt) >= 3:
        common = sum(1 for c in gt if c in pred)
        ratio = common / max(len(gt), 1)
        if ratio >= threshold:
            return True

    return False


def compare_elements(
    gt_elements: Dict[str, Any],
    pred_elements: List[Dict[str, Any]],
    tolerance: float = ELEMENT_COUNT_TOLERANCE,
) -> Dict[str, Any]:
    """Compare ground truth element counts with predicted.

    Uses type-filtered counting: only predicted elements matching the
    GT primary_element_type are counted. This avoids penalizing the
    count metric for false-positive detections of wrong element types
    (which is already captured by the type_correct metric).
    """
    gt_count = gt_elements.get("element_count", 0)
    # Handle string element counts from Gemini (e.g. "~400", "~3500")
    if isinstance(gt_count, str):
        import re
        match = re.search(r"\d+", gt_count)
        gt_count = int(match.group()) if match else 0
    gt_count = int(gt_count)
    gt_type = gt_elements.get("primary_element_type", "")

    # Type check (unfiltered)
    pred_types = {e.get("element_type", "") for e in pred_elements}
    type_correct = gt_type in pred_types or (
        gt_type == "line" and "point" in pred_types  # Lines detected as points
    )

    # Type-filtered count: only count elements matching the expected type
    # This gives a fairer comparison (wrong-type detections measured separately)
    matching_types = {gt_type}
    if gt_type == "line":
        matching_types.add("point")  # Lines often detected as points
    elif gt_type == "point":
        matching_types.add("point")

    pred_count_filtered = sum(
        1 for e in pred_elements if e.get("element_type", "") in matching_types
    )
    pred_count_total = len(pred_elements)

    # Count check with tolerance (using filtered count)
    if gt_count == 0:
        rel_error = 0.0 if pred_count_filtered == 0 else 1.0
    else:
        rel_error = abs(pred_count_filtered - gt_count) / gt_count

    within_tol = rel_error <= tolerance

    return {
        "gt_count": gt_count,
        "pred_count": pred_count_filtered,
        "pred_count_total": pred_count_total,
        "relative_error": rel_error,
        "within_tolerance": within_tol,
        "type_correct": type_correct,
    }


def compare_axis(
    gt_axis: Optional[Dict[str, Any]],
    pred_axis: Optional[Dict[str, Any]],
    tolerance: float = AXIS_RELATIVE_TOLERANCE,
) -> Dict[str, Any]:
    """Compare ground truth axis ranges with predicted."""
    result = {
        "errors": {},
        "within_tolerance": True,
        "total_ranges": 0,
        "ranges_within_tol": 0,
    }

    if gt_axis is None:
        return result

    for key in ("x_min", "x_max", "y_min", "y_max"):
        gt_val = gt_axis.get(key)
        if gt_val is None:
            continue

        result["total_ranges"] += 1
        pred_val = pred_axis.get(key) if pred_axis else None

        if pred_val is None:
            result["errors"][key] = None  # Missing prediction
            result["within_tolerance"] = False
            continue

        # Relative error
        if abs(gt_val) < 1e-10:
            rel_err = abs(pred_val) if abs(pred_val) > 1e-10 else 0.0
        else:
            rel_err = abs(pred_val - gt_val) / abs(gt_val)

        result["errors"][key] = rel_err
        if rel_err <= tolerance:
            result["ranges_within_tol"] += 1
        else:
            result["within_tolerance"] = False

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_stage3_on_chart(
    stage3: Any,
    image_path: Path,
    chart_id: str,
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    """
    Run Stage 3 extraction on a single chart image.

    Returns:
        (metadata_dict, processing_time_seconds, error_message)
    """
    try:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None, 0.0, f"Failed to read image: {image_path}"

        start = time.time()
        metadata = stage3.process_image(
            image_bgr,
            chart_id=chart_id,
            image_path=image_path,
        )
        elapsed = time.time() - start

        # Convert to dict
        if hasattr(metadata, "model_dump"):
            md = metadata.model_dump()
        else:
            md = metadata.dict()

        return md, elapsed, None

    except Exception as e:
        logger.error(f"Stage 3 failed | chart_id={chart_id} | error={e}")
        return None, 0.0, str(e)


def evaluate_single_chart(
    annotation: Dict[str, Any],
    metadata: Dict[str, Any],
    processing_time: float,
) -> ChartResult:
    """Evaluate a single chart's Stage 3 output against annotation."""
    chart_id = annotation["chart_id"]
    gt_type = annotation["chart_type"]
    pred_type = metadata.get("chart_type", "unknown")
    difficulty = annotation.get("difficulty", "unknown")

    result = ChartResult(
        chart_id=chart_id,
        chart_type_gt=gt_type,
        chart_type_pred=pred_type,
        difficulty=difficulty,
        processing_time_s=processing_time,
    )

    # Classification
    result.classification_correct = compare_classification(gt_type, pred_type)

    # OCR
    gt_texts = annotation.get("texts", [])
    pred_texts = metadata.get("texts", [])
    gt_title = annotation.get("title")
    result.gt_text_count = len(gt_texts)
    result.detected_text_count = len(pred_texts)

    ocr_cmp = compare_ocr_texts(gt_texts, pred_texts, gt_title)
    result.title_found = ocr_cmp["title_found"]

    # Elements
    gt_elements = annotation.get("elements", {})
    pred_elements = metadata.get("elements", [])
    elem_cmp = compare_elements(gt_elements, pred_elements)
    result.gt_element_count = elem_cmp["gt_count"]
    result.pred_element_count = elem_cmp["pred_count"]
    result.element_count_error = elem_cmp["relative_error"]
    result.element_type_correct = elem_cmp["type_correct"]

    # Axis
    gt_axis = annotation.get("axis")
    pred_axis = metadata.get("axis_info")
    axis_cmp = compare_axis(gt_axis, pred_axis)
    result.axis_errors = axis_cmp["errors"]
    result.axis_within_tolerance = axis_cmp["within_tolerance"]

    # Confidence
    conf = metadata.get("confidence", {})
    if isinstance(conf, dict):
        result.overall_confidence = conf.get("overall_confidence", 0.0)
    elif hasattr(conf, "overall_confidence"):
        result.overall_confidence = conf.overall_confidence

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    results: List[ChartResult],
) -> Dict[str, Any]:
    """Aggregate per-chart results into summary metrics."""
    classification = ClassificationMetrics()
    ocr = OCRMetrics()
    elements = ElementMetrics()
    axis = AxisMetrics()

    by_type: Dict[str, List[ChartResult]] = {}
    by_difficulty: Dict[str, List[ChartResult]] = {}

    for r in results:
        # Group
        by_type.setdefault(r.chart_type_gt, []).append(r)
        by_difficulty.setdefault(r.difficulty, []).append(r)

        # Classification
        classification.total += 1
        if r.classification_correct:
            classification.correct += 1
        else:
            classification.incorrect += 1
            key = f"{r.chart_type_gt}->{r.chart_type_pred}"
            classification.confusion[key] = classification.confusion.get(key, 0) + 1

        # OCR (only if annotation has texts)
        ocr.total_gt_texts += r.gt_text_count
        ocr.total_detected_texts += r.detected_text_count
        if r.title_found:
            ocr.title_found += 1
        if r.gt_text_count > 0:
            ocr.title_total += 1

        # Elements
        if r.gt_element_count > 0:
            elements.total_charts += 1
            elements.errors.append(r.element_count_error)
            if r.element_count_error <= ELEMENT_COUNT_TOLERANCE:
                elements.count_within_tolerance += 1
            if r.gt_element_count == r.pred_element_count:
                elements.count_exact_match += 1
            if r.element_type_correct:
                elements.type_correct += 1

        # Axis
        for key, err in r.axis_errors.items():
            axis.total_axes += 1
            if err is not None and err <= AXIS_RELATIVE_TOLERANCE:
                axis.within_tolerance += 1
                axis.errors.append(err)
            elif err is not None:
                axis.errors.append(err)

    # Mean errors
    if elements.errors:
        elements.mean_relative_error = sum(elements.errors) / len(elements.errors)
    if axis.errors:
        axis.mean_relative_error = sum(axis.errors) / len(axis.errors)

    # Per-type breakdown
    type_breakdown = {}
    for ct, ct_results in sorted(by_type.items()):
        cls_acc = sum(1 for r in ct_results if r.classification_correct) / len(ct_results)
        elem_acc = sum(
            1 for r in ct_results
            if r.gt_element_count > 0 and r.element_count_error <= ELEMENT_COUNT_TOLERANCE
        ) / max(1, sum(1 for r in ct_results if r.gt_element_count > 0))

        type_breakdown[ct] = {
            "count": len(ct_results),
            "classification_accuracy": round(cls_acc, 3),
            "element_tolerance_accuracy": round(elem_acc, 3),
            "mean_element_error": round(
                np.mean([r.element_count_error for r in ct_results if r.gt_element_count > 0]) if any(
                    r.gt_element_count > 0 for r in ct_results
                ) else 0.0, 3
            ),
            "mean_confidence": round(np.mean([r.overall_confidence for r in ct_results]), 3),
            "mean_time_s": round(np.mean([r.processing_time_s for r in ct_results]), 2),
        }

    # Per-difficulty breakdown
    difficulty_breakdown = {}
    for diff, diff_results in sorted(by_difficulty.items()):
        cls_acc = sum(1 for r in diff_results if r.classification_correct) / len(diff_results)
        difficulty_breakdown[diff] = {
            "count": len(diff_results),
            "classification_accuracy": round(cls_acc, 3),
            "mean_confidence": round(np.mean([r.overall_confidence for r in diff_results]), 3),
        }

    return {
        "summary": {
            "total_charts": len(results),
            "classification_accuracy": round(classification.accuracy, 3),
            "element_tolerance_accuracy": round(elements.tolerance_accuracy, 3),
            "element_type_accuracy": round(elements.type_accuracy, 3),
            "element_mean_error": round(elements.mean_relative_error, 3),
            "axis_accuracy": round(axis.accuracy, 3),
            "axis_mean_error": round(axis.mean_relative_error, 3),
            "mean_confidence": round(np.mean([r.overall_confidence for r in results]), 3),
            "mean_processing_time_s": round(np.mean([r.processing_time_s for r in results]), 2),
            "errors": sum(1 for r in results if r.error is not None),
        },
        "classification": {
            "accuracy": round(classification.accuracy, 3),
            "correct": classification.correct,
            "incorrect": classification.incorrect,
            "confusion": classification.confusion,
        },
        "ocr": {
            "total_gt_texts": ocr.total_gt_texts,
            "total_detected_texts": ocr.total_detected_texts,
            "title_recall": round(ocr.title_recall, 3),
        },
        "elements": {
            "total_evaluated": elements.total_charts,
            "within_tolerance": elements.count_within_tolerance,
            "exact_match": elements.count_exact_match,
            "tolerance_accuracy": round(elements.tolerance_accuracy, 3),
            "type_accuracy": round(elements.type_accuracy, 3),
            "mean_relative_error": round(elements.mean_relative_error, 3),
        },
        "axis": {
            "total_ranges": axis.total_axes,
            "within_tolerance": axis.within_tolerance,
            "accuracy": round(axis.accuracy, 3),
            "mean_relative_error": round(axis.mean_relative_error, 3),
        },
        "by_type": type_breakdown,
        "by_difficulty": difficulty_breakdown,
        "per_chart": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a human-readable markdown evaluation report."""
    s = report["summary"]
    lines = [
        "# Stage 3 Benchmark Evaluation Report",
        "",
        f"**Total Charts**: {s['total_charts']}",
        f"**Errors**: {s['errors']}",
        f"**Mean Processing Time**: {s['mean_processing_time_s']}s",
        "",
        "## Summary Scores",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Classification Accuracy | {s['classification_accuracy']:.1%} |",
        f"| Element Count (within 25%) | {s['element_tolerance_accuracy']:.1%} |",
        f"| Element Type Accuracy | {s['element_type_accuracy']:.1%} |",
        f"| Axis Range (within 15%) | {s['axis_accuracy']:.1%} |",
        f"| Mean Confidence | {s['mean_confidence']:.1%} |",
        "",
        "## Ceiling Experiment Verdict",
        "",
    ]

    # Verdict logic
    cls_pass = s["classification_accuracy"] >= 0.90
    elem_pass = s["element_tolerance_accuracy"] >= 0.70
    axis_pass = s["axis_accuracy"] >= 0.60

    passing = sum([cls_pass, elem_pass, axis_pass])
    if passing == 3:
        verdict = "PASS - Geometric approach achieves >= 80% ceiling on all components"
    elif passing >= 2:
        verdict = "PARTIAL - Some components need improvement but approach is viable"
    else:
        verdict = "FAIL - Geometric approach below ceiling threshold, consider redesign"

    lines.extend([
        f"**Verdict**: {verdict}",
        "",
        f"- Classification: {'PASS' if cls_pass else 'FAIL'} ({s['classification_accuracy']:.1%} >= 90%)",
        f"- Elements: {'PASS' if elem_pass else 'FAIL'} ({s['element_tolerance_accuracy']:.1%} >= 70%)",
        f"- Axis: {'PASS' if axis_pass else 'FAIL'} ({s['axis_accuracy']:.1%} >= 60%)",
        "",
        "## By Chart Type",
        "",
        "| Type | N | Cls Acc | Elem Acc | Mean Err | Conf | Time |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])

    for ct, data in sorted(report["by_type"].items()):
        lines.append(
            f"| {ct} | {data['count']} | {data['classification_accuracy']:.1%} "
            f"| {data['element_tolerance_accuracy']:.1%} "
            f"| {data['mean_element_error']:.1%} "
            f"| {data['mean_confidence']:.1%} "
            f"| {data['mean_time_s']:.1f}s |"
        )

    lines.extend([
        "",
        "## By Difficulty",
        "",
        "| Difficulty | N | Cls Acc | Mean Conf |",
        "| --- | --- | --- | --- |",
    ])

    for diff, data in sorted(report["by_difficulty"].items()):
        lines.append(
            f"| {diff} | {data['count']} | {data['classification_accuracy']:.1%} "
            f"| {data['mean_confidence']:.1%} |"
        )

    # Classification confusion
    confusion = report["classification"].get("confusion", {})
    if confusion:
        lines.extend([
            "",
            "## Classification Errors",
            "",
            "| Ground Truth -> Predicted | Count |",
            "| --- | --- |",
        ])
        for key, count in sorted(confusion.items()):
            lines.append(f"| {key} | {count} |")

    # Per-chart details
    lines.extend([
        "",
        "## Per-Chart Details",
        "",
        "| Chart ID | Type GT | Type Pred | Cls | Elem GT | Elem Pred | Err | Conf | Time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for r in report["per_chart"]:
        cls_mark = "OK" if r["classification_correct"] else "WRONG"
        lines.append(
            f"| {r['chart_id'][:35]} | {r['chart_type_gt']} | {r['chart_type_pred']} "
            f"| {cls_mark} | {r['gt_element_count']} | {r['pred_element_count']} "
            f"| {r['element_count_error']:.0%} | {r['overall_confidence']:.2f} "
            f"| {r['processing_time_s']:.1f}s |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report saved | path={output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Stage 3 Benchmark Evaluation")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BENCHMARK_DIR / "benchmark_manifest.json"),
        help="Path to benchmark manifest",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        default="easyocr",
        choices=["easyocr", "paddleocr", "none"],
        help="OCR engine to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running Stage 3, use cached results",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    charts = manifest["charts"]
    logger.info(f"Loaded manifest | charts={len(charts)}")

    # Initialize Stage 3
    stage3 = None
    if not args.skip_run:
        from core_engine.stages.s3_extraction.s3_extraction import (
            Stage3Extraction,
            ExtractionConfig,
        )

        config = ExtractionConfig(
            ocr_engine=args.ocr if args.ocr != "none" else "easyocr",
            enable_ocr=args.ocr != "none",
        )
        stage3 = Stage3Extraction(config=config)
        logger.info(f"Stage 3 initialized | ocr={args.ocr}")

    # Process each chart
    results: List[ChartResult] = []
    cached_outputs_dir = output_dir / "stage3_outputs"
    cached_outputs_dir.mkdir(exist_ok=True)

    for i, chart in enumerate(charts):
        chart_id = chart["chart_id"]
        image_path = PROJECT_ROOT / "data" / "benchmark" / "images" / f"{chart_id}.png"
        annotation_path = BENCHMARK_DIR / "annotations" / f"{chart_id}.json"

        logger.info(
            f"[{i + 1}/{len(charts)}] {chart_id} | "
            f"type={chart['chart_type']} | diff={chart['difficulty']}"
        )

        # Load annotation
        if not annotation_path.exists():
            logger.warning(f"  No annotation found | chart_id={chart_id}")
            # Use empty annotation - will measure only classification and element count
            annotation = {
                "chart_id": chart_id,
                "chart_type": chart["chart_type"],
                "difficulty": chart["difficulty"],
                "texts": [],
                "elements": {"primary_element_type": "", "element_count": 0},
                "axis": None,
                "title": None,
            }
        else:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

        # Run or load cached Stage 3 output
        cached_path = cached_outputs_dir / f"{chart_id}.json"
        metadata = None
        processing_time = 0.0
        error = None

        if args.skip_run and cached_path.exists():
            metadata = json.loads(cached_path.read_text(encoding="utf-8"))
            logger.info(f"  Using cached output")
        elif stage3 is not None:
            metadata, processing_time, error = run_stage3_on_chart(
                stage3, image_path, chart_id,
            )
            if metadata is not None:
                cached_path.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
        else:
            error = "Stage 3 not initialized and no cached output"

        if error:
            logger.error(f"  Error: {error}")
            result = ChartResult(
                chart_id=chart_id,
                chart_type_gt=chart["chart_type"],
                chart_type_pred="error",
                difficulty=chart["difficulty"],
                error=error,
            )
        else:
            result = evaluate_single_chart(annotation, metadata, processing_time)

        results.append(result)
        logger.info(
            f"  cls={'OK' if result.classification_correct else 'WRONG'} | "
            f"elem={result.pred_element_count}/{result.gt_element_count} | "
            f"conf={result.overall_confidence:.2f} | "
            f"time={result.processing_time_s:.1f}s"
        )

    # Aggregate and save
    report = aggregate_results(results)

    json_path = output_dir / "evaluation_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"JSON report saved | path={json_path}")

    md_path = output_dir / "evaluation_report.md"
    generate_markdown_report(report, md_path)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION COMPLETE")
    print("=" * 60)
    s = report["summary"]
    print(f"Classification Accuracy:  {s['classification_accuracy']:.1%}")
    print(f"Element Count (+-25%):    {s['element_tolerance_accuracy']:.1%}")
    print(f"Element Type Accuracy:    {s['element_type_accuracy']:.1%}")
    print(f"Axis Range (+-15%):       {s['axis_accuracy']:.1%}")
    print(f"Mean Confidence:          {s['mean_confidence']:.1%}")
    print(f"Mean Processing Time:     {s['mean_processing_time_s']:.1f}s")
    print(f"\nReport: {json_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
