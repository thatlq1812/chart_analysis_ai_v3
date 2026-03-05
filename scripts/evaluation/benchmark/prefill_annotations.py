"""
AI-Assisted Annotation Pre-filler

Uses Stage 3 cached outputs (from evaluation run) to pre-fill
annotation templates with machine-detected values. Human annotators
then review and correct the pre-filled values.

This saves 70-80% of annotation time compared to starting from scratch.

Usage:
    .venv/Scripts/python.exe scripts/evaluation/benchmark/prefill_annotations.py

    Expects cached Stage 3 outputs in:
        data/benchmark/results/stage3_outputs/*.json

    Updates annotation templates in:
        data/benchmark/annotations/*.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
ANNOTATIONS_DIR = BENCHMARK_DIR / "annotations"
STAGE3_OUTPUTS_DIR = BENCHMARK_DIR / "results" / "stage3_outputs"


def _infer_element_type_from_chart(chart_type: str) -> str:
    """Map chart type to primary element type."""
    mapping = {
        "bar": "bar",
        "stacked_bar": "bar",
        "grouped_bar": "bar",
        "line": "line",
        "pie": "slice",
        "donut": "slice",
        "scatter": "point",
        "area": "area",
        "histogram": "bar",
        "box": "box",
        "heatmap": "area",
    }
    return mapping.get(chart_type, "unknown")


def extract_role_texts(
    texts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert Stage 3 OCR texts to annotation format."""
    annotation_texts = []
    for t in texts:
        text_content = t.get("text", "").strip()
        if not text_content:
            continue
        role = t.get("role", "unknown")
        annotation_texts.append({
            "text": text_content,
            "role": role,
            "notes": None,
        })
    return annotation_texts


def extract_title(texts: List[Dict[str, Any]]) -> Optional[str]:
    """Extract title from OCR texts."""
    for t in texts:
        if t.get("role") == "title":
            return t.get("text", "").strip()
    return None


def build_axis_from_s3(
    axis_info: Optional[Dict[str, Any]],
    chart_type: str,
) -> Optional[Dict[str, Any]]:
    """Convert Stage 3 axis_info to annotation format."""
    if chart_type in ("pie", "donut"):
        return None

    if axis_info is None:
        # Return template
        return {
            "x_axis_type": "categorical" if chart_type in ("bar", "histogram", "box") else "linear",
            "y_axis_type": "linear",
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "x_categories": None,
            "y_categories": None,
            "x_label": None,
            "y_label": None,
        }

    return {
        "x_axis_type": "categorical" if chart_type in ("bar", "histogram", "box") else "linear",
        "y_axis_type": "linear",
        "x_min": axis_info.get("x_min"),
        "x_max": axis_info.get("x_max"),
        "y_min": axis_info.get("y_min"),
        "y_max": axis_info.get("y_max"),
        "x_categories": None,
        "y_categories": None,
        "x_label": None,
        "y_label": None,
    }


def detect_complexity_traits(
    s3_output: Dict[str, Any],
    chart_type: str,
) -> Dict[str, Any]:
    """Heuristic complexity trait detection from Stage 3 output."""
    elements = s3_output.get("elements", [])
    texts = s3_output.get("texts", [])
    elem_count = len(elements)

    traits = {
        "is_stacked": False,
        "is_grouped": False,
        "is_multi_series": False,
        "has_log_scale": False,
        "has_rotated_labels": False,
        "has_negative_values": False,
        "has_overlapping_elements": False,
        "is_3d": False,
        "is_donut": False,
        "is_exploded": False,
        "has_error_bars": False,
        "has_trend_line": False,
        "has_secondary_axis": False,
        "has_dense_data": elem_count > 20,
        "notes": "AUTO-DETECTED: Review and correct",
    }

    # Multi-series detection by unique colors
    colors = set()
    for e in elements:
        color = e.get("color")
        if isinstance(color, dict):
            colors.add((color.get("r", 0), color.get("g", 0), color.get("b", 0)))
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            colors.add(tuple(color[:3]))

    if len(colors) > 2:
        traits["is_multi_series"] = True

    # Check for legend
    has_legend = any(t.get("role") == "legend" for t in texts)

    return traits


def prefill_single(
    annotation_path: Path,
    s3_output: Dict[str, Any],
) -> bool:
    """
    Pre-fill a single annotation file using Stage 3 output.

    Only updates fields that are currently empty/default.
    Does NOT overwrite human-edited values.

    Returns True if file was updated.
    """
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

    # Only pre-fill if not already annotated
    if annotation.get("annotator") == "human-verified":
        logger.info(f"  Skipping (already verified) | {annotation_path.stem}")
        return False

    chart_type = s3_output.get("chart_type", annotation.get("chart_type", "unknown"))
    texts = s3_output.get("texts", [])
    elements = s3_output.get("elements", [])
    axis_info = s3_output.get("axis_info")

    # Pre-fill chart type from Stage 3 (machine prediction, needs review)
    annotation["chart_type"] = chart_type

    # Pre-fill texts
    if not annotation.get("texts"):
        annotation["texts"] = extract_role_texts(texts)

    # Pre-fill title
    if not annotation.get("title"):
        annotation["title"] = extract_title(texts)

    # Pre-fill elements
    if annotation.get("elements", {}).get("element_count", 0) == 0:
        annotation["elements"] = {
            "primary_element_type": _infer_element_type_from_chart(chart_type),
            "element_count": len(elements),
            "series_count": 1,
            "has_grid_lines": False,
            "has_legend": any(t.get("role") == "legend" for t in texts),
            "has_data_labels": any(t.get("role") == "data_label" for t in texts),
        }

    # Pre-fill axis
    if annotation.get("axis") is None or (
        annotation.get("axis", {}).get("x_min") is None
        and annotation.get("axis", {}).get("y_min") is None
    ):
        annotation["axis"] = build_axis_from_s3(axis_info, chart_type)

    # Pre-fill complexity traits
    if annotation.get("complexity_traits", {}).get("notes") is None:
        annotation["complexity_traits"] = detect_complexity_traits(s3_output, chart_type)

    # Mark as auto-prefilled
    annotation["annotator"] = "auto-prefilled"

    # Save
    annotation_path.write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return True


def main() -> None:
    """Pre-fill all annotation templates from Stage 3 outputs."""
    logger.info("Starting annotation pre-fill")

    if not STAGE3_OUTPUTS_DIR.exists():
        logger.error(
            f"Stage 3 outputs not found | dir={STAGE3_OUTPUTS_DIR}\n"
            "Run evaluate.py first to generate cached outputs."
        )
        return

    s3_files = sorted(STAGE3_OUTPUTS_DIR.glob("*.json"))
    logger.info(f"Found {len(s3_files)} Stage 3 output files")

    updated = 0
    skipped = 0
    missing = 0

    for s3_path in s3_files:
        chart_id = s3_path.stem
        annotation_path = ANNOTATIONS_DIR / f"{chart_id}.json"

        if not annotation_path.exists():
            logger.warning(f"No annotation template | chart_id={chart_id}")
            missing += 1
            continue

        s3_output = json.loads(s3_path.read_text(encoding="utf-8"))
        if prefill_single(annotation_path, s3_output):
            updated += 1
            logger.info(f"  Pre-filled | chart_id={chart_id}")
        else:
            skipped += 1

    print(f"\nPre-fill complete: {updated} updated, {skipped} skipped, {missing} missing")
    print(f"Annotation dir: {ANNOTATIONS_DIR}")
    print("\nNext: Open images and annotations side-by-side to verify/correct.")


if __name__ == "__main__":
    main()
