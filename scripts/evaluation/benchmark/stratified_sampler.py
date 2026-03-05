"""
Stratified Chart Sampler for Benchmark

Selects 50 charts from the academic dataset with intentional diversity:
- 10 bar     (5 simple, 3 complex/stacked, 2 extreme)
- 10 line    (5 single-series, 3 multi-series, 2 complex)
- 10 pie     (5 simple, 3 many-slice, 2 donut/exploded)
- 10 scatter (5 simple, 3 multi-color, 2 log-scale/dense)
- 10 mixed   (3 area, 3 histogram, 2 box, 2 heatmap)

Selection is semi-random with visual diversity heuristics:
- Element count variance (mix low/high)
- Image size variance (small charts are harder)
- Prefer charts from different papers

Usage:
    .venv/Scripts/python.exe scripts/evaluation/benchmark/stratified_sampler.py

Output:
    data/benchmark/benchmark_manifest.json
    data/benchmark/images/  (symlinks or copies)
"""

import hashlib
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIED_DIR = PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts"
FEATURES_DIR = PROJECT_ROOT / "data" / "academic_dataset" / "stage3_features"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
BENCHMARK_IMAGES_DIR = BENCHMARK_DIR / "images"
BENCHMARK_ANNOTATIONS_DIR = BENCHMARK_DIR / "annotations"

SEED = 42

# Stratification spec: (chart_type, count, difficulty_distribution)
# difficulty_distribution = [(difficulty, count), ...]
STRATA: List[Tuple[str, List[Tuple[str, int]]]] = [
    ("bar", [("simple", 5), ("moderate", 3), ("complex", 2)]),
    ("line", [("simple", 5), ("moderate", 3), ("complex", 2)]),
    ("pie", [("simple", 5), ("moderate", 3), ("complex", 2)]),
    ("scatter", [("simple", 5), ("moderate", 3), ("complex", 2)]),
    # Mixed bucket
    ("area", [("simple", 2), ("moderate", 1)]),
    ("histogram", [("simple", 2), ("moderate", 1)]),
    ("box", [("simple", 1), ("moderate", 1)]),
    ("heatmap", [("simple", 1), ("moderate", 1)]),
]


# ---------------------------------------------------------------------------
# Heuristics for difficulty estimation
# ---------------------------------------------------------------------------


def estimate_difficulty(
    chart_type: str,
    features: Dict[str, Any],
    image_path: Path,
) -> str:
    """
    Heuristic difficulty estimation from Stage 3 features and image metadata.

    Uses element count, text count, and image size as proxies.
    Returns 'simple', 'moderate', or 'complex'.
    """
    elem_count = len(features.get("elements", []))
    text_count = len(features.get("texts", []))
    image_size = features.get("image_size", {})
    width = image_size.get("width", 500)
    height = image_size.get("height", 500)
    pixels = width * height

    # Chart-type-specific thresholds
    if chart_type in ("bar", "histogram"):
        if elem_count <= 6:
            return "simple"
        elif elem_count <= 20:
            return "moderate"
        else:
            return "complex"

    elif chart_type == "line":
        # Line charts: complexity from element count (points/line segments)
        if elem_count <= 10:
            return "simple"
        elif elem_count <= 40:
            return "moderate"
        else:
            return "complex"

    elif chart_type in ("pie", "donut"):
        if elem_count <= 5:
            return "simple"
        elif elem_count <= 8:
            return "moderate"
        else:
            return "complex"

    elif chart_type == "scatter":
        if elem_count <= 20:
            return "simple"
        elif elem_count <= 80:
            return "moderate"
        else:
            return "complex"

    elif chart_type == "area":
        if elem_count <= 5:
            return "simple"
        elif elem_count <= 15:
            return "moderate"
        else:
            return "complex"

    elif chart_type == "heatmap":
        if pixels < 200_000:
            return "simple"
        elif pixels < 500_000:
            return "moderate"
        else:
            return "complex"

    elif chart_type == "box":
        if elem_count <= 4:
            return "simple"
        elif elem_count <= 10:
            return "moderate"
        else:
            return "complex"

    # Fallback
    if elem_count <= 8:
        return "simple"
    elif elem_count <= 25:
        return "moderate"
    return "complex"


def extract_paper_id(chart_id: str) -> str:
    """Extract paper identifier from chart_id for diversity."""
    # Format: arxiv_XXXX_XXXXvN_page_N_img_N  or  arxiv_XXXX_XXXXvN_pNN_imgNN
    parts = chart_id.split("_page_")
    if len(parts) >= 2:
        return parts[0]
    parts = chart_id.split("_p")
    if len(parts) >= 2:
        return parts[0]
    return chart_id


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


MAX_CANDIDATES_PER_TYPE = 200  # Pre-sample pool to avoid loading all JSONs


def load_chart_candidates(chart_type: str) -> List[Dict[str, Any]]:
    """
    Load chart candidates for a given type with pre-sampling for speed.

    For types with >MAX_CANDIDATES_PER_TYPE images, randomly pre-sample
    a pool of MAX_CANDIDATES_PER_TYPE before loading features.

    Returns list of dicts with: chart_id, image_path, features, difficulty.
    """
    image_dir = CLASSIFIED_DIR / chart_type
    feature_dir = FEATURES_DIR / chart_type

    if not image_dir.exists():
        logger.warning(f"Image directory not found | chart_type={chart_type}")
        return []

    all_images = sorted(image_dir.glob("*.png"))

    # Pre-sample to avoid loading thousands of JSONs
    if len(all_images) > MAX_CANDIDATES_PER_TYPE:
        logger.info(
            f"  Pre-sampling {MAX_CANDIDATES_PER_TYPE}/{len(all_images)} "
            f"candidates for speed"
        )
        all_images = random.sample(all_images, MAX_CANDIDATES_PER_TYPE)

    candidates = []
    for img_path in all_images:
        chart_id = img_path.stem
        feature_path = feature_dir / f"{chart_id}.json"

        features: Dict[str, Any] = {}
        if feature_path.exists():
            try:
                features = json.loads(feature_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        difficulty = estimate_difficulty(chart_type, features, img_path)

        candidates.append({
            "chart_id": chart_id,
            "chart_type": chart_type,
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "feature_path": str(feature_path.relative_to(PROJECT_ROOT)) if feature_path.exists() else None,
            "difficulty": difficulty,
            "element_count": len(features.get("elements", [])),
            "text_count": len(features.get("texts", [])),
            "paper_id": extract_paper_id(chart_id),
            "image_size": features.get("image_size", {}),
        })

    return candidates


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------


def select_diverse_sample(
    candidates: List[Dict[str, Any]],
    target_difficulty: str,
    count: int,
    used_papers: set,
) -> List[Dict[str, Any]]:
    """
    Select `count` charts matching target_difficulty with paper diversity.

    Prefers charts from papers not yet used in the benchmark.
    Falls back to any chart of matching difficulty if not enough diversity.
    """
    matching = [c for c in candidates if c["difficulty"] == target_difficulty]

    if not matching:
        # Relax difficulty constraint - take any available
        logger.warning(
            f"No candidates for difficulty={target_difficulty} | "
            f"chart_type={candidates[0]['chart_type'] if candidates else '?'} | "
            f"relaxing constraint"
        )
        matching = candidates[:]

    if len(matching) <= count:
        return matching[:count]

    # Prefer diverse papers
    diverse = [c for c in matching if c["paper_id"] not in used_papers]
    from_used = [c for c in matching if c["paper_id"] in used_papers]

    selected = []
    random.shuffle(diverse)
    random.shuffle(from_used)

    # Take from diverse first
    for c in diverse:
        if len(selected) >= count:
            break
        selected.append(c)
        used_papers.add(c["paper_id"])

    # Fill remainder from used papers
    for c in from_used:
        if len(selected) >= count:
            break
        selected.append(c)

    return selected


def run_stratified_sampling() -> List[Dict[str, Any]]:
    """
    Execute stratified sampling to produce the 50-chart benchmark set.

    Returns:
        List of selected chart metadata dicts.
    """
    random.seed(SEED)
    all_selected: List[Dict[str, Any]] = []
    used_papers: set = set()

    for chart_type, difficulty_spec in STRATA:
        logger.info(
            f"Sampling | chart_type={chart_type} | "
            f"target={sum(c for _, c in difficulty_spec)} charts"
        )
        candidates = load_chart_candidates(chart_type)
        logger.info(
            f"  Loaded {len(candidates)} candidates | "
            f"simple={sum(1 for c in candidates if c['difficulty'] == 'simple')}, "
            f"moderate={sum(1 for c in candidates if c['difficulty'] == 'moderate')}, "
            f"complex={sum(1 for c in candidates if c['difficulty'] == 'complex')}"
        )

        for difficulty, count in difficulty_spec:
            selected = select_diverse_sample(
                candidates, difficulty, count, used_papers,
            )
            # Remove selected from candidates to avoid duplicates
            selected_ids = {c["chart_id"] for c in selected}
            candidates = [c for c in candidates if c["chart_id"] not in selected_ids]

            all_selected.extend(selected)
            logger.info(
                f"  Selected {len(selected)} {difficulty} charts | "
                f"ids={[c['chart_id'][:30] for c in selected]}"
            )

    logger.info(f"Total selected: {len(all_selected)} charts")
    return all_selected


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def generate_annotation_templates(
    selected: List[Dict[str, Any]],
) -> None:
    """
    Generate empty annotation JSON templates for each selected chart.

    Creates one JSON file per chart in data/benchmark/annotations/ with
    the annotation schema pre-filled with chart_id and image_path.
    """
    BENCHMARK_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for chart in selected:
        template = {
            "chart_id": chart["chart_id"],
            "image_path": chart["image_path"],
            "chart_type": chart["chart_type"],
            "difficulty": chart["difficulty"],
            "complexity_traits": {
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
                "has_dense_data": False,
                "notes": None,
            },
            "title": None,
            "texts": [
                # Template entries - annotator fills in
                # {"text": "...", "role": "title", "notes": null}
            ],
            "elements": {
                "primary_element_type": _infer_element_type(chart["chart_type"]),
                "element_count": 0,
                "series_count": 1,
                "has_grid_lines": False,
                "has_legend": False,
                "has_data_labels": False,
            },
            "axis": _make_axis_template(chart["chart_type"]),
            "data_series": [
                # Template entry
                # {"name": null, "points": [{"x": ..., "y": ...}]}
            ],
            "annotator": "human",
            "annotation_notes": None,
            "source_paper": chart.get("paper_id"),
        }

        out_path = BENCHMARK_ANNOTATIONS_DIR / f"{chart['chart_id']}.json"
        out_path.write_text(
            json.dumps(template, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info(
        f"Generated {len(selected)} annotation templates | "
        f"dir={BENCHMARK_ANNOTATIONS_DIR}"
    )


def _infer_element_type(chart_type: str) -> str:
    """Infer primary element type from chart type."""
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


def _make_axis_template(chart_type: str) -> Optional[Dict[str, Any]]:
    """Create axis template (None for pie/donut)."""
    if chart_type in ("pie", "donut"):
        return None
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


def copy_images(selected: List[Dict[str, Any]]) -> None:
    """Copy selected chart images to benchmark directory."""
    BENCHMARK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for chart in selected:
        src = PROJECT_ROOT / chart["image_path"]
        dst = BENCHMARK_IMAGES_DIR / f"{chart['chart_id']}.png"
        if src.exists():
            shutil.copy2(src, dst)
        else:
            logger.warning(f"Image not found | path={src}")

    logger.info(f"Copied {len(selected)} images to {BENCHMARK_IMAGES_DIR}")


def save_manifest(selected: List[Dict[str, Any]]) -> Path:
    """Save the benchmark manifest with all metadata."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BENCHMARK_DIR / "benchmark_manifest.json"

    manifest = {
        "version": "1.0.0",
        "total_charts": len(selected),
        "seed": SEED,
        "strata_spec": {
            chart_type: {
                "count": sum(c for _, c in diffs),
                "difficulties": {d: c for d, c in diffs},
            }
            for chart_type, diffs in STRATA
        },
        "distribution": {},
        "charts": selected,
    }

    # Compute actual distribution
    for chart in selected:
        ct = chart["chart_type"]
        diff = chart["difficulty"]
        if ct not in manifest["distribution"]:
            manifest["distribution"][ct] = {"total": 0, "simple": 0, "moderate": 0, "complex": 0}
        manifest["distribution"][ct]["total"] += 1
        manifest["distribution"][ct][diff] += 1

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Manifest saved | path={manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run stratified sampling and generate benchmark artifacts."""
    logger.info("Starting stratified benchmark sampling")
    logger.info(f"Project root: {PROJECT_ROOT}")

    selected = run_stratified_sampling()

    if not selected:
        logger.error("No charts selected - check data directories")
        return

    # Save manifest
    manifest_path = save_manifest(selected)

    # Copy images
    copy_images(selected)

    # Generate annotation templates
    generate_annotation_templates(selected)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SAMPLING COMPLETE")
    print("=" * 60)
    print(f"Total charts: {len(selected)}")
    print(f"Manifest: {manifest_path}")
    print(f"Images: {BENCHMARK_IMAGES_DIR}")
    print(f"Annotations: {BENCHMARK_ANNOTATIONS_DIR}")
    print()

    # Distribution table
    dist: Dict[str, Dict[str, int]] = {}
    for c in selected:
        ct = c["chart_type"]
        d = c["difficulty"]
        if ct not in dist:
            dist[ct] = {"simple": 0, "moderate": 0, "complex": 0, "total": 0}
        dist[ct][d] += 1
        dist[ct]["total"] += 1

    print(f"{'Type':<12} {'Simple':>7} {'Moderate':>9} {'Complex':>8} {'Total':>6}")
    print("-" * 44)
    for ct in sorted(dist.keys()):
        d = dist[ct]
        print(f"{ct:<12} {d['simple']:>7} {d['moderate']:>9} {d['complex']:>8} {d['total']:>6}")
    total = sum(d["total"] for d in dist.values())
    print("-" * 44)
    print(f"{'TOTAL':<12} {'':>7} {'':>9} {'':>8} {total:>6}")

    print(f"\nNext step: Annotate charts in {BENCHMARK_ANNOTATIONS_DIR}/")
    print("Then run: python scripts/evaluation/benchmark/evaluate.py")


if __name__ == "__main__":
    main()
