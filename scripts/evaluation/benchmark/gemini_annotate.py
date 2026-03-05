"""
Gemini Vision Benchmark Annotator

Sends each benchmark chart image to Gemini Vision API to extract
precise ground truth annotations: element count, axis ranges, title,
data values, and structural info.

This replaces the heuristic QA-to-annotation approach with direct
vision-based annotation for maximum accuracy.

Usage:
    .venv/Scripts/python.exe scripts/evaluation/benchmark/gemini_annotate.py

    Options:
        --manifest   Path to benchmark manifest
        --model      Gemini model name (default: gemini-2.5-flash)
        --dry-run    Show prompts without calling API
        --chart-id   Annotate a single chart by ID (for debugging)
        --force      Re-annotate even if annotation already has Gemini data
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[3] / "config" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

ANNOTATION_PROMPT = """You are a precise chart data extraction expert. Analyze this {chart_type} chart image and extract the following information as accurately as possible.

Respond with ONLY a valid JSON object (no markdown, no explanation, no code blocks):

{{
    "title": "exact chart title text or null if none visible",
    "chart_type_verified": "bar|line|pie|scatter|area|histogram|heatmap|box",
    "elements": {{
        "primary_element_type": "bar|point|slice|line|cell|box",
        "element_count": <integer: total number of primary data elements>,
        "series_count": <integer: number of data series/groups>,
        "has_grid_lines": true/false,
        "has_legend": true/false,
        "has_data_labels": true/false
    }},
    "axis": {{
        "x_axis_type": "categorical|linear|log|datetime|none",
        "y_axis_type": "linear|log|categorical|none",
        "x_min": <number or null>,
        "x_max": <number or null>,
        "y_min": <number or null>,
        "y_max": <number or null>,
        "x_label": "x-axis label text or null",
        "y_label": "y-axis label text or null",
        "x_categories": ["list", "of", "category", "labels"] or null,
        "y_categories": null
    }},
    "texts": [
        {{"text": "visible text", "role": "title|x_axis_label|y_axis_label|legend|tick_label|data_label|annotation"}}
    ],
    "data_values": [
        {{"label": "category/point name", "value": <number>, "series": "series name or null"}}
    ],
    "notes": "any relevant observations about chart complexity, readability, or special features"
}}

RULES:
- element_count: Count ALL individual data elements visible in the chart:
  - For bar charts: count each individual bar (a grouped bar chart with 3 groups of 4 bars = 12 bars total)
  - For line charts: count total data points across all lines
  - For pie charts: count the number of slices/segments
  - For scatter plots: count the total number of points (estimate if dense, e.g. "~50")
  - For histograms: count the number of bins
  - For heatmaps: count rows x columns of cells
  - For box plots: count the number of box plots
- series_count: Number of distinct data series (e.g. a multi-line chart with 3 lines = 3)
- axis values: Read the actual min/max from tick labels on the axes. Use null if no axis is visible
- x_categories: List the category labels on x-axis if categorical
- data_values: Extract as many individual data points as you can read from the chart
  - For bar charts: each bar's value
  - For pie charts: each slice's percentage/value
  - For line charts: key data points (start, end, peaks, valleys)
- texts: List ALL visible text in the chart (title, labels, tick marks, legends, annotations)
- If a value is estimated/approximate, note it in the notes field
"""

PIE_EXTRA = """
ADDITIONAL RULES FOR PIE CHARTS:
- element_count = number of slices (including "Other" if present)
- data_values should include the percentage for each slice
- Total percentages should sum to approximately 100%
- axis should be null (pie charts have no axes)
"""

SCATTER_EXTRA = """
ADDITIONAL RULES FOR SCATTER PLOTS:
- element_count = estimated total number of data points
- If points are very dense, provide your best estimate
- Note the approximate range of x and y values from axis labels
"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiAnnotator:
    """Sends chart images to Gemini Vision for structured annotation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY env var."
            )
        self.model = model
        self._init_client()

    def _init_client(self) -> None:
        """Initialize google.genai client."""
        from google import genai
        from google.genai import types

        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"timeout": 120_000},  # 120s timeout per request
        )
        self.types = types
        self.config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=16384,
        )
        logger.info(f"Gemini client initialized | model={self.model}")

    def annotate_chart(
        self,
        image_path: Path,
        chart_type: str,
    ) -> Dict[str, Any]:
        """
        Send chart image to Gemini and get structured annotation.

        Args:
            image_path: Path to chart image file
            chart_type: Expected chart type from manifest

        Returns:
            Parsed annotation dict from Gemini response
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Build prompt
        prompt = ANNOTATION_PROMPT.format(chart_type=chart_type)
        if chart_type == "pie":
            prompt += PIE_EXTRA
        elif chart_type == "scatter":
            prompt += SCATTER_EXTRA

        # Load image bytes
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        mime_type = mime_map.get(suffix, "image/png")

        # Call Gemini with retry on parse failure
        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    self.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=self.config,
            )
            try:
                return self._parse_response(response.text)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Parse failed (attempt {attempt + 1}/{max_retries + 1}), retrying | error={e}"
                    )
                    time.sleep(2.0)
        raise last_error

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parse failed, attempting cleanup | error={e}")
            # Try to extract JSON from the text
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                cleaned = match.group()
                # Fix common issues: trailing commas before } or ]
                cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
                # Fix string values like "~400" for element_count (make into int)
                cleaned = re.sub(
                    r'"element_count"\s*:\s*"~?(\d+)"',
                    r'"element_count": \1',
                    cleaned,
                )
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from Gemini response: {text[:300]}")


# ---------------------------------------------------------------------------
# Annotation merger
# ---------------------------------------------------------------------------

def merge_gemini_into_annotation(
    annotation: Dict[str, Any],
    gemini_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge Gemini Vision annotation into existing benchmark annotation.

    Gemini data overrides existing fields (more accurate).
    """
    # Title
    if gemini_data.get("title"):
        annotation["title"] = gemini_data["title"]

    # Elements
    gem_elem = gemini_data.get("elements", {})
    if gem_elem:
        annotation["elements"] = {
            "primary_element_type": gem_elem.get("primary_element_type", annotation["elements"].get("primary_element_type", "")),
            "element_count": gem_elem.get("element_count", 0),
            "series_count": gem_elem.get("series_count", 1),
            "has_grid_lines": gem_elem.get("has_grid_lines", False),
            "has_legend": gem_elem.get("has_legend", False),
            "has_data_labels": gem_elem.get("has_data_labels", False),
        }

    # Axis
    gem_axis = gemini_data.get("axis")
    if gem_axis:
        # For pie charts, axis should be null
        chart_type = annotation.get("chart_type", "")
        if chart_type == "pie":
            annotation["axis"] = None
        else:
            annotation["axis"] = {
                "x_axis_type": gem_axis.get("x_axis_type", "categorical"),
                "y_axis_type": gem_axis.get("y_axis_type", "linear"),
                "x_min": gem_axis.get("x_min"),
                "x_max": gem_axis.get("x_max"),
                "y_min": gem_axis.get("y_min"),
                "y_max": gem_axis.get("y_max"),
                "x_categories": gem_axis.get("x_categories"),
                "y_categories": gem_axis.get("y_categories"),
                "x_label": gem_axis.get("x_label"),
                "y_label": gem_axis.get("y_label"),
            }

    # Texts
    gem_texts = gemini_data.get("texts", [])
    if gem_texts:
        annotation["texts"] = gem_texts

    # Data values -> data_series
    gem_values = gemini_data.get("data_values", [])
    if gem_values:
        annotation["data_series"] = gem_values

    # Chart type verification
    verified_type = gemini_data.get("chart_type_verified")
    if verified_type:
        annotation["chart_type_verified"] = verified_type

    # Notes
    if gemini_data.get("notes"):
        annotation["annotation_notes"] = gemini_data["notes"]

    annotation["annotator"] = "gemini_vision"

    return annotation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Gemini Vision annotation on benchmark charts."""
    parser = argparse.ArgumentParser(description="Gemini Vision Benchmark Annotator")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BENCHMARK_DIR / "benchmark_manifest.json"),
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chart-id", type=str, default=None, help="Single chart to annotate")
    parser.add_argument("--force", action="store_true", help="Re-annotate even if already done")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    charts = manifest["charts"]

    # Filter to single chart if specified
    if args.chart_id:
        charts = [c for c in charts if c["chart_id"] == args.chart_id]
        if not charts:
            logger.error(f"Chart not found: {args.chart_id}")
            sys.exit(1)

    logger.info(f"Processing {len(charts)} charts | model={args.model} | force={args.force}")

    # Initialize Gemini client
    annotator = None
    if not args.dry_run:
        annotator = GeminiAnnotator(model=args.model)

    stats = {
        "total": 0,
        "annotated": 0,
        "skipped": 0,
        "errors": 0,
    }

    # Save raw Gemini responses for debugging
    raw_dir = BENCHMARK_DIR / "results" / "gemini_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for i, chart in enumerate(charts):
        chart_id = chart["chart_id"]
        chart_type = chart["chart_type"]
        stats["total"] += 1

        ann_path = BENCHMARK_DIR / "annotations" / f"{chart_id}.json"
        image_path = BENCHMARK_DIR / "images" / f"{chart_id}.png"

        if not image_path.exists():
            logger.warning(f"Image not found | chart_id={chart_id}")
            stats["errors"] += 1
            continue

        # Check if already annotated by Gemini
        if ann_path.exists() and not args.force:
            existing = json.loads(ann_path.read_text(encoding="utf-8"))
            if existing.get("annotator") == "gemini_vision":
                logger.info(f"[{i+1}/{len(charts)}] SKIP (already annotated) | {chart_id}")
                stats["skipped"] += 1
                continue

        logger.info(
            f"[{i+1}/{len(charts)}] Annotating | {chart_id} | type={chart_type}"
        )

        if args.dry_run:
            prompt = ANNOTATION_PROMPT.format(chart_type=chart_type)
            logger.info(f"  [DRY RUN] Would send {image_path.name} with prompt ({len(prompt)} chars)")
            stats["annotated"] += 1
            continue

        # Call Gemini (with connection error handling)
        try:
            gemini_data = annotator.annotate_chart(image_path, chart_type)

            # Save raw response
            raw_path = raw_dir / f"{chart_id}.json"
            raw_path.write_text(
                json.dumps(gemini_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Load existing annotation
            if ann_path.exists():
                annotation = json.loads(ann_path.read_text(encoding="utf-8"))
            else:
                annotation = {
                    "chart_id": chart_id,
                    "chart_type": chart_type,
                    "difficulty": chart.get("difficulty", "unknown"),
                    "elements": {},
                    "axis": None,
                    "texts": [],
                    "data_series": [],
                }

            # Merge
            merged = merge_gemini_into_annotation(annotation, gemini_data)

            # Write
            ann_path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            elem_count = merged.get("elements", {}).get("element_count", 0)
            has_axis = (
                merged.get("axis") is not None
                and any(
                    merged["axis"].get(k) is not None
                    for k in ("x_min", "x_max", "y_min", "y_max")
                )
            )
            n_texts = len(merged.get("texts", []))
            n_values = len(merged.get("data_series", []))

            logger.info(
                f"  OK | elem={elem_count} | axis={'Y' if has_axis else 'N'} | "
                f"texts={n_texts} | values={n_values}"
            )
            stats["annotated"] += 1

            # Rate limiting: 1 second between requests
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"  FAILED | chart_id={chart_id} | error={type(e).__name__}: {e}")
            stats["errors"] += 1
            # Continue to next chart
            time.sleep(2.0)

    # Summary
    print("\n" + "=" * 60)
    print("GEMINI ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Total charts:  {stats['total']}")
    print(f"Annotated:     {stats['annotated']}")
    print(f"Skipped:       {stats['skipped']}")
    print(f"Errors:        {stats['errors']}")


if __name__ == "__main__":
    main()
