"""
Enrich benchmark annotations from chart_qa_v2 QA pairs (v2).

Comprehensive extraction of ground truth from Gemini-generated QA pairs.
Covers element counts, axis ranges, titles, and texts for all chart types.

Key improvements over v1:
- Handles `counting` question_type directly
- Parses number words from structural answers ("two", "three")
- Extracts "data point N" max for bar charts
- Counts listed items in threshold/comparison answers
- Extracts axis ranges from range answers ("from X to Y")
- Uses extraction/interpolation answer_values as axis bounds
- Type-filtered element counting (bar->bars, pie->slices, etc.)
- Manual fallback annotations for charts QA cannot resolve

Usage:
    .venv/Scripts/python.exe scripts/evaluation/benchmark/enrich_annotations_v2.py
    .venv/Scripts/python.exe scripts/evaluation/benchmark/enrich_annotations_v2.py --dry-run
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
QA_DIR = PROJECT_ROOT / "data" / "academic_dataset" / "chart_qa_v2" / "generated"

# Map chart types to primary element type names
CHART_TYPE_TO_ELEMENT = {
    "bar": "bar",
    "histogram": "bar",
    "line": "point",
    "scatter": "point",
    "area": "point",
    "pie": "slice",
    "box": "box",
    "heatmap": "cell",
}

# Number words to integers
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


# ---------------------------------------------------------------------------
# Helper: find QA file
# ---------------------------------------------------------------------------


def find_qa_file(chart_id: str) -> Optional[Path]:
    """Find the QA file for a given chart ID across all type folders."""
    matches = list(QA_DIR.glob(f"*/{chart_id}.json"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------


def extract_title(qa_pairs: List[Dict[str, Any]]) -> Optional[str]:
    """Extract chart title from QA pairs."""
    for q in qa_pairs:
        question_lower = q["question"].lower()
        if "title" in question_lower and q.get("answer"):
            answer = q["answer"].strip()
            for prefix in [
                "The title of the chart is ",
                "The title is ",
                "The chart title is ",
                "The chart is titled ",
                "The title of the graph is ",
            ]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):]
                    break
            answer = answer.strip(".").strip("'").strip('"').strip()
            if answer and len(answer) > 1:
                return answer
    return None


# ---------------------------------------------------------------------------
# Element count extraction (v2 - comprehensive)
# ---------------------------------------------------------------------------


def _parse_number(text: str) -> Optional[int]:
    """Parse a number from text (digit or word form)."""
    text_lower = text.strip().lower()
    # Direct digit
    m = re.match(r"^(\d+)", text_lower)
    if m:
        return int(m.group(1))
    # Number word
    for word, val in NUMBER_WORDS.items():
        if text_lower.startswith(word):
            return val
    return None


def _count_listed_items(text: str) -> int:
    """Count comma/and-separated items in a text like 'A, B, C, and D'."""
    # Remove surrounding quotes/parens
    text = text.strip().rstrip(".")
    # Split by comma or " and "
    parts = re.split(r",\s*|\s+and\s+", text)
    # Filter out empty parts and very short fragments
    items = [p.strip() for p in parts if len(p.strip()) > 0]
    return len(items)


def extract_element_count(
    qa_pairs: List[Dict[str, Any]],
    chart_type: str,
) -> Optional[int]:
    """
    Extract element count from QA pairs using multiple strategies.

    Returns the count of visual data elements:
    - bar/histogram: number of individual bars
    - pie: number of slices
    - line/area: number of data series (lines/areas)
    - scatter: number of data points (if determinable)
    - box: number of box groups
    """
    # Strategy 1: Direct `counting` question_type
    for q in qa_pairs:
        if q.get("question_type") == "counting":
            val = q.get("answer_value")
            if val is not None and isinstance(val, (int, float)) and 0 < val < 200:
                return int(val)
            # Parse from answer text
            n = _parse_number(q["answer"])
            if n is not None and 0 < n < 200:
                return n

    # Strategy 2: Direct count questions (how many bars/slices/points)
    count_patterns = [
        r"how many (?:bars?|columns?|groups?|categories|items?|entries|elements?)",
        r"how many (?:slices?|segments?|sections?|portions?|wedges?)",
        r"how many (?:data\s*)?points?",
        r"how many (?:lines?|series|curves?|traces?)",
        r"how many (?:boxes?|box\s*plots?|distributions?)",
        r"number of (?:bars?|slices?|points?|lines?|categories|items)",
    ]
    for q in qa_pairs:
        question_lower = q["question"].lower()
        # Skip hypothetical / derived questions
        if q.get("question_type") in ("multi_hop", "threshold", "interpolation"):
            continue
        if question_lower.strip()[:3] == "if ":
            continue
        for pattern in count_patterns:
            if re.search(pattern, question_lower):
                if q.get("answer_value") is not None:
                    val = q["answer_value"]
                    if isinstance(val, (int, float)) and 0 < val < 200:
                        return int(val)
                n = _parse_number(q["answer"])
                if n is not None and 0 < n < 200:
                    return n

    # Strategy 3: Structural answers mentioning counts
    # e.g., "What are the three categories?" or "two distributions"
    for q in qa_pairs:
        if q.get("question_type") != "structural":
            continue
        question_lower = q["question"].lower()
        answer_lower = q["answer"].lower()

        # "What are the N categories/items?" in question
        m = re.search(
            r"(?:what are the\s+)(\w+)\s+(?:categories|items|groups|bars|slices|"
            r"distributions|datasets?|series|lines?|curves?|colors|variables|"
            r"interaction|classes|types|models?|methods?|approaches)",
            question_lower,
        )
        if m:
            n = _parse_number(m.group(1))
            if n is not None and 1 <= n <= 50:
                return n

        # "There are N ..." in answer
        m = re.search(
            r"there (?:are|is)\s+(\w+)\s+(?:section|slice|bar|point|line|"
            r"serie|curve|categor|distribut|dataset|group|box)",
            answer_lower,
        )
        if m:
            n = _parse_number(m.group(1))
            if n is not None and 1 <= n <= 50:
                return n

    # Strategy 4: Chart-type-specific extraction
    count = _chart_specific_count(qa_pairs, chart_type)
    if count is not None:
        return count

    return None


def _chart_specific_count(
    qa_pairs: List[Dict[str, Any]],
    chart_type: str,
) -> Optional[int]:
    """Chart-type-specific element count strategies."""

    if chart_type in ("bar", "histogram"):
        return _count_bar_elements(qa_pairs)
    elif chart_type == "pie":
        return _count_pie_elements(qa_pairs)
    elif chart_type in ("line", "area"):
        return _count_line_series(qa_pairs)
    elif chart_type == "scatter":
        return _count_scatter_points(qa_pairs)
    elif chart_type == "box":
        return _count_box_elements(qa_pairs)
    elif chart_type == "heatmap":
        return _count_heatmap_elements(qa_pairs)
    return None


def _count_bar_elements(qa_pairs: List[Dict[str, Any]]) -> Optional[int]:
    """Count bars from QA data."""
    # Strategy A: Find max "data point N" reference
    max_data_point = 0
    for q in qa_pairs:
        for text in [q["question"], q["answer"]]:
            for m in re.finditer(r"data\s*point\s*(\d+)", text, re.IGNORECASE):
                dp = int(m.group(1))
                max_data_point = max(max_data_point, dp)
    if max_data_point >= 2:
        return max_data_point

    # Strategy B: Count distinct named items from extraction/comparison
    named_items = set()
    for q in qa_pairs:
        if q.get("question_type") not in ("extraction", "comparison", "threshold"):
            continue
        question = q["question"]
        answer = q["answer"]

        # "What is the value/speed/etc of X?"
        m = re.search(
            r"(?:value|speed|price|accuracy|score|percentage|count|number|rate|"
            r"size|internet speed|frequency|height|width|performance|result|"
            r"cost|revenue|amount|quantity|level|proportion|ratio|weight)"
            r".*?(?:of|for|in)\s+(.+?)(?:\?|$)",
            question,
            re.IGNORECASE,
        )
        if m:
            item = m.group(1).strip().rstrip("?").strip("'\"")
            if len(item) > 1 and len(item) < 60:
                named_items.add(item.lower())

        # "Which X has the highest/lowest Y?" -> answer names one item
        m2 = re.search(
            r"(?:which|what)\s+\w+\s+has\s+(?:the\s+)?(?:highest|lowest|"
            r"largest|smallest|greatest|most|least|best|worst)",
            question,
            re.IGNORECASE,
        )
        if m2 and answer:
            item = answer.strip().rstrip(".").strip("'\"")
            if len(item) > 1 and len(item) < 60:
                named_items.add(item.lower())

    if len(named_items) >= 2:
        return len(named_items)

    # Strategy C: Count items listed in threshold answers
    for q in qa_pairs:
        if q.get("question_type") == "threshold":
            answer = q["answer"]
            # "7 countries meet this threshold: A, B, C, D, E, F, G"
            m = re.search(r"(\d+)\s+\w+\s+(?:meet|exceed|above|below|satisfy)", answer, re.IGNORECASE)
            if m:
                count_from_text = int(m.group(1))
                # Also try to count listed items after ":"
                colon_idx = answer.find(":")
                if colon_idx >= 0:
                    listed = _count_listed_items(answer[colon_idx + 1:])
                    if listed >= count_from_text:
                        return listed
                # Use count from text as minimum
                if 2 <= count_from_text <= 50:
                    return count_from_text

    return None


def _count_pie_elements(qa_pairs: List[Dict[str, Any]]) -> Optional[int]:
    """Count pie slices from QA data."""
    # Count distinct extraction questions with percentage-like values
    percentage_qs = set()
    for q in qa_pairs:
        if q.get("question_type") == "extraction" and q.get("answer_value") is not None:
            val = q["answer_value"]
            if isinstance(val, (int, float)) and 0 < val <= 100:
                percentage_qs.add(q["question"][:50])
    if percentage_qs:
        return max(len(percentage_qs), 2)

    # Count category names from comparison/structural
    categories = set()
    for q in qa_pairs:
        if q.get("question_type") in ("comparison", "structural"):
            # Look for quoted items or listed items
            for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"", q["answer"]):
                item = m.group(1) or m.group(2)
                categories.add(item.lower())
    if len(categories) >= 2:
        return len(categories)

    return None


def _count_line_series(qa_pairs: List[Dict[str, Any]]) -> Optional[int]:
    """Count line/area series from QA data."""
    series_names = set()
    for q in qa_pairs:
        question_lower = q["question"].lower()
        answer_lower = q["answer"].lower()

        # Direct: "How many lines/series?"
        if q.get("question_type") == "structural":
            m = re.search(
                r"how many\s+(?:lines?|series|curves?|traces?|areas?)",
                question_lower,
            )
            if m and q.get("answer_value") is not None:
                val = q["answer_value"]
                if isinstance(val, (int, float)) and 1 <= val <= 50:
                    return int(val)
            if m:
                n = _parse_number(q["answer"])
                if n is not None and 1 <= n <= 50:
                    return n

        # "What are the two distributions?"
        m = re.search(
            r"(?:what are the\s+)(\w+)\s+(?:distributions?|series|lines?|curves?|areas?)",
            question_lower,
        )
        if m:
            n = _parse_number(m.group(1))
            if n is not None:
                return n

        # Collect series names from comparison answers  
        if q.get("question_type") in ("comparison", "structural"):
            # "Full-rank and SwitchLoRA" -> 2 series
            m = re.search(r"(\w[\w\s-]+)\s+and\s+(\w[\w\s-]+)", answer_lower)
            if m:
                series_names.add(m.group(1).strip()[:30])
                series_names.add(m.group(2).strip()[:30])

        # Color-coded series: "blue", "orange", "green", "red"
        colors = {"blue", "orange", "green", "red", "purple", "yellow", "brown",
                  "pink", "gray", "grey", "cyan", "magenta", "black"}
        for color in colors:
            if re.search(rf"\b{color}\b", question_lower + " " + answer_lower):
                series_names.add(color)

        # Named curves: "Fitted curve X", "Model A", etc.
        for m in re.finditer(
            r"(?:fitted curve|curve|model|method|approach|algorithm|line)\s+([\w.-]+)",
            question_lower + " " + answer_lower,
        ):
            series_names.add(m.group(0)[:30])

    if len(series_names) >= 2:
        return len(series_names)

    return None


def _count_scatter_points(qa_pairs: List[Dict[str, Any]]) -> Optional[int]:
    """Count scatter data points from QA data. Often hard to determine."""
    # Direct counting question
    for q in qa_pairs:
        if q.get("question_type") in ("counting", "structural"):
            question_lower = q["question"].lower()
            if re.search(r"how many\s+(?:data\s*)?points?", question_lower):
                if q.get("answer_value") is not None:
                    val = q["answer_value"]
                    if isinstance(val, (int, float)) and 1 <= val <= 500:
                        return int(val)
                n = _parse_number(q["answer"])
                if n is not None and 1 <= n <= 500:
                    return n
    return None


def _count_box_elements(qa_pairs: List[Dict[str, Any]]) -> Optional[int]:
    """Count box plot groups from QA data."""
    categories = set()
    for q in qa_pairs:
        if q.get("question_type") == "structural":
            question_lower = q["question"].lower()
            answer = q["answer"]
            # "What are the three categories?"
            m = re.search(
                r"(?:what are the\s+)(\w+)\s+(?:categories|groups|variables|"
                r"conditions|methods|types|levels|classes|interaction)",
                question_lower,
            )
            if m:
                n = _parse_number(m.group(1))
                if n is not None:
                    return n

            # Count listed items in answer
            if "categor" in question_lower or "group" in question_lower:
                listed = _count_listed_items(answer)
                if listed >= 2:
                    return listed

    # Count from comparison answers
    for q in qa_pairs:
        if q.get("question_type") == "comparison":
            answer_lower = q["answer"].lower()
            # "None", "No 3-var", "No 2-var" -> 3 categories
            m = re.search(r"(\w[\w\s'-]+)\s+(?:has|is|shows)", answer_lower)
            if m:
                categories.add(m.group(1).strip()[:30])

    if len(categories) >= 2:
        return len(categories)

    return None


def _count_heatmap_elements(qa_pairs: List[Dict[str, Any]]) -> Optional[int]:
    """Count heatmap cells from QA data. Usually hard to determine."""
    for q in qa_pairs:
        if q.get("question_type") in ("counting", "structural"):
            question_lower = q["question"].lower()
            if re.search(r"how many\s+(?:cells?|rows?|columns?|items?)", question_lower):
                if q.get("answer_value") is not None:
                    val = q["answer_value"]
                    if isinstance(val, (int, float)) and 1 <= val <= 1000:
                        return int(val)
    return None


# ---------------------------------------------------------------------------
# Axis extraction (v2 - comprehensive)
# ---------------------------------------------------------------------------


def extract_axis_info(
    qa_pairs: List[Dict[str, Any]],
    chart_type: str,
) -> Dict[str, Any]:
    """
    Extract axis range and label information from QA pairs.

    Uses multiple strategies:
    1. Direct range answers ("ranges from X to Y")
    2. Min/max from comparison answers
    3. Axis bounds from extraction/interpolation answer_values
    4. Axis labels from structural questions
    """
    axis: Dict[str, Any] = {
        "x_min": None, "x_max": None,
        "y_min": None, "y_max": None,
        "x_label": None, "y_label": None,
    }

    all_x_values: List[float] = []
    all_y_values: List[float] = []

    for q in qa_pairs:
        question_lower = q["question"].lower()
        answer = q["answer"]
        answer_lower = answer.lower()
        answer_value = q.get("answer_value")

        # --- Axis labels ---
        if q.get("question_type") == "structural":
            # X-axis label
            if re.search(r"x[- ]?axis|horizontal", question_lower) and \
               re.search(r"label|plotted|represent|what is|title", question_lower):
                label = _clean_axis_label(answer)
                if label:
                    axis["x_label"] = label

            # Y-axis label
            if re.search(r"y[- ]?axis|vertical", question_lower) and \
               re.search(r"label|plotted|represent|what is|title", question_lower):
                label = _clean_axis_label(answer)
                if label:
                    axis["y_label"] = label

            # Unit questions -> could extend axis labels
            if "unit" in question_lower:
                pass  # Handled separately if needed

        # --- Range questions ---
        if q.get("question_type") == "range" or "range" in question_lower:
            # "ranges from approximately X to Y" or "from X to Y"
            m = re.search(
                r"(?:ranges?\s+)?(?:from\s+)?(?:approximately\s+)?([\d]+(?:\.\d+)?)\s+"
                r"(?:to|and)\s+(?:approximately\s+)?([\d]+(?:\.\d+)?)",
                answer,
            )
            if m:
                try:
                    lo, hi = float(m.group(1)), float(m.group(2))
                except ValueError:
                    lo, hi = None, None
            if m and lo is not None and hi is not None:
                if lo > hi:
                    lo, hi = hi, lo
                if re.search(r"x[- ]?axis|horizontal|age|month|year|time|date", question_lower):
                    axis["x_min"] = lo
                    axis["x_max"] = hi
                elif re.search(r"y[- ]?axis|vertical|precision|recall|accuracy|"
                               r"frequency|density|score|rate|proportion|percentage|"
                               r"value|speed|price|cost|revenue|rmse|error",
                               question_lower):
                    axis["y_min"] = lo
                    axis["y_max"] = hi
                elif "x" in question_lower[:30] and "y" not in question_lower[:30]:
                    axis["x_min"] = lo
                    axis["x_max"] = hi
                elif "y" in question_lower[:30] and "x" not in question_lower[:30]:
                    axis["y_min"] = lo
                    axis["y_max"] = hi

        # --- Min/max from comparison/extraction ---
        if answer_value is not None and isinstance(answer_value, (int, float)):
            val = float(answer_value)

            if "minimum" in question_lower or "lowest" in question_lower or "smallest" in question_lower:
                if re.search(r"x[- ]?axis|horizontal", question_lower):
                    axis["x_min"] = val
                elif chart_type in ("bar", "line", "area", "histogram", "box"):
                    axis["y_min"] = val
            elif "maximum" in question_lower or "highest" in question_lower or "largest" in question_lower:
                if re.search(r"x[- ]?axis|horizontal", question_lower):
                    axis["x_max"] = val
                elif chart_type in ("bar", "line", "area", "histogram", "box"):
                    axis["y_max"] = val

            # Collect all numeric values for axis bound estimation
            if q.get("question_type") in ("extraction", "interpolation"):
                # Determine if this is an x or y value based on question context
                if re.search(r"x[- ]?axis|horizontal|at what\s+.*\s+value|"
                             r"what\s+.*\s+does|when|at\s+.*\s+=", question_lower):
                    all_x_values.append(val)
                else:
                    all_y_values.append(val)

    # --- Use collected values to fill missing axis bounds ---
    if chart_type in ("scatter", "line", "area"):
        if axis["x_min"] is None and all_x_values:
            axis["x_min"] = min(all_x_values)
        if axis["x_max"] is None and all_x_values:
            axis["x_max"] = max(all_x_values)
    if axis["y_min"] is None and all_y_values and len(all_y_values) >= 2:
        axis["y_min"] = min(all_y_values)
    if axis["y_max"] is None and all_y_values and len(all_y_values) >= 2:
        axis["y_max"] = max(all_y_values)

    # For bar/histogram charts, y_min is often 0
    if chart_type in ("bar", "histogram") and axis["y_max"] is not None and axis["y_min"] is None:
        axis["y_min"] = 0.0

    # Validate: x_min < x_max, y_min < y_max
    if axis["x_min"] is not None and axis["x_max"] is not None:
        if axis["x_min"] > axis["x_max"]:
            axis["x_min"], axis["x_max"] = axis["x_max"], axis["x_min"]
        if axis["x_min"] == axis["x_max"]:
            axis["x_min"] = None
            axis["x_max"] = None
    if axis["y_min"] is not None and axis["y_max"] is not None:
        if axis["y_min"] > axis["y_max"]:
            axis["y_min"], axis["y_max"] = axis["y_max"], axis["y_min"]
        if axis["y_min"] == axis["y_max"]:
            axis["y_min"] = None
            axis["y_max"] = None

    # Return empty dict if no useful data
    has_values = any(axis[k] is not None for k in ("x_min", "x_max", "y_min", "y_max"))
    has_labels = axis["x_label"] or axis["y_label"]
    if not has_values and not has_labels:
        return {}

    return axis


def _clean_axis_label(text: str) -> Optional[str]:
    """Clean axis label from Gemini answer text."""
    val = text.strip().rstrip(".")
    for prefix in [
        "The y-axis represents ", "The x-axis represents ",
        "The y-axis is labeled ", "The x-axis is labeled ",
        "The y-axis label is ", "The x-axis label is ",
        "The y-axis is ", "The x-axis is ",
        "The label of the x-axis is ", "The label of the y-axis is ",
        "The label is ",
    ]:
        if val.lower().startswith(prefix.lower()):
            val = val[len(prefix):]
            break
    val = val.strip().rstrip(".").strip("'\"")
    return val if val and len(val) > 0 else None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_texts(
    qa_pairs: List[Dict[str, Any]],
    title: Optional[str],
) -> List[Dict[str, str]]:
    """Extract text items (title, axis labels, legend items) from QA pairs."""
    texts = []
    seen = set()

    if title:
        texts.append({"text": title, "role": "title"})
        seen.add(title.lower())

    for q in qa_pairs:
        question_lower = q["question"].lower()
        if q.get("question_type") != "structural":
            continue

        # Axis labels
        if re.search(r"x[- ]?axis|horizontal", question_lower) and \
           re.search(r"label|plotted|represent", question_lower):
            label = _clean_axis_label(q["answer"])
            if label and label.lower() not in seen:
                texts.append({"text": label, "role": "x_axis_label"})
                seen.add(label.lower())

        if re.search(r"y[- ]?axis|vertical", question_lower) and \
           re.search(r"label|plotted|represent", question_lower):
            label = _clean_axis_label(q["answer"])
            if label and label.lower() not in seen:
                texts.append({"text": label, "role": "y_axis_label"})
                seen.add(label.lower())

        # Unit questions
        if "unit" in question_lower:
            val = _clean_axis_label(q["answer"])
            if val and val.lower() not in seen:
                texts.append({"text": val, "role": "axis_label"})
                seen.add(val.lower())

    return texts


# ---------------------------------------------------------------------------
# Data values (for reference)
# ---------------------------------------------------------------------------


def extract_data_values(
    qa_pairs: List[Dict[str, Any]],
    chart_type: str,
) -> List[Dict[str, Any]]:
    """Extract individual data values from extraction-type QA pairs."""
    values = []
    for q in qa_pairs:
        if q.get("question_type") == "extraction" and q.get("answer_value") is not None:
            values.append({
                "question": q["question"],
                "value": q["answer_value"],
                "answer_text": q["answer"],
            })
    return values


# ---------------------------------------------------------------------------
# Manual fallback annotations
# ---------------------------------------------------------------------------


# For charts where QA data is insufficient, provide manual GT.
# These are determined by visual inspection of the chart images.
# Format: chart_id -> {element_count, axis_overrides}
MANUAL_ANNOTATIONS: Dict[str, Dict[str, Any]] = {
    # Scatter charts: hard to count exact points from QA
    # Leave element_count=None to skip evaluation for these
    # (scatter points require visual counting or Gemini Vision)
}


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------


def enrich_annotation(
    annotation: Dict[str, Any],
    qa_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Enrich a single annotation with GT from QA data.

    Returns (annotation, change_log) where change_log describes what was added.
    """
    chart_id = annotation["chart_id"]
    chart_type = annotation["chart_type"]
    qa_pairs = qa_data.get("qa_pairs", [])
    changes: Dict[str, str] = {}

    if not qa_pairs:
        return annotation, changes

    # 1. Title
    title = extract_title(qa_pairs)
    if title:
        annotation["title"] = title
        changes["title"] = title[:40]

    # 2. Element count
    element_count = extract_element_count(qa_pairs, chart_type)

    # Check manual overrides
    manual = MANUAL_ANNOTATIONS.get(chart_id, {})
    if "element_count" in manual:
        element_count = manual["element_count"]
        changes["element_count_source"] = "manual"

    if element_count is not None and element_count > 0:
        if annotation.get("elements") is None:
            annotation["elements"] = {}
        annotation["elements"]["element_count"] = element_count
        annotation["elements"]["primary_element_type"] = CHART_TYPE_TO_ELEMENT.get(
            chart_type, "unknown"
        )
        changes["element_count"] = str(element_count)

    # 3. Axis info
    axis_info = extract_axis_info(qa_pairs, chart_type)

    # Check manual axis overrides
    if "axis" in manual:
        for k, v in manual["axis"].items():
            axis_info[k] = v
        changes["axis_source"] = "manual_override"

    if axis_info:
        if annotation.get("axis") is None:
            annotation["axis"] = {
                "x_axis_type": "categorical" if chart_type in ("bar", "histogram", "box") else "linear",
                "y_axis_type": "linear",
                "x_min": None, "x_max": None,
                "y_min": None, "y_max": None,
                "x_categories": None, "y_categories": None,
                "x_label": None, "y_label": None,
            }
        for key, val in axis_info.items():
            if val is not None:
                annotation["axis"][key] = val
        axis_vals = {k: axis_info.get(k) for k in ("x_min", "x_max", "y_min", "y_max") if axis_info.get(k) is not None}
        if axis_vals:
            changes["axis"] = str(axis_vals)

    # 4. Texts
    texts = extract_texts(qa_pairs, title)
    if texts:
        annotation["texts"] = texts
        changes["texts"] = str(len(texts))

    # 5. Data values
    data_values = extract_data_values(qa_pairs, chart_type)
    if data_values:
        annotation["data_series"] = data_values

    # 6. Quality info
    verification = qa_data.get("verification", {})
    if verification.get("chart_quality"):
        annotation["chart_quality"] = verification["chart_quality"]

    annotation["annotator"] = "gemini_qa_v2_enriched"

    return annotation, changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Enrich all benchmark annotations from chart_qa_v2 data (v2)."""
    parser = argparse.ArgumentParser(description="Enrich benchmark annotations (v2)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    manifest_path = BENCHMARK_DIR / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    charts = manifest["charts"]

    logger.info(f"Processing {len(charts)} benchmark charts (v2 enrichment)")

    stats = {
        "total": 0,
        "enriched": 0,
        "title_added": 0,
        "elements_added": 0,
        "axis_values_added": 0,
        "axis_labels_added": 0,
        "texts_added": 0,
        "qa_not_found": 0,
    }

    per_type_stats: Dict[str, Dict[str, int]] = {}

    for chart in charts:
        chart_id = chart["chart_id"]
        chart_type = chart["chart_type"]
        stats["total"] += 1

        if chart_type not in per_type_stats:
            per_type_stats[chart_type] = {"total": 0, "has_count": 0, "has_axis": 0}
        per_type_stats[chart_type]["total"] += 1

        # Find QA file
        qa_file = find_qa_file(chart_id)
        if qa_file is None:
            logger.warning(f"No QA file | chart_id={chart_id}")
            stats["qa_not_found"] += 1
            continue

        qa_data = json.loads(qa_file.read_text(encoding="utf-8"))

        # Load current annotation
        ann_path = BENCHMARK_DIR / "annotations" / f"{chart_id}.json"
        if not ann_path.exists():
            logger.warning(f"No annotation file | chart_id={chart_id}")
            continue

        annotation = json.loads(ann_path.read_text(encoding="utf-8"))

        # Track pre-enrichment state
        had_count = (annotation.get("elements") or {}).get("element_count", 0) > 0
        had_axis_vals = (
            annotation.get("axis") is not None
            and any(
                (annotation.get("axis") or {}).get(k) is not None
                for k in ("x_min", "x_max", "y_min", "y_max")
            )
        )

        # Enrich
        enriched, changes = enrich_annotation(annotation, qa_data)

        # Count changes
        changed = False
        new_count = (enriched.get("elements") or {}).get("element_count", 0)
        new_axis_vals = any(
            (enriched.get("axis") or {}).get(k) is not None
            for k in ("x_min", "x_max", "y_min", "y_max")
        )

        if new_count > 0 and not had_count:
            stats["elements_added"] += 1
            changed = True
        if new_count > 0:
            per_type_stats[chart_type]["has_count"] += 1

        if new_axis_vals and not had_axis_vals:
            stats["axis_values_added"] += 1
            changed = True
        if new_axis_vals:
            per_type_stats[chart_type]["has_axis"] += 1

        if "title" in changes:
            stats["title_added"] += 1
            changed = True
        if "texts" in changes:
            stats["texts_added"] += 1
            changed = True
        if changes.get("axis") and not had_axis_vals:
            pass  # Already counted

        if changed:
            stats["enriched"] += 1

        # Log
        logger.info(
            f"  [{chart_type:10s}] {chart_id[:45]:45s} | "
            f"count={new_count:3d} | "
            f"axis={'Y' if new_axis_vals else 'N'} | "
            f"title={'Y' if enriched.get('title') else 'N'} | "
            f"changes={list(changes.keys())}"
        )

        # Write
        if not args.dry_run:
            ann_path.write_text(
                json.dumps(enriched, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Summary
    print("\n" + "=" * 65)
    print("ANNOTATION ENRICHMENT V2 - SUMMARY")
    print("=" * 65)
    print(f"Total charts:         {stats['total']}")
    print(f"Enriched (new data):  {stats['enriched']}")
    print(f"Titles added:         {stats['title_added']}")
    print(f"Element counts added: {stats['elements_added']}")
    print(f"Axis values added:    {stats['axis_values_added']}")
    print(f"Texts added:          {stats['texts_added']}")
    print(f"QA not found:         {stats['qa_not_found']}")

    print("\n--- Coverage by chart type ---")
    for ct in sorted(per_type_stats):
        s = per_type_stats[ct]
        print(
            f"  {ct:12s}: {s['total']:2d} charts | "
            f"count: {s['has_count']:2d}/{s['total']:2d} | "
            f"axis: {s['has_axis']:2d}/{s['total']:2d}"
        )

    # Count total coverage
    ann_dir = BENCHMARK_DIR / "annotations"
    total_with_count = 0
    total_with_axis = 0
    for f in ann_dir.glob("*.json"):
        ann = json.loads(f.read_text(encoding="utf-8"))
        if (ann.get("elements") or {}).get("element_count", 0) > 0:
            total_with_count += 1
        if any((ann.get("axis") or {}).get(k) is not None for k in ("x_min", "x_max", "y_min", "y_max")):
            total_with_axis += 1
    print(f"\nFinal coverage: {total_with_count}/50 with element_count, {total_with_axis}/50 with axis values")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")


if __name__ == "__main__":
    main()
