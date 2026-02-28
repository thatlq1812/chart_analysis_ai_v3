"""
AI Prompts

All prompt templates for the AI reasoning layer.
Prompts are stored here as constants -- NEVER inside adapter classes.

Versioning: each prompt constant carries a version comment so results
can be reproduced by pinning to a git commit.

Usage:
    from core_engine.ai.prompts import CHART_REASONING_SYSTEM, format_reasoning_user
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# System Prompts  (v1.0 - 2026-02-28)
# =============================================================================

CHART_REASONING_SYSTEM = """\
You are an expert chart analysis AI. Your task is to analyze chart metadata
extracted by computer vision and OCR, then produce a clean, structured JSON output.

You will receive:
- chart_type: The detected chart type (bar, line, scatter, pie, etc.)
- ocr_texts: Raw OCR results (may contain errors)
- detected_elements: Visual elements found by CV (axes, bars, lines, etc.)
- axis_info: Pixel-to-value mappings for axes
- color_map: Legend color associations

Your output MUST be valid JSON matching the RefinedChartData schema.
Do NOT include markdown code fences or any text outside the JSON object.
"""

OCR_CORRECTION_SYSTEM = """\
You are an OCR error corrector specialized in chart text.
Common chart OCR errors include:
- Digit confusion: 0/O, 1/l/I, 5/S, 8/B
- Merged tokens: "Q1Q2" instead of "Q1, Q2"
- Truncated text from tight bounding boxes
- Axis value drift: "10.0" read as "1O.0"

Given a list of OCR tokens and chart context, return corrected tokens.
Output MUST be a JSON array of correction objects: [{"original": "...", "corrected": "..."}]
Return an empty array [] if no corrections are needed.
"""

DESCRIPTION_GEN_SYSTEM = """\
You are a scientific writing assistant specializing in data visualization.
Given structured chart data, write a concise, academic-style description
suitable for inclusion in a research paper figure caption.

Requirements:
- 2-4 sentences
- Mention chart type, key trends, notable values
- Use precise, objective language (no subjective opinions)
- Do NOT start with "This chart shows..." or "The figure depicts..."
- Output ONLY the description text, no JSON or extra formatting
"""

DATA_VALIDATION_SYSTEM = """\
You are a data quality validator for chart extraction pipelines.
Given extracted data series and the original chart image description,
identify inconsistencies, missing data, or implausible values.

Output MUST be a JSON object:
{
  "valid": true/false,
  "issues": ["issue1", "issue2"],
  "confidence": 0.0-1.0
}
"""


# =============================================================================
# User Prompt Formatters
# =============================================================================

def format_reasoning_user(
    chart_type: str,
    ocr_texts: List[Dict[str, Any]],
    detected_elements: List[Dict[str, Any]],
    axis_info: Dict[str, Any],
    color_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build the user-turn prompt for CHART_REASONING task.

    Args:
        chart_type: Detected chart type string
        ocr_texts: List of OCR result dicts with keys: text, role, confidence
        detected_elements: Visual element list from Stage 3
        axis_info: Axis mapping from geometric mapper
        color_map: Legend color associations (optional)

    Returns:
        Formatted user prompt string
    """
    import json

    payload: Dict[str, Any] = {
        "chart_type": chart_type,
        "ocr_texts": ocr_texts,
        "detected_elements": detected_elements,
        "axis_info": axis_info,
    }
    if color_map:
        payload["color_map"] = color_map

    return (
        "Analyze the following chart data and return a structured JSON result:\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def format_ocr_correction_user(
    tokens: List[str],
    chart_type: str,
    axis_range: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the user-turn prompt for OCR_CORRECTION task.

    Args:
        tokens: Raw OCR token strings to be corrected
        chart_type: Chart type for context
        axis_range: X/Y axis value ranges for numeric validation

    Returns:
        Formatted user prompt string
    """
    import json

    payload: Dict[str, Any] = {
        "chart_type": chart_type,
        "tokens": tokens,
    }
    if axis_range:
        payload["axis_range"] = axis_range

    return (
        "Correct OCR errors in these chart tokens:\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def format_description_user(
    chart_type: str,
    title: Optional[str],
    x_label: Optional[str],
    y_label: Optional[str],
    series_summary: List[Dict[str, Any]],
) -> str:
    """
    Build the user-turn prompt for DESCRIPTION_GEN task.

    Args:
        chart_type: Detected chart type
        title: Chart title (may be None)
        x_label: X-axis label
        y_label: Y-axis label
        series_summary: List of series dicts with name, min, max, mean

    Returns:
        Formatted user prompt string
    """
    import json

    payload: Dict[str, Any] = {
        "chart_type": chart_type,
        "title": title,
        "x_axis_label": x_label,
        "y_axis_label": y_label,
        "series": series_summary,
    }
    return (
        "Write an academic-style description for this chart:\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )
