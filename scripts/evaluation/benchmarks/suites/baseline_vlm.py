"""
Baseline VLM Benchmark Suite

Sends chart images directly to commercial VLM APIs (Gemini, GPT-4o) with a
zero-shot prompt asking for structured JSON extraction. This serves as the
**baseline** against which the hybrid pipeline is compared.

Key thesis question: "Does the hybrid multi-stage pipeline outperform a
single large VLM given the same chart image?"

Metrics:
    value_recall     - How many GT numeric values appear in VLM JSON output
    text_recall      - How many GT text tokens appear in VLM JSON output
    anls_title       - ANLS score for chart title extraction
    numeric_accuracy - Fraction of values within 5% tolerance
    latency_s        - API call time per chart

Usage:
    runner.run_suite("baseline_vlm", models=["gemini"])
    runner.run_suite("baseline_vlm", models=["gemini", "openai"])
"""

import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..metrics import anls, numeric_accuracy, table_value_recall, text_overlap
from ..registry import REGISTRY, BenchmarkResult, BenchmarkSuite, PerChartResult

logger = logging.getLogger(__name__)

# Zero-shot extraction prompt
EXTRACTION_PROMPT = """Analyze this chart image and extract ALL data as structured JSON.

Return ONLY a JSON object with this exact schema:
{
  "title": "chart title or null",
  "chart_type": "bar|line|pie|scatter|area|histogram|box|heatmap",
  "x_axis_label": "label or null",
  "y_axis_label": "label or null",
  "series": [
    {
      "name": "series name",
      "data": [{"x": "label", "y": "value"}, ...]
    }
  ]
}

Be precise with numeric values. Extract ALL data points visible in the chart."""

DEFAULT_MODELS = ["gemini"]


def _encode_image_base64(image_path: Path) -> str:
    """Encode image file as base64 string."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _call_gemini(image_path: Path, prompt: str) -> Dict[str, Any]:
    """Call Gemini API with image + text prompt. Returns parsed JSON or raw text."""
    try:
        import google.generativeai as genai
        from PIL import Image

        img = Image.open(image_path)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        # Try to parse as JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return {"parsed": json.loads(text), "raw": text, "error": None}
        except json.JSONDecodeError:
            return {"parsed": None, "raw": text, "error": "json_parse_failed"}
    except Exception as e:
        return {"parsed": None, "raw": "", "error": str(e)}


def _call_openai(image_path: Path, prompt: str) -> Dict[str, Any]:
    """Call OpenAI API with image + text prompt."""
    try:
        from openai import OpenAI

        client = OpenAI()
        b64 = _encode_image_base64(image_path)
        ext = image_path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=2048,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return {"parsed": json.loads(text), "raw": text, "error": None}
        except json.JSONDecodeError:
            return {"parsed": None, "raw": text, "error": "json_parse_failed"}
    except Exception as e:
        return {"parsed": None, "raw": "", "error": str(e)}


VLM_CALLERS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
}


def _extract_values_from_json(parsed: Optional[Dict]) -> List[str]:
    """Extract all data values from parsed VLM JSON response."""
    if not parsed:
        return []
    values = []
    if parsed.get("title"):
        values.append(str(parsed["title"]))
    for series in parsed.get("series", []):
        for point in series.get("data", []):
            if point.get("x"):
                values.append(str(point["x"]))
            if point.get("y"):
                values.append(str(point["y"]))
    return values


def _extract_gt_values(annotation: Dict) -> List[str]:
    """Extract all ground truth values from annotation JSON."""
    values = []
    if annotation.get("title"):
        values.append(annotation["title"])
    for series in annotation.get("data_series", []):
        for point in series.get("points", []):
            if point.get("x") is not None:
                values.append(str(point["x"]))
            if point.get("y") is not None:
                values.append(str(point["y"]))
    # Also collect tick labels
    for axis_key in ("x_axis", "y_axis"):
        axis = annotation.get(axis_key, {})
        for tick in axis.get("tick_labels", []):
            values.append(str(tick))
    return values


@REGISTRY.register("baseline_vlm")
class BaselineVLMSuite(BenchmarkSuite):
    """
    Benchmark commercial VLMs on zero-shot chart data extraction.

    Sends chart image + structured prompt to Gemini/GPT-4o and compares
    the JSON output against Gemini-annotated ground truth.
    """

    name = "baseline_vlm"
    description = "Zero-shot VLM baseline (Gemini/GPT-4o) for chart data extraction"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        **kwargs: Any,
    ) -> BenchmarkResult:
        models = kwargs.get("models", DEFAULT_MODELS)
        if isinstance(models, str):
            models = [models]

        all_results: List[PerChartResult] = []
        model_aggregates: Dict[str, Dict[str, List[float]]] = {}

        for model_name in models:
            if model_name not in VLM_CALLERS:
                logger.warning(f"Unknown model '{model_name}', skipping. Available: {list(VLM_CALLERS.keys())}")
                continue

            caller = VLM_CALLERS[model_name]
            model_scores: Dict[str, List[float]] = {
                "value_recall": [],
                "text_recall": [],
                "anls_title": [],
                "json_valid_rate": [],
                "latency_s": [],
            }

            for chart_id in chart_ids:
                image_path = self._find_image(chart_id, images_dir)
                annotation = self._load_annotation(chart_id, annotations_dir)

                if image_path is None or annotation is None:
                    all_results.append(PerChartResult(
                        chart_id=chart_id,
                        chart_type=annotation.get("chart_type", "unknown") if annotation else "unknown",
                        difficulty=annotation.get("difficulty", "unknown") if annotation else "unknown",
                        success=False,
                        error="missing_image_or_annotation",
                    ))
                    continue

                t0 = time.time()
                result = caller(image_path, EXTRACTION_PROMPT)
                latency = time.time() - t0

                gt_values = _extract_gt_values(annotation)
                gt_title = annotation.get("title", "")

                json_valid = 1.0 if result["parsed"] is not None else 0.0
                pred_values = _extract_values_from_json(result["parsed"])
                raw_text = result.get("raw", "")

                # Compute metrics
                vr = table_value_recall(raw_text, gt_values) if gt_values else {"recall": 1.0, "text_recall": 1.0}
                title_score = anls(
                    result["parsed"].get("title", "") if result["parsed"] else "",
                    gt_title,
                ) if gt_title else 1.0

                scores = {
                    f"{model_name}_value_recall": vr.get("recall", 0.0),
                    f"{model_name}_text_recall": vr.get("text_recall", 0.0),
                    f"{model_name}_anls_title": title_score,
                    f"{model_name}_json_valid": json_valid,
                    f"{model_name}_latency_s": latency,
                }

                model_scores["value_recall"].append(vr.get("recall", 0.0))
                model_scores["text_recall"].append(vr.get("text_recall", 0.0))
                model_scores["anls_title"].append(title_score)
                model_scores["json_valid_rate"].append(json_valid)
                model_scores["latency_s"].append(latency)

                all_results.append(PerChartResult(
                    chart_id=chart_id,
                    chart_type=annotation.get("chart_type", "unknown"),
                    difficulty=annotation.get("difficulty", "unknown"),
                    success=result["error"] is None,
                    error=result.get("error"),
                    latency_s=latency,
                    scores=scores,
                    details={
                        "model": model_name,
                        "raw_output_len": len(raw_text),
                        "json_valid": json_valid > 0,
                    },
                ))

                logger.info(
                    f"  {model_name} | {chart_id} | "
                    f"val_recall={vr.get('recall', 0):.3f} | "
                    f"title_anls={title_score:.3f} | "
                    f"json_ok={json_valid > 0} | "
                    f"time={latency:.1f}s"
                )

            model_aggregates[model_name] = model_scores

        # Build aggregate scores
        aggregate: Dict[str, float] = {}
        for model_name, scores in model_aggregates.items():
            for metric, values in scores.items():
                if values:
                    avg = sum(values) / len(values)
                    aggregate[f"{model_name}_{metric}"] = round(avg, 4)

        n_success = sum(1 for r in all_results if r.success)
        n_error = len(all_results) - n_success

        return BenchmarkResult(
            suite_name=self.name,
            run_id="",
            config={"models": models, "n_charts": len(chart_ids), "prompt": "zero_shot_json"},
            per_chart=all_results,
            aggregate=aggregate,
            n_success=n_success,
            n_error=n_error,
        )
