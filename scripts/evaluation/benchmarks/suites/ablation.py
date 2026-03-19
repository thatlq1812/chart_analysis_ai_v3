"""
Ablation Study Benchmark Suite

Systematically removes or downgrades individual pipeline components to measure
their contribution to overall accuracy. Answers the question:
"Does each component in the hybrid pipeline actually help?"

Configurations:
    full_pipeline       - All components active (control)
    no_classifier       - Skip EfficientNet, use "unknown" chart type
    deplot_only         - Only DePlot extraction, no S4 reasoning
    gemini_only         - Skip local pipeline, send image to Gemini API directly
    no_s4_reasoning     - Skip Stage 4 entirely, output raw Stage 3 results

Metrics (same as e2e_pipeline):
    value_recall, text_recall, anls_title, chart_type_acc, latency_s

Usage:
    runner.run_suite("ablation")
    runner.run_suite("ablation", configs=["full_pipeline", "deplot_only"])
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..metrics import anls, table_value_recall
from ..registry import REGISTRY, BenchmarkResult, BenchmarkSuite, PerChartResult

logger = logging.getLogger(__name__)

# Available ablation configurations
ABLATION_CONFIGS = {
    "full_pipeline": {
        "description": "All components active (control)",
        "use_classifier": True,
        "use_s4_reasoning": True,
        "extractor_backend": "deplot",
        "use_gemini_direct": False,
    },
    "no_classifier": {
        "description": "Skip EfficientNet classifier, chart_type='unknown'",
        "use_classifier": False,
        "use_s4_reasoning": True,
        "extractor_backend": "deplot",
        "use_gemini_direct": False,
    },
    "deplot_only": {
        "description": "DePlot extraction only, no Stage 4 reasoning",
        "use_classifier": True,
        "use_s4_reasoning": False,
        "extractor_backend": "deplot",
        "use_gemini_direct": False,
    },
    "matcha_backend": {
        "description": "MatCha instead of DePlot as VLM extractor",
        "use_classifier": True,
        "use_s4_reasoning": True,
        "extractor_backend": "matcha",
        "use_gemini_direct": False,
    },
    "no_s4_reasoning": {
        "description": "Skip Stage 4 reasoning entirely",
        "use_classifier": True,
        "use_s4_reasoning": False,
        "extractor_backend": "deplot",
        "use_gemini_direct": False,
    },
    "gemini_only": {
        "description": "Skip local pipeline, use Gemini API zero-shot",
        "use_classifier": False,
        "use_s4_reasoning": False,
        "extractor_backend": None,
        "use_gemini_direct": True,
    },
}

DEFAULT_CONFIGS = ["full_pipeline", "no_classifier", "deplot_only", "no_s4_reasoning"]


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
    for axis_key in ("x_axis", "y_axis"):
        axis = annotation.get(axis_key, {})
        for tick in axis.get("tick_labels", []):
            values.append(str(tick))
    return values


def _run_gemini_direct(image_path: Path) -> Dict[str, Any]:
    """Send image directly to Gemini API for zero-shot extraction."""
    try:
        import google.generativeai as genai
        from PIL import Image

        img = Image.open(image_path)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            "Extract ALL data from this chart as JSON with fields: "
            "title, chart_type, series (array of {name, data: [{x, y}]}). "
            "Be precise with numbers."
        )
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return {"parsed": json.loads(text), "raw": text, "error": None}
        except json.JSONDecodeError:
            return {"parsed": None, "raw": text, "error": None}
    except Exception as e:
        return {"parsed": None, "raw": "", "error": str(e)}


@REGISTRY.register("ablation")
class AblationSuite(BenchmarkSuite):
    """
    Systematic ablation study across pipeline configurations.

    Runs multiple configurations on the same chart set and produces
    a comparison table showing each component's contribution.
    """

    name = "ablation"
    description = "Ablation study: measure contribution of each pipeline component"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        **kwargs: Any,
    ) -> BenchmarkResult:
        configs = kwargs.get("configs", DEFAULT_CONFIGS)
        if isinstance(configs, str):
            configs = [configs]
        device = kwargs.get("device", "cpu")

        all_results: List[PerChartResult] = []
        config_aggregates: Dict[str, Dict[str, List[float]]] = {}

        for config_name in configs:
            if config_name not in ABLATION_CONFIGS:
                logger.warning(
                    f"Unknown ablation config '{config_name}', skipping. "
                    f"Available: {list(ABLATION_CONFIGS.keys())}"
                )
                continue

            ablation_cfg = ABLATION_CONFIGS[config_name]
            logger.info(f"\n--- Ablation: {config_name} ---")
            logger.info(f"    {ablation_cfg['description']}")

            score_lists: Dict[str, List[float]] = {
                "value_recall": [],
                "text_recall": [],
                "anls_title": [],
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
                        details={"ablation_config": config_name},
                    ))
                    continue

                gt_values = _extract_gt_values(annotation)
                gt_title = annotation.get("title", "")

                t0 = time.time()
                try:
                    if ablation_cfg["use_gemini_direct"]:
                        result = _run_gemini_direct(image_path)
                        raw_text = result.get("raw", "")
                        pred_title = ""
                        if result.get("parsed"):
                            pred_title = result["parsed"].get("title", "")
                        error_msg = result.get("error")
                    else:
                        raw_text, pred_title, error_msg = self._run_pipeline_config(
                            image_path, ablation_cfg, device
                        )
                    latency = time.time() - t0
                except Exception as e:
                    latency = time.time() - t0
                    raw_text = ""
                    pred_title = ""
                    error_msg = str(e)

                # Compute metrics
                vr = table_value_recall(raw_text, gt_values) if gt_values else {"recall": 1.0, "text_recall": 1.0}
                title_score = anls(pred_title or "", gt_title) if gt_title else 1.0

                scores = {
                    f"{config_name}_value_recall": vr.get("recall", 0.0),
                    f"{config_name}_text_recall": vr.get("text_recall", 0.0),
                    f"{config_name}_anls_title": title_score,
                    f"{config_name}_latency_s": latency,
                }

                score_lists["value_recall"].append(vr.get("recall", 0.0))
                score_lists["text_recall"].append(vr.get("text_recall", 0.0))
                score_lists["anls_title"].append(title_score)
                score_lists["latency_s"].append(latency)

                all_results.append(PerChartResult(
                    chart_id=chart_id,
                    chart_type=annotation.get("chart_type", "unknown"),
                    difficulty=annotation.get("difficulty", "unknown"),
                    success=error_msg is None,
                    error=error_msg,
                    latency_s=latency,
                    scores=scores,
                    details={"ablation_config": config_name},
                ))

            config_aggregates[config_name] = score_lists

        # Build aggregate scores with config prefix
        aggregate: Dict[str, float] = {}
        for config_name, scores in config_aggregates.items():
            for metric, values in scores.items():
                if values:
                    aggregate[f"{config_name}_{metric}"] = round(sum(values) / len(values), 4)

        n_success = sum(1 for r in all_results if r.success)
        n_error = len(all_results) - n_success

        return BenchmarkResult(
            suite_name=self.name,
            run_id="",
            config={
                "ablation_configs": configs,
                "device": device,
                "n_charts": len(chart_ids),
            },
            per_chart=all_results,
            aggregate=aggregate,
            n_success=n_success,
            n_error=n_error,
        )

    def _run_pipeline_config(
        self,
        image_path: Path,
        ablation_cfg: Dict[str, Any],
        device: str,
    ) -> tuple:
        """
        Run pipeline with specific ablation configuration.

        Returns:
            (raw_text, predicted_title, error_message)
        """
        try:
            from core_engine.stages.s3_extraction import ExtractionConfig, Stage3Extraction
            from core_engine.stages.s3_extraction.extractors import create_extractor

            backend = ablation_cfg.get("extractor_backend", "deplot")
            extractor = create_extractor(backend, device=device)

            import cv2
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                return "", "", "failed_to_load_image"

            chart_id = image_path.stem
            result = extractor.extract(image_bgr, chart_id=chart_id)

            # Build raw text from extraction result
            raw_parts = []
            if hasattr(result, "raw_html") and result.raw_html:
                raw_parts.append(result.raw_html)
            if hasattr(result, "headers") and result.headers:
                raw_parts.append(" | ".join(result.headers))
            if hasattr(result, "rows") and result.rows:
                for row in result.rows:
                    raw_parts.append(" | ".join(str(c) for c in row))
            raw_text = "\n".join(raw_parts)

            # Extract title if available
            pred_title = ""
            if hasattr(result, "raw_html") and result.raw_html:
                for line in result.raw_html.split("\n"):
                    if "TITLE" in line.upper():
                        pred_title = line.split("|")[-1].strip() if "|" in line else line.strip()
                        break

            return raw_text, pred_title, None

        except Exception as e:
            logger.warning(f"Pipeline config error: {e}")
            return "", "", str(e)
