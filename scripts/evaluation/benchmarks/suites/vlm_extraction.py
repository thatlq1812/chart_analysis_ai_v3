"""
VLM Extraction Benchmark Suite

Compares DePlot, MatCha, Pix2Struct, and SVLM backends on chart-to-table
derendering accuracy against Gemini-annotated ground truth.

Metrics:
    value_recall     - How many GT numeric values appear in extracted table
    text_recall      - How many GT text tokens appear in extracted table
    anls_title       - ANLS score for chart title extraction
    avg_anls         - Mean ANLS across all GT text tokens
    structure_score  - Whether headers and rows are present (non-empty table)
    latency_s        - Extraction time per chart

Usage:
    runner.run_suite("vlm_extraction", models=["deplot", "matcha"])
    runner.run_suite("vlm_extraction", models=["deplot"])   # quick run
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..metrics import anls, table_value_recall, text_overlap
from ..registry import REGISTRY, BenchmarkResult, BenchmarkSuite, PerChartResult

logger = logging.getLogger(__name__)

# Models to test by default
DEFAULT_MODELS = ["deplot", "matcha", "pix2struct"]

MODEL_SPECS: Dict[str, Tuple[str, Optional[Path]]] = {
    "deplot": ("deplot", PROJECT_ROOT / "models" / "vlm" / "deplot"),
    "matcha": ("matcha", PROJECT_ROOT / "models" / "vlm" / "matcha-base"),
    "matcha_chartqa": (
        "matcha",
        PROJECT_ROOT / "models" / "vlm" / "matcha-chartqa",
    ),
    "pix2struct": (
        "pix2struct",
        PROJECT_ROOT / "models" / "vlm" / "pix2struct-base",
    ),
    "pix2struct_large": (
        "pix2struct",
        PROJECT_ROOT / "models" / "vlm" / "pix2struct-large",
    ),
    "svlm": ("svlm", PROJECT_ROOT / "models" / "vlm" / "qwen2-vl-2b"),
}


@REGISTRY.register("vlm_extraction")
class VLMExtractionSuite(BenchmarkSuite):
    """
    Benchmark VLM chart-to-table extraction backends.

    For each chart and each model:
      1. Load chart image
      2. Run VLM extractor -> get linearized table string
      3. Compare vs GT annotation (tick labels + data values + title)
      4. Compute: value_recall, text_recall, anls_title, structure_score
    """

    description = "Compare DePlot/MatCha/Pix2Struct/SVLM on chart-to-table extraction"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        models: Optional[List[str]] = None,
        device: str = "cpu",
        max_patches: int = 512,
        **kwargs: Any,
    ) -> BenchmarkResult:
        models = models or DEFAULT_MODELS
        config = {"models": models, "device": device, "max_patches": max_patches}

        logger.info(f"VLM Extraction benchmark | models={models} | charts={len(chart_ids)}")

        # Load extractors (lazy-import to avoid heavy deps at module load time)
        extractors = self._load_extractors(models, device, max_patches)
        if not extractors:
            logger.error("No extractors loaded. Check model availability.")

        per_chart_results: List[PerChartResult] = []
        aggregate_accum: Dict[str, Dict[str, List[float]]] = {m: {} for m in models}

        for chart_id in chart_ids:
            ann = self._load_annotation(chart_id, annotations_dir)
            if ann is None:
                logger.warning(f"No annotation | chart_id={chart_id}")
                continue

            img_path = self._find_image(chart_id, images_dir)
            if img_path is None:
                logger.warning(f"Image not found | chart_id={chart_id}")
                continue

            chart_type = ann.get("chart_type", "unknown")
            difficulty = ann.get("difficulty", "unknown")
            gt_texts = self._extract_gt_texts(ann)
            gt_values = [t for t in gt_texts if self._is_numeric(t)]
            gt_all_texts = gt_texts
            gt_title = ann.get("title", "")

            scores: Dict[str, float] = {}
            details: Dict[str, Any] = {}
            total_latency = 0.0

            for model_name, extractor in extractors.items():
                try:
                    import cv2
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        raise ValueError(f"cv2.imread returned None for {img_path}")

                    t0 = time.time()
                    result = extractor.extract(img_bgr, chart_id=chart_id)
                    latency = time.time() - t0
                    total_latency += latency

                    if result is not None:
                        table_str = self._result_to_table_str(result)
                        structure_score = 1.0 if (result.headers or result.rows) else 0.0
                    else:
                        table_str = ""
                        structure_score = 0.0

                    val_rec = table_value_recall(table_str, gt_values)
                    txt_rec = table_value_recall(table_str, gt_all_texts)
                    anls_title = anls(table_str, gt_title) if gt_title else 1.0

                    m_scores = {
                        f"{model_name}.value_recall": val_rec["recall"],
                        f"{model_name}.text_recall": txt_rec["text_recall"],
                        f"{model_name}.anls_title": anls_title,
                        f"{model_name}.structure_score": structure_score,
                        f"{model_name}.latency_s": latency,
                    }
                    scores.update(m_scores)
                    details[model_name] = {
                        "table_str": table_str[:300] if table_str else "",
                        "value_recall": val_rec,
                        "text_recall": txt_rec,
                        "latency_s": round(latency, 2),
                    }

                    # Accumulate for aggregate
                    for metric_key, metric_val in m_scores.items():
                        agg_key = metric_key
                        aggregate_accum[model_name].setdefault(agg_key, []).append(metric_val)

                    logger.info(
                        f"VLM | chart_id={chart_id} | model={model_name} | "
                        f"val_recall={val_rec['recall']:.3f} | "
                        f"txt_recall={txt_rec['text_recall']:.3f} | "
                        f"latency={latency:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Extraction failed | chart_id={chart_id} | model={model_name} | error={e}")
                    scores[f"{model_name}.value_recall"] = 0.0
                    scores[f"{model_name}.error"] = 1.0
                    details[model_name] = {"error": str(e)}

            per_chart_results.append(
                PerChartResult(
                    chart_id=chart_id,
                    chart_type=chart_type,
                    difficulty=difficulty,
                    success=True,
                    latency_s=total_latency,
                    scores=scores,
                    details=details,
                )
            )

        # Compute aggregate (mean per metric across charts)
        aggregate: Dict[str, float] = {}
        for model_name in models:
            for metric_key, vals in aggregate_accum[model_name].items():
                if vals:
                    aggregate[metric_key] = round(sum(vals) / len(vals), 4)

        # Add per-model summary scores for easy comparison
        for model_name in models:
            vr = aggregate.get(f"{model_name}.value_recall", 0.0)
            tr = aggregate.get(f"{model_name}.text_recall", 0.0)
            ss = aggregate.get(f"{model_name}.structure_score", 0.0)
            aggregate[f"{model_name}.overall"] = round((vr + tr + ss) / 3, 4)

        n_success = len(per_chart_results)
        result = BenchmarkResult(
            suite_name="vlm_extraction",
            run_id="",  # set by runner
            config=config,
            per_chart=per_chart_results,
            aggregate=aggregate,
            metadata={
                "models_tested": models,
                "n_charts": len(chart_ids),
                "n_success": n_success,
            },
            n_success=n_success,
            n_error=len(chart_ids) - n_success,
        )
        return result

    def _load_extractors(
        self,
        models: List[str],
        device: str,
        max_patches: int,
    ) -> Dict[str, Any]:
        """Load VLM extractor instances, skipping unavailable ones."""
        try:
            from core_engine.stages.s3_extraction.extractors import create_extractor
        except ImportError:
            logger.error("Cannot import create_extractor. Is src/ in sys.path?")
            return {}

        loaded: Dict[str, Any] = {}
        for model_name in models:
            try:
                backend_name, model_override = self._resolve_model_spec(model_name)
                extractor = create_extractor(
                    backend=backend_name,
                    model_override=model_override,
                    device=device,
                    max_patches=max_patches,
                )
                loaded[model_name] = extractor
                logger.info(
                    f"Loaded extractor | backend={model_name} | resolved_backend={backend_name} | "
                    f"model_override={model_override or 'default'} | device={device}"
                )
            except Exception as e:
                logger.warning(f"Could not load extractor | backend={model_name} | error={e}")
        return loaded

    def _resolve_model_spec(self, model_name: str) -> Tuple[str, Optional[str]]:
        """Resolve benchmark alias to extractor backend + optional local model path."""
        backend_name, local_path = MODEL_SPECS.get(model_name, (model_name, None))
        if local_path is not None and local_path.exists():
            return backend_name, str(local_path)
        return backend_name, None

    def _result_to_table_str(self, result: Any) -> str:
        """Convert Pix2StructResult to a single string for text matching."""
        parts = []
        if result.headers:
            parts.append(" | ".join(str(h) for h in result.headers))
        if result.rows:
            for row in result.rows:
                parts.append(" | ".join(str(v) for v in row))
        raw_text = getattr(result, "raw_html", "")
        if raw_text:
            parts.append(raw_text)
        return "\n".join(parts)

    def _extract_gt_texts(self, ann: Dict) -> List[str]:
        """Extract all ground truth text tokens from annotation."""
        texts = []
        if ann.get("title"):
            texts.append(ann["title"])
        for t in ann.get("texts", []):
            text = t.get("text", "").strip()
            if text:
                texts.append(text)
        # Add values from data_values if present
        for series in ann.get("data_values", {}).get("series", []):
            for pt in series.get("points", []):
                if pt.get("y") is not None:
                    texts.append(str(pt["y"]))
                if pt.get("x") is not None:
                    texts.append(str(pt["x"]))
                if pt.get("value") is not None:
                    texts.append(str(pt["value"]))
        return texts

    @staticmethod
    def _is_numeric(text: str) -> bool:
        try:
            float(text.replace(",", "").replace("%", "").strip())
            return True
        except ValueError:
            return False
