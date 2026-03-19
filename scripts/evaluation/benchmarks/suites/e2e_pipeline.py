"""
End-to-End Pipeline Benchmark Suite

Runs the full 5-stage pipeline (S1 Ingestion -> S2 Detection -> S3 Extraction
-> S4 Reasoning -> S5 Reporting) on benchmark charts and compares the final
output against ground truth annotations.

This is the most important benchmark: it measures the TOTAL system accuracy
including error cascading between stages.

Metrics:
    value_recall     - How many GT values appear in final pipeline output
    text_recall      - How many GT text tokens appear in final output
    anls_title       - ANLS score for chart title
    chart_type_acc   - Chart type classification accuracy
    latency_s        - Total pipeline time per chart

Usage:
    runner.run_suite("e2e_pipeline")
    runner.run_suite("e2e_pipeline", extractor_backend="deplot")
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


def _load_pipeline_config() -> Dict[str, Any]:
    """Load pipeline config from config/pipeline.yaml."""
    try:
        from omegaconf import OmegaConf

        config_path = PROJECT_ROOT / "config" / "pipeline.yaml"
        if config_path.exists():
            return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except ImportError:
        pass
    return {}


def _extract_values_from_pipeline_result(result: Any) -> Dict[str, Any]:
    """Extract comparable values from pipeline Stage5 output."""
    output: Dict[str, Any] = {"title": None, "chart_type": None, "values": [], "raw_text": ""}

    if result is None:
        return output

    # Handle Stage5Output or PipelineResult (Pydantic model or dict)
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif hasattr(result, "__dict__"):
        result = result.__dict__

    if isinstance(result, dict):
        # Navigate to chart data
        charts = result.get("charts", [result]) if "charts" in result else [result]
        for chart in charts:
            if isinstance(chart, dict):
                output["title"] = chart.get("title") or chart.get("data", {}).get("title")
                output["chart_type"] = chart.get("chart_type") or chart.get("data", {}).get("chart_type")

                # Extract series data
                data = chart.get("data", chart)
                for series in data.get("series", []):
                    for point in series.get("points", series.get("data", [])):
                        if isinstance(point, dict):
                            if point.get("label"):
                                output["values"].append(str(point["label"]))
                            if point.get("value") is not None:
                                output["values"].append(str(point["value"]))
                            if point.get("x"):
                                output["values"].append(str(point["x"]))
                            if point.get("y") is not None:
                                output["values"].append(str(point["y"]))

        # Build raw text for table_value_recall matching
        output["raw_text"] = json.dumps(result, default=str)

    return output


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


@REGISTRY.register("e2e_pipeline")
class E2EPipelineSuite(BenchmarkSuite):
    """
    End-to-end pipeline benchmark.

    Runs the full chart analysis pipeline on benchmark images and compares
    final output against ground truth. Measures total system accuracy
    including error cascading effects.
    """

    name = "e2e_pipeline"
    description = "Full pipeline S1->S5 accuracy and latency measurement"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        **kwargs: Any,
    ) -> BenchmarkResult:
        device = kwargs.get("device", "cpu")
        extractor_backend = kwargs.get("extractor_backend", "deplot")

        # Lazy-load pipeline to avoid import overhead for other suites
        pipeline = self._build_pipeline(extractor_backend, device)

        results: List[PerChartResult] = []
        score_lists: Dict[str, List[float]] = {
            "value_recall": [],
            "text_recall": [],
            "anls_title": [],
            "chart_type_acc": [],
            "latency_s": [],
        }

        for chart_id in chart_ids:
            image_path = self._find_image(chart_id, images_dir)
            annotation = self._load_annotation(chart_id, annotations_dir)

            if image_path is None or annotation is None:
                results.append(PerChartResult(
                    chart_id=chart_id,
                    chart_type=annotation.get("chart_type", "unknown") if annotation else "unknown",
                    difficulty=annotation.get("difficulty", "unknown") if annotation else "unknown",
                    success=False,
                    error="missing_image_or_annotation",
                ))
                continue

            gt_values = _extract_gt_values(annotation)
            gt_title = annotation.get("title", "")
            gt_type = annotation.get("chart_type", "")

            t0 = time.time()
            try:
                pipeline_result = pipeline.run(image_path)
                latency = time.time() - t0
                extracted = _extract_values_from_pipeline_result(pipeline_result)
                error_msg = None
            except Exception as e:
                latency = time.time() - t0
                extracted = {"title": None, "chart_type": None, "values": [], "raw_text": ""}
                error_msg = str(e)
                logger.warning(f"  Pipeline error | chart_id={chart_id} | error={e}")

            # Compute metrics
            vr = table_value_recall(extracted["raw_text"], gt_values) if gt_values else {"recall": 1.0, "text_recall": 1.0}
            title_score = anls(extracted.get("title") or "", gt_title) if gt_title else 1.0
            type_match = 1.0 if extracted.get("chart_type") == gt_type else 0.0

            scores = {
                "value_recall": vr.get("recall", 0.0),
                "text_recall": vr.get("text_recall", 0.0),
                "anls_title": title_score,
                "chart_type_acc": type_match,
                "latency_s": latency,
            }

            for k, v in scores.items():
                score_lists[k].append(v)

            results.append(PerChartResult(
                chart_id=chart_id,
                chart_type=annotation.get("chart_type", "unknown"),
                difficulty=annotation.get("difficulty", "unknown"),
                success=error_msg is None,
                error=error_msg,
                latency_s=latency,
                scores=scores,
                details={
                    "extractor_backend": extractor_backend,
                    "n_values_extracted": len(extracted.get("values", [])),
                    "n_gt_values": len(gt_values),
                },
            ))

            status = "OK" if error_msg is None else f"ERR: {error_msg[:50]}"
            logger.info(
                f"  {chart_id} | {status} | "
                f"val_recall={vr.get('recall', 0):.3f} | "
                f"type={'OK' if type_match else 'MISS'} | "
                f"time={latency:.1f}s"
            )

        # Build aggregate
        aggregate: Dict[str, float] = {}
        for metric, values in score_lists.items():
            if values:
                aggregate[metric] = round(sum(values) / len(values), 4)

        n_success = sum(1 for r in results if r.success)
        n_error = len(results) - n_success

        return BenchmarkResult(
            suite_name=self.name,
            run_id="",
            config={
                "extractor_backend": extractor_backend,
                "device": device,
                "n_charts": len(chart_ids),
            },
            per_chart=results,
            aggregate=aggregate,
            n_success=n_success,
            n_error=n_error,
        )

    def _build_pipeline(self, extractor_backend: str, device: str) -> Any:
        """Build the chart analysis pipeline for benchmarking."""
        try:
            from core_engine.pipeline import ChartAnalysisPipeline
            from core_engine.stages.s3_extraction import ExtractionConfig

            pipeline = ChartAnalysisPipeline.from_config()

            # Override extraction config for benchmark
            extraction_config = ExtractionConfig(
                extractor_backend=extractor_backend,
                extractor_device=device,
            )
            # Override Stage 3 config if builder supports it
            if hasattr(pipeline, "stages") and len(pipeline.stages) >= 3:
                stage3 = pipeline.stages[2]
                if hasattr(stage3, "config"):
                    stage3.config = extraction_config

            return pipeline
        except Exception as e:
            logger.error(f"Failed to build pipeline: {e}")
            raise RuntimeError(
                f"Cannot build pipeline for benchmark. "
                f"Ensure src/core_engine is importable. Error: {e}"
            ) from e
