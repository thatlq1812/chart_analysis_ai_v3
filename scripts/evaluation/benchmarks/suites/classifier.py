"""
Chart Classifier Benchmark Suite

Evaluates EfficientNet-B0 chart type classifier accuracy on the 50-chart
benchmark set against Gemini-annotated ground truth chart types.

Metrics:
    accuracy        - Overall correct classification rate
    per_type_f1     - Per-class F1 score
    confidence_mean - Mean confidence score for correct predictions
    confidence_fp   - Mean confidence score for false positives (overconfidence check)
    latency_s       - Mean inference time per chart
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..registry import REGISTRY, BenchmarkResult, BenchmarkSuite, PerChartResult

logger = logging.getLogger(__name__)


@REGISTRY.register("classifier")
class ClassifierSuite(BenchmarkSuite):
    """
    Benchmark the EfficientNet-B0 chart type classifier.

    Measures accuracy, per-class F1, confidence calibration, and latency
    on the 50-chart annotated benchmark set.
    """

    description = "EfficientNet-B0 chart type classification accuracy and confidence calibration"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
        **kwargs: Any,
    ) -> BenchmarkResult:
        from omegaconf import OmegaConf
        config = {
            "model_path": model_path or "auto",
            "device": device,
            "confidence_threshold": confidence_threshold,
        }
        logger.info(f"Classifier benchmark | device={device} | charts={len(chart_ids)}")

        classifier = self._load_classifier(model_path, device, confidence_threshold)

        per_chart_results: List[PerChartResult] = []
        # Tracking
        correct = 0
        total = 0
        conf_correct: List[float] = []
        conf_wrong: List[float] = []
        per_type_tp: Dict[str, int] = {}
        per_type_fp: Dict[str, int] = {}
        per_type_fn: Dict[str, int] = {}
        latencies: List[float] = []

        for chart_id in chart_ids:
            ann = self._load_annotation(chart_id, annotations_dir)
            if ann is None:
                continue
            img_path = self._find_image(chart_id, images_dir)
            if img_path is None:
                continue

            chart_type_gt = ann.get("chart_type", "unknown")
            difficulty = ann.get("difficulty", "unknown")

            try:
                import cv2
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    raise ValueError("cv2.imread returned None")

                t0 = time.time()
                pred_type, confidence = self._classify(classifier, img_bgr)
                latency = time.time() - t0
                latencies.append(latency)

                is_correct = self._types_match(pred_type, chart_type_gt)
                total += 1
                if is_correct:
                    correct += 1
                    conf_correct.append(confidence)
                else:
                    conf_wrong.append(confidence)

                # Per-type tracking
                tp = per_type_tp.setdefault(chart_type_gt, 0)
                fp = per_type_fp.setdefault(pred_type, 0)
                fn = per_type_fn.setdefault(chart_type_gt, 0)
                if is_correct:
                    per_type_tp[chart_type_gt] = tp + 1
                else:
                    per_type_fp[pred_type] = fp + 1
                    per_type_fn[chart_type_gt] = fn + 1

                scores = {
                    "correct": 1.0 if is_correct else 0.0,
                    "confidence": confidence,
                }
                details = {
                    "predicted": pred_type,
                    "ground_truth": chart_type_gt,
                    "confidence": round(confidence, 4),
                    "correct": is_correct,
                    "latency_s": round(latency, 3),
                }

                logger.info(
                    f"Classifier | chart_id={chart_id} | gt={chart_type_gt} | "
                    f"pred={pred_type} | conf={confidence:.3f} | "
                    f"{'OK' if is_correct else 'FAIL'} | {latency:.3f}s"
                )

                per_chart_results.append(
                    PerChartResult(
                        chart_id=chart_id,
                        chart_type=chart_type_gt,
                        difficulty=difficulty,
                        success=True,
                        latency_s=latency,
                        scores=scores,
                        details=details,
                    )
                )

            except Exception as e:
                logger.error(f"Classifier failed | chart_id={chart_id} | error={e}")
                per_chart_results.append(
                    PerChartResult(
                        chart_id=chart_id,
                        chart_type=chart_type_gt,
                        difficulty=difficulty,
                        success=False,
                        error=str(e),
                    )
                )

        # Aggregate metrics
        accuracy = correct / total if total > 0 else 0.0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        mean_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
        mean_conf_wrong = sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0.0

        # Per-type F1
        all_types = sorted(set(list(per_type_tp.keys()) + list(per_type_fn.keys())))
        per_type_f1: Dict[str, float] = {}
        f1_sum = 0.0
        for t in all_types:
            tp = per_type_tp.get(t, 0)
            fp = per_type_fp.get(t, 0)
            fn = per_type_fn.get(t, 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_type_f1[t] = round(f1, 4)
            f1_sum += f1
        macro_f1 = f1_sum / len(all_types) if all_types else 0.0

        aggregate: Dict[str, float] = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "mean_confidence_correct": round(mean_conf_correct, 4),
            "mean_confidence_wrong": round(mean_conf_wrong, 4),
            "mean_latency_s": round(mean_latency, 4),
        }
        for t, f1_val in per_type_f1.items():
            aggregate[f"f1_{t}"] = f1_val

        n_success = sum(1 for r in per_chart_results if r.success)
        return BenchmarkResult(
            suite_name="classifier",
            run_id="",
            config=config,
            per_chart=per_chart_results,
            aggregate=aggregate,
            metadata={"n_correct": correct, "n_total": total, "per_type_f1": per_type_f1},
            n_success=n_success,
            n_error=len(chart_ids) - n_success,
        )

    def _load_classifier(
        self,
        model_path: Optional[str],
        device: str,
        confidence_threshold: float,
    ) -> Any:
        """Load EfficientNet-B0 classifier from config or explicit path."""
        try:
            from core_engine.stages.s3_extraction.classifier import EfficientNetClassifier
            from omegaconf import OmegaConf

            if model_path:
                cfg = OmegaConf.create({
                    "classifier": {
                        "model_path": model_path,
                        "device": device,
                        "confidence_threshold": confidence_threshold,
                    }
                })
            else:
                import yaml
                cfg_file = PROJECT_ROOT / "config" / "models.yaml"
                cfg = OmegaConf.load(cfg_file)

            return EfficientNetClassifier(cfg)
        except Exception as e:
            logger.warning(f"EfficientNetClassifier not available: {e}. Using fallback.")
            return None

    def _classify(self, classifier: Any, img_bgr: Any) -> tuple:
        """Run classifier and return (predicted_type_str, confidence)."""
        if classifier is None:
            return "unknown", 0.0
        result = classifier.classify(img_bgr)
        pred_type = result.chart_type.value if hasattr(result.chart_type, "value") else str(result.chart_type)
        return pred_type, result.confidence

    @staticmethod
    def _types_match(pred: str, gt: str) -> bool:
        """Compare predicted vs GT type, handling aliases."""
        aliases = {
            "histogram": "bar",
            "stacked_bar": "bar",
            "grouped_bar": "bar",
            "donut": "pie",
        }
        pred_norm = aliases.get(pred.lower(), pred.lower())
        gt_norm = aliases.get(gt.lower(), gt.lower())
        return pred_norm == gt_norm
