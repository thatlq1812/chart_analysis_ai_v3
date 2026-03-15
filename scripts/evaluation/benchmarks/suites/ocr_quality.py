"""
OCR Quality Benchmark Suite

Compares OCR engines on chart text extraction accuracy against
Gemini-annotated ground truth text (titles, axis labels, tick labels, legends).

Engines supported:
    paddleocr   - PaddleOCR v3 (via direct import, no server needed)
    easyocr     - EasyOCR (legacy, for baseline comparison)
    paddlevl    - PaddleOCR-VL via microservice (paddle_server.py)
    qwen2vl     - Qwen2-VL-2B-Instruct as VLM OCR (SVLM extractor)

Metrics:
    precision   - % of predicted texts that match a GT text
    recall      - % of GT texts that were found
    f1          - harmonic mean of precision/recall
    title_recall - specifically for chart titles
    tick_recall  - specifically for tick/axis labels
    cer_mean     - mean Character Error Rate across matched pairs
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..metrics import text_overlap, cer, anls
from ..registry import REGISTRY, BenchmarkResult, BenchmarkSuite, PerChartResult

logger = logging.getLogger(__name__)

DEFAULT_ENGINES = ["paddleocr", "easyocr"]


@REGISTRY.register("ocr_quality")
class OCRQualitySuite(BenchmarkSuite):
    """
    Benchmark OCR engines on chart text extraction.

    For each chart and engine:
      1. Run OCR on chart image -> list of text strings
      2. Compare vs GT texts (all roles: title, tick, axis_label, legend)
      3. Compute precision/recall/F1 and per-role recall
    """

    description = "Compare OCR engines (PaddleOCR vs EasyOCR) on chart text accuracy"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        engines: Optional[List[str]] = None,
        lang: str = "en",
        device: str = "cpu",
        **kwargs: Any,
    ) -> BenchmarkResult:
        engines = engines or DEFAULT_ENGINES
        config = {"engines": engines, "lang": lang, "device": device}

        logger.info(f"OCR Quality benchmark | engines={engines} | charts={len(chart_ids)}")

        per_chart_results: List[PerChartResult] = []
        aggregate_accum: Dict[str, Dict[str, List[float]]] = {e: {} for e in engines}

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

            # Build GT text lists per role
            gt_by_role = self._build_gt_by_role(ann)
            gt_all = [t for ts in gt_by_role.values() for t in ts]

            scores: Dict[str, float] = {}
            details: Dict[str, Any] = {}
            total_latency = 0.0

            for engine_name in engines:
                try:
                    t0 = time.time()
                    predicted_texts = self._run_ocr(engine_name, img_path, lang, device)
                    latency = time.time() - t0
                    total_latency += latency

                    # Overall metrics
                    overall = text_overlap(predicted_texts, gt_all, match_fn="anls", threshold=0.7)

                    # Per-role recall
                    role_recalls: Dict[str, float] = {}
                    for role, gt_role_texts in gt_by_role.items():
                        if gt_role_texts:
                            role_result = text_overlap(
                                predicted_texts, gt_role_texts, match_fn="anls", threshold=0.7
                            )
                            role_recalls[role] = role_result["recall"]

                    # Mean CER on best-matched pairs
                    cer_vals = self._compute_matched_cer(predicted_texts, gt_all)

                    m_scores = {
                        f"{engine_name}.precision": overall["precision"],
                        f"{engine_name}.recall": overall["recall"],
                        f"{engine_name}.f1": overall["f1"],
                        f"{engine_name}.cer_mean": round(sum(cer_vals) / len(cer_vals), 4) if cer_vals else 0.0,
                        f"{engine_name}.latency_s": latency,
                    }
                    for role, recall in role_recalls.items():
                        m_scores[f"{engine_name}.{role}_recall"] = recall

                    scores.update(m_scores)
                    details[engine_name] = {
                        "predicted_texts": predicted_texts[:20],
                        "gt_texts": gt_all[:20],
                        "overall": overall,
                        "role_recalls": role_recalls,
                        "latency_s": round(latency, 2),
                    }

                    for metric_key, metric_val in m_scores.items():
                        aggregate_accum[engine_name].setdefault(metric_key, []).append(metric_val)

                    logger.info(
                        f"OCR | chart_id={chart_id} | engine={engine_name} | "
                        f"P={overall['precision']:.3f} R={overall['recall']:.3f} "
                        f"F1={overall['f1']:.3f} | latency={latency:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"OCR failed | chart_id={chart_id} | engine={engine_name} | error={e}")
                    scores[f"{engine_name}.recall"] = 0.0
                    details[engine_name] = {"error": str(e)}

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

        # Compute aggregate
        aggregate: Dict[str, float] = {}
        for engine_name in engines:
            for metric_key, vals in aggregate_accum[engine_name].items():
                if vals:
                    aggregate[metric_key] = round(sum(vals) / len(vals), 4)

        n_success = len(per_chart_results)
        return BenchmarkResult(
            suite_name="ocr_quality",
            run_id="",
            config=config,
            per_chart=per_chart_results,
            aggregate=aggregate,
            metadata={"engines_tested": engines, "n_charts": len(chart_ids)},
            n_success=n_success,
            n_error=len(chart_ids) - n_success,
        )

    # -------------------------------------------------------------------------
    # OCR backends
    # -------------------------------------------------------------------------

    def _run_ocr(
        self, engine: str, img_path: Path, lang: str, device: str
    ) -> List[str]:
        """Dispatch to the appropriate OCR engine and return list of text strings."""
        if engine == "paddleocr":
            return self._run_paddleocr(img_path, lang)
        elif engine == "easyocr":
            return self._run_easyocr(img_path, lang)
        elif engine == "paddlevl":
            return self._run_paddlevl(img_path)
        elif engine == "qwen2vl":
            return self._run_qwen2vl(img_path, device)
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

    def _run_paddleocr(self, img_path: Path, lang: str) -> List[str]:
        """PaddleOCR v3 direct inference."""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr not installed. Run: pip install paddleocr")
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        result = ocr.ocr(str(img_path), cls=True)
        texts: List[str] = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        texts.append(str(text_info[0]).strip())
        return [t for t in texts if t]

    def _run_easyocr(self, img_path: Path, lang: str) -> List[str]:
        """EasyOCR inference (legacy baseline)."""
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr not installed. Run: pip install easyocr")
        lang_list = ["en"] if lang == "en" else [lang, "en"]
        reader = easyocr.Reader(lang_list, gpu=False, verbose=False)
        results = reader.readtext(str(img_path))
        return [r[1].strip() for r in results if r[1].strip()]

    def _run_paddlevl(self, img_path: Path) -> List[str]:
        """PaddleOCR-VL via HTTP microservice (paddle_server.py must be running)."""
        import requests
        try:
            with open(img_path, "rb") as f:
                resp = requests.post(
                    "http://localhost:8001/extract",
                    files={"file": (img_path.name, f, "image/png")},
                    timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("result", "")
            # Split result into lines as text tokens
            return [line.strip() for line in raw.split("\n") if line.strip()]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "PaddleVL server not running. Start with: python paddle_server.py"
            )

    def _run_qwen2vl(self, img_path: Path, device: str) -> List[str]:
        """Qwen2-VL as OCR - uses SVLM extractor with OCR prompt."""
        try:
            from core_engine.stages.s3_extraction.extractors import create_extractor
        except ImportError:
            raise ImportError("Cannot import create_extractor")
        extractor = create_extractor("svlm", device=device)
        import cv2
        img_bgr = cv2.imread(str(img_path))
        result = extractor.extract(img_bgr, chart_id=img_path.stem)
        if result is None:
            return []
        texts: List[str] = []
        if result.headers:
            texts.extend(str(h) for h in result.headers)
        if result.rows:
            for row in result.rows:
                texts.extend(str(v) for v in row if str(v).strip())
        if result.raw_output:
            texts.extend(line.strip() for line in result.raw_output.split("\n") if line.strip())
        return texts

    # -------------------------------------------------------------------------
    # GT parsing
    # -------------------------------------------------------------------------

    def _build_gt_by_role(self, ann: Dict) -> Dict[str, List[str]]:
        """Group GT texts by role from annotation."""
        by_role: Dict[str, List[str]] = {
            "title": [],
            "tick_label": [],
            "axis_label": [],
            "legend": [],
            "data_label": [],
            "other": [],
        }
        if ann.get("title"):
            by_role["title"].append(ann["title"])

        for t in ann.get("texts", []):
            text = t.get("text", "").strip()
            role = t.get("role", "other")
            if not text:
                continue
            # Map annotation roles to our groups
            if "tick" in role or role in ("x_tick", "y_tick"):
                by_role["tick_label"].append(text)
            elif "axis_label" in role or role in ("x_axis_label", "y_axis_label"):
                by_role["axis_label"].append(text)
            elif role == "legend":
                by_role["legend"].append(text)
            elif "data_label" in role:
                by_role["data_label"].append(text)
            elif role == "title":
                by_role["title"].append(text)
            else:
                by_role["other"].append(text)
        return {k: v for k, v in by_role.items() if v}

    def _compute_matched_cer(
        self, preds: List[str], gts: List[str]
    ) -> List[float]:
        """For each GT, find best matching prediction and compute CER."""
        cer_vals: List[float] = []
        preds_lower = [p.lower().strip() for p in preds]
        for gt in gts[:20]:  # Cap at 20 to avoid slow O(n^2) on large sets
            gt_lower = gt.lower().strip()
            if not gt_lower:
                continue
            best_cer = min(
                (cer(p, gt_lower) for p in preds_lower if p), default=1.0
            )
            cer_vals.append(best_cer)
        return cer_vals
