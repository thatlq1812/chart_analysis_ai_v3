#!/usr/bin/env python3
"""
Full Pipeline Demo: Stage 1 → Stage 5

Runs the complete chart analysis pipeline on all three sample images
(bar, line, pie) using the latest production models:
  - Stage 2 (detection):  yolo_chart_detector.pt
  - Stage 3 (classifier): efficientnet_b0_3class_v1_best.pt  (97.54% acc)
  - Stage 4 (reasoning):  AIRouter → Gemini / OpenAI fallback chain
  - Stage 5 (reporting):  JSON + text report

Usage:
    .venv/Scripts/python.exe scripts/pipeline/run_demo_s1_s5.py
    .venv/Scripts/python.exe scripts/pipeline/run_demo_s1_s5.py --no-llm
    .venv/Scripts/python.exe scripts/pipeline/run_demo_s1_s5.py --image path/to/custom.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

# Suppress paddle / OCR verbose logs
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_s1_s5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_id() -> str:
    return f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _config_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _hr(char: str = "-", width: int = 70) -> str:
    return char * width


def _section(title: str) -> None:
    print()
    print(_hr("="))
    print(f"  {title}")
    print(_hr("="))


def _sub(title: str) -> None:
    print(_hr("-"))
    print(f"  {title}")
    print(_hr("-"))


# ---------------------------------------------------------------------------
# Stage 1: Ingestion
# ---------------------------------------------------------------------------

def run_stage1(image_paths: List[Path]) -> "Stage1Output":
    """
    Simulate Stage 1 Ingestion: wrap raw image paths into Stage1Output.

    In production, Stage 1 processes PDFs/DOCX and normalises images.
    For this demo, sample images are already clean PNGs.
    """
    from core_engine.schemas.stage_outputs import Stage1Output, CleanImage
    from core_engine.schemas.common import SessionInfo

    session = SessionInfo(
        session_id=_session_id(),
        source_file=image_paths[0],
        total_pages=len(image_paths),
        config_hash=_config_hash("demo_s1_s5_v1"),
    )

    clean_images = []
    for idx, p in enumerate(image_paths, start=1):
        img = cv2.imread(str(p))
        h, w = img.shape[:2] if img is not None else (0, 0)
        clean_images.append(
            CleanImage(
                image_path=p,
                original_path=p,
                page_number=idx,
                width=w,
                height=h,
                is_grayscale=False,
                source_format=p.suffix.lstrip(".").lower(),
            )
        )

    output = Stage1Output(session=session, images=clean_images, warnings=[])
    logger.info(
        f"Stage 1 done | session={session.session_id} | images={output.total_images}"
    )
    return output, session


# ---------------------------------------------------------------------------
# Stage 2: Detection
# ---------------------------------------------------------------------------

def run_stage2(
    stage1_output: "Stage1Output",
    session: "SessionInfo",
    yolo_model_path: Path,
    output_dir: Path,
) -> "Stage2Output":
    """
    Run Stage 2 YOLO detection.

    Falls back to whole-image crop if YOLO detects nothing (sample images
    are already cropped charts — confidence may be high or low depending
    on the model's training distribution).
    """
    from core_engine.stages.s2_detection import Stage2Detection, DetectionConfig
    from core_engine.schemas.stage_outputs import Stage2Output, DetectedChart
    from core_engine.schemas.common import BoundingBox

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    config = DetectionConfig(
        model_path=yolo_model_path,
        device="auto",
        confidence_threshold=0.35,  # lower threshold; sample images are already cropped
        output_dir=crops_dir,
    )

    detector = Stage2Detection(config)
    stage2_out = detector.process(stage1_output)

    # Fallback: if YOLO found nothing, treat entire image as the chart region
    if not stage2_out.charts:
        logger.warning("YOLO found no charts — using full-image fallback for all images")
        fallback_charts = []
        for ci in stage1_output.images:
            fallback_crop = crops_dir / f"{ci.image_path.stem}_fullimg.png"
            shutil.copy(ci.image_path, fallback_crop)
            img = cv2.imread(str(ci.image_path))
            h, w = img.shape[:2] if img is not None else (900, 900)
            fallback_charts.append(
                DetectedChart(
                    chart_id=f"{ci.image_path.stem}_full",
                    source_image=ci.image_path,
                    cropped_path=fallback_crop,
                    bbox=BoundingBox(
                        x_min=0, y_min=0, x_max=w, y_max=h,
                        normalized_x_min=0.0, normalized_y_min=0.0,
                        normalized_x_max=1.0, normalized_y_max=1.0,
                    ),
                    page_number=ci.page_number,
                )
            )
        stage2_out = Stage2Output(
            session=session,
            charts=fallback_charts,
            total_detected=len(fallback_charts),
            skipped_low_confidence=0,
        )
    else:
        logger.info(f"Stage 2 done | charts_detected={stage2_out.total_detected}")

    return stage2_out


# ---------------------------------------------------------------------------
# Stage 3: Extraction
# ---------------------------------------------------------------------------

def run_stage3(
    stage2_output: "Stage2Output",
    efficientnet_path: Path,
) -> "Stage3Output":
    """Run Stage 3 extraction with EfficientNet-B0 3-class classifier."""
    from core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig

    config = ExtractionConfig(
        ocr_engine="paddleocr",
        enable_vectorization=True,
        enable_element_detection=True,
        enable_ocr=True,
        enable_classification=True,
        # EfficientNet-B0 3-class (97.54% acc) — new production model
        use_efficientnet_classifier=True,
        efficientnet_model_path=efficientnet_path,
        efficientnet_classes=["bar", "line", "pie"],
        resnet_confidence_threshold=0.55,
        # Keep ResNet disabled to force EfficientNet path
        use_resnet_classifier=False,
    )

    stage3 = Stage3Extraction(config)
    output = stage3.process(stage2_output)
    logger.info(
        f"Stage 3 done | charts_extracted={len(output.metadata)}"
    )
    return output


# ---------------------------------------------------------------------------
# Stage 4: Reasoning
# ---------------------------------------------------------------------------

def run_stage4(
    stage3_output: "Stage3Output",
    use_llm: bool = True,
) -> "Stage4Output":
    """Run Stage 4 reasoning via AI Router (Gemini → OpenAI fallback)."""
    from core_engine.stages.s4_reasoning import Stage4Reasoning, ReasoningConfig

    engine = "router" if use_llm else "rule_based"
    config = ReasoningConfig(engine=engine)

    stage4 = Stage4Reasoning(config)
    output = stage4.process(stage3_output)
    logger.info(
        f"Stage 4 done | engine={engine} | charts_reasoned={len(output.charts)}"
    )
    return output


# ---------------------------------------------------------------------------
# Stage 5: Reporting
# ---------------------------------------------------------------------------

def run_stage5(
    stage4_output: "Stage4Output",
    output_dir: Optional[Path] = None,
) -> "PipelineResult":
    """Run Stage 5 reporting and save output."""
    from core_engine.stages.s5_reporting import Stage5Reporting, ReportingConfig

    config = ReportingConfig(
        output_dir=str(output_dir) if output_dir else "data/output",
        save_json=True,
        save_report=True,
        save_markdown=True,
    )
    stage5 = Stage5Reporting(config)
    result = stage5.process(stage4_output)
    logger.info("Stage 5 done | report generated")
    return result


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_stage2(stage2_out) -> None:
    _sub("Stage 2 — Detection")
    print(f"  Charts detected : {stage2_out.total_detected}")
    for c in stage2_out.charts:
        bb = c.bbox
        print(
            f"    [{c.chart_id}]  bbox=({bb.x_min},{bb.y_min},{bb.x_max},{bb.y_max})"
            f"  crop={c.cropped_path.name}"
        )


def _print_stage3(stage3_out) -> None:
    _sub("Stage 3 — Extraction  (EfficientNet-B0 3-class, 97.54% acc)")
    for m in stage3_out.metadata:
        conf = m.confidence
        texts_preview = [t.text for t in m.texts[:5]]
        print(f"  [{m.chart_id}]")
        print(f"    chart_type      : {m.chart_type.value if hasattr(m.chart_type, 'value') else m.chart_type}")
        print(f"    elements        : {len(m.elements)}")
        print(f"    texts_detected  : {len(m.texts)}")
        print(f"    texts_preview   : {texts_preview}")
        overall = conf.overall_confidence if conf else 0.0
        cls_c = conf.classification_confidence if conf else 0.0
        ocr_c = conf.ocr_mean_confidence if conf else 0.0
        print(f"    confidence      : overall={overall:.2f}  cls={cls_c:.2f}  ocr={ocr_c:.2f}")
        if m.warnings:
            for w in m.warnings:
                print(f"    WARNING         : {w}")


def _print_stage4(stage4_out) -> None:
    _sub("Stage 4 — Reasoning  (AIRouter → Gemini / OpenAI)")
    for c in stage4_out.charts:
        print(f"  [{c.chart_id}]")
        print(f"    title     : {c.title or '(none)'}")
        print(f"    type      : {c.chart_type.value if hasattr(c.chart_type,'value') else c.chart_type}")
        print(f"    x_label   : {c.x_axis_label or '(none)'}")
        print(f"    y_label   : {c.y_axis_label or '(none)'}")
        for s in c.series[:3]:
            pts = [(p.label, p.value) for p in s.points[:5]]
            print(f"    series    : {s.name or 'series'} | {pts}")
        if c.description:
            desc_preview = c.description[:120].replace("\n", " ")
            print(f"    desc      : {desc_preview}...")


def _print_stage5(result) -> None:
    _sub("Stage 5 — Report")
    print(f"  Summary : {result.summary or '(no summary)'}")
    print(f"  Charts  : {len(result.charts)}")
    for c in result.charts:
        insights = c.insights or []
        print(f"    [{c.chart_id}]  insights={len(insights)}")
        for ins in insights[:3]:
            print(f"      - {ins.text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline demo S1→S5")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single custom image path. Overrides sample set.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory of images to process. Overrides sample set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (useful with --dir).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Stage 4 LLM call (use rule_based engine). Useful for offline testing.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve input images
    # -----------------------------------------------------------------------
    samples_dir = PROJECT_ROOT / "data" / "samples"
    if args.image:
        image_paths = [args.image.resolve()]
    elif args.dir:
        img_dir = args.dir.resolve()
        image_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        if args.limit:
            image_paths = image_paths[: args.limit]
    else:
        image_paths = sorted(samples_dir.glob("*.png")) + sorted(samples_dir.glob("*.jpg"))

    if not image_paths:
        print(f"[ERROR] No images found in {samples_dir}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Resolve model paths
    # -----------------------------------------------------------------------
    weights_dir = PROJECT_ROOT / "models" / "weights"
    yolo_model = weights_dir / "yolo_chart_detector.pt"
    effnet_model = weights_dir / "efficientnet_b0_3class_v1_best.pt"

    if not yolo_model.exists():
        # Try alternate YOLO weight names
        for candidate in ("yolov8n.pt", "yolo26n.pt"):
            p = weights_dir / candidate
            if p.exists():
                logger.warning(f"yolo_chart_detector.pt not found; using {candidate}")
                yolo_model = p
                break

    if not effnet_model.exists():
        print(f"[ERROR] EfficientNet weights not found: {effnet_model}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / "data" / "output" / "demo_s1_s5"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "demo_results.json"

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    _section("Geo-SLM Chart Analysis — Full Pipeline Demo  (S1 → S5)")
    print(f"  Images    : {[p.name for p in image_paths]}")
    print(f"  YOLO      : {yolo_model.name}  (exists={yolo_model.exists()})")
    print(f"  Classifier: {effnet_model.name}  (97.54% acc, 3-class)")
    print(f"  LLM       : {'DISABLED (rule_based)' if args.no_llm else 'AIRouter (Gemini → OpenAI)'}")
    print(f"  Output    : {json_output}")

    results_all = {}
    t_total = time.perf_counter()

    # -----------------------------------------------------------------------
    # Stage 1
    # -----------------------------------------------------------------------
    _section("Stage 1 — Ingestion")
    t0 = time.perf_counter()
    stage1_out, session = run_stage1(image_paths)
    print(f"  Images loaded : {stage1_out.total_images}")
    for ci in stage1_out.images:
        print(f"    [{ci.page_number}] {ci.image_path.name}  ({ci.width}x{ci.height})")
    print(f"  Session ID    : {session.session_id}")
    print(f"  Elapsed       : {time.perf_counter() - t0:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 2
    # -----------------------------------------------------------------------
    _section("Stage 2 — Detection")
    t0 = time.perf_counter()
    stage2_out = run_stage2(stage1_out, session, yolo_model, output_dir)
    _print_stage2(stage2_out)
    print(f"  Elapsed : {time.perf_counter() - t0:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 3
    # -----------------------------------------------------------------------
    _section("Stage 3 — Extraction")
    t0 = time.perf_counter()
    stage3_out = run_stage3(stage2_out, effnet_model)
    _print_stage3(stage3_out)
    print(f"  Elapsed : {time.perf_counter() - t0:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 4
    # -----------------------------------------------------------------------
    _section("Stage 4 — Reasoning")
    t0 = time.perf_counter()
    stage4_out = run_stage4(stage3_out, use_llm=not args.no_llm)
    _print_stage4(stage4_out)
    print(f"  Elapsed : {time.perf_counter() - t0:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 5
    # -----------------------------------------------------------------------
    _section("Stage 5 — Reporting")
    t0 = time.perf_counter()
    pipeline_result = run_stage5(stage4_out, output_dir=output_dir)
    _print_stage5(pipeline_result)
    print(f"  Elapsed : {time.perf_counter() - t0:.2f}s")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    _section("Summary")
    total_elapsed = time.perf_counter() - t_total
    print(f"  Total images processed : {len(image_paths)}")
    print(f"  Charts detected        : {stage2_out.total_detected}")
    print(f"  Charts extracted       : {len(stage3_out.metadata)}")  
    print(f"  Charts reasoned        : {len(stage4_out.charts)}")
    print(f"  Pipeline elapsed       : {total_elapsed:.2f}s")
    print()

    # Per-chart quick view
    col_w = [24, 12, 8, 8, 8]
    header = (
        f"{'Chart ID':<{col_w[0]}}"
        f"{'Type':<{col_w[1]}}"
        f"{'Texts':>{col_w[2]}}"
        f"{'Elems':>{col_w[3]}}"
        f"{'Conf':>{col_w[4]}}"
    )
    print(header)
    print(_hr("-", sum(col_w) + 4))
    for c4 in stage4_out.charts:
        # Find matching Stage3 metadata
        s3_match = next(
            (m for m in stage3_out.metadata if m.chart_id == c4.chart_id),
            None,
        )
        n_texts = len(s3_match.texts) if s3_match else "?"
        n_elems = len(s3_match.elements) if s3_match else "?"
        conf = (
            f"{s3_match.confidence.overall_confidence:.2f}"
            if s3_match and s3_match.confidence
            else "?"
        )
        ctype = (
            c4.chart_type.value
            if hasattr(c4.chart_type, "value")
            else str(c4.chart_type)
        )
        print(
            f"{c4.chart_id:<{col_w[0]}}"
            f"{ctype:<{col_w[1]}}"
            f"{str(n_texts):>{col_w[2]}}"
            f"{str(n_elems):>{col_w[3]}}"
            f"{conf:>{col_w[4]}}"
        )

    print()
    print(f"  Full JSON → {json_output}")
    print(_hr("="))


if __name__ == "__main__":
    main()
