"""
Stage 3 Weakness Audit Script

Systematically tests the extraction pipeline on real chart images
to identify critical weaknesses and failure modes.

Produces a detailed JSON + Markdown report with:
- Per-chart-type success/failure rates
- Specific failure categories (OCR, element detection, axis calibration, etc.)
- Confidence distributions
- Actionable weakness analysis

Usage:
    .venv/Scripts/python.exe scripts/evaluation/stage3_weakness_audit.py
"""

import json
import logging
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from core_engine.stages.s3_extraction.s3_extraction import (
    ExtractionConfig,
    Stage3Extraction,
)
from core_engine.schemas.enums import ChartType, ElementType, TextRole

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("stage3_audit")
logger.setLevel(logging.INFO)

# Suppress noisy sub-module logs
for name in [
    "Stage3Extraction", "ImagePreprocessor", "OCREngine",
    "ElementDetector", "Skeletonizer", "Vectorizer",
    "GeometricMapper", "ChartClassifier", "ResNet18Classifier",
    "MLChartClassifier",
]:
    logging.getLogger(name).setLevel(logging.ERROR)


# ============================================================
# Data structures
# ============================================================

@dataclass
class ChartAuditResult:
    """Result of auditing a single chart image."""
    file_name: str
    true_type: str
    predicted_type: str
    classification_correct: bool
    classification_confidence: float
    ocr_count: int
    ocr_mean_confidence: float
    ocr_roles_found: List[str]
    element_count: int
    element_types_found: List[str]
    bars_found: int
    markers_found: int
    slices_found: int
    axis_detected: bool
    x_axis_detected: bool
    y_axis_detected: bool
    axis_calibration_confidence: float
    overall_confidence: float
    warnings: List[str]
    error: Optional[str] = None
    processing_time_ms: float = 0.0

    # Failure flags
    classification_failed: bool = False
    ocr_failed: bool = False
    no_elements: bool = False
    axis_failed: bool = False
    low_confidence: bool = False


@dataclass
class TypeSummary:
    """Summary statistics for a chart type."""
    chart_type: str
    total: int = 0
    classification_correct: int = 0
    classification_accuracy: float = 0.0
    ocr_success: int = 0
    ocr_failure: int = 0
    ocr_mean_confidence: float = 0.0
    element_success: int = 0
    element_failure: int = 0
    axis_success: int = 0
    axis_failure: int = 0
    low_confidence_count: int = 0
    error_count: int = 0
    avg_processing_time_ms: float = 0.0
    avg_overall_confidence: float = 0.0
    
    # Specific failure categories
    no_text_detected: int = 0
    no_title_detected: int = 0
    no_tick_labels: int = 0
    no_elements_detected: int = 0
    type_mismatch_details: List[str] = field(default_factory=list)
    common_warnings: Dict[str, int] = field(default_factory=dict)


# ============================================================
# Audit logic
# ============================================================

SAMPLES_PER_TYPE = 10  # Number of charts to test per type (reduced for faster iteration)
CHART_TYPES = ["pie", "bar", "line", "scatter", "area", "histogram", "heatmap", "box"]
DATA_DIR = project_root / "data" / "academic_dataset" / "classified_charts"
OUTPUT_DIR = project_root / "docs" / "reports"


def sample_charts(chart_type: str, n: int = SAMPLES_PER_TYPE) -> List[Path]:
    """Randomly sample chart images of a given type."""
    type_dir = DATA_DIR / chart_type
    if not type_dir.exists():
        return []
    
    files = [
        type_dir / f
        for f in os.listdir(type_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    
    if len(files) <= n:
        return files
    
    random.seed(42)  # Reproducible sampling
    return random.sample(files, n)


def audit_single_chart(
    stage3: Stage3Extraction,
    image_path: Path,
    true_type: str,
) -> ChartAuditResult:
    """Audit a single chart image through Stage 3."""
    start_time = time.time()
    
    try:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return ChartAuditResult(
                file_name=image_path.name,
                true_type=true_type,
                predicted_type="ERROR",
                classification_correct=False,
                classification_confidence=0.0,
                ocr_count=0,
                ocr_mean_confidence=0.0,
                ocr_roles_found=[],
                element_count=0,
                element_types_found=[],
                bars_found=0,
                markers_found=0,
                slices_found=0,
                axis_detected=False,
                x_axis_detected=False,
                y_axis_detected=False,
                axis_calibration_confidence=0.0,
                overall_confidence=0.0,
                warnings=[],
                error="Failed to load image",
                classification_failed=True,
            )
        
        # Run extraction
        metadata = stage3.process_image(
            image_bgr,
            chart_id=image_path.stem,
            image_path=image_path,
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        # Analyze results
        predicted_type = metadata.chart_type.value if metadata.chart_type else "unknown"
        classification_correct = predicted_type == true_type
        
        # OCR analysis
        ocr_count = len(metadata.texts)
        ocr_mean_conf = (
            sum(t.confidence for t in metadata.texts) / ocr_count
            if ocr_count > 0 else 0.0
        )
        ocr_roles = list(set(
            t.role for t in metadata.texts if t.role
        ))
        
        # Element analysis
        element_count = len(metadata.elements)
        element_types = list(set(
            e.element_type for e in metadata.elements
        ))
        bars_found = sum(1 for e in metadata.elements if e.element_type == ElementType.BAR.value)
        markers_found = sum(1 for e in metadata.elements if e.element_type == ElementType.POINT.value)
        slices_found = sum(1 for e in metadata.elements if e.element_type == ElementType.SLICE.value)
        
        # Axis analysis
        axis_detected = metadata.axis_info is not None
        x_axis = metadata.axis_info.x_axis_detected if metadata.axis_info else False
        y_axis = metadata.axis_info.y_axis_detected if metadata.axis_info else False
        axis_conf = 0.0
        if metadata.axis_info:
            confs = []
            if metadata.axis_info.x_calibration_confidence > 0:
                confs.append(metadata.axis_info.x_calibration_confidence)
            if metadata.axis_info.y_calibration_confidence > 0:
                confs.append(metadata.axis_info.y_calibration_confidence)
            axis_conf = sum(confs) / len(confs) if confs else 0.0
        
        # Confidence
        overall_conf = metadata.confidence.overall_confidence if metadata.confidence else 0.0
        class_conf = metadata.confidence.classification_confidence if metadata.confidence else 0.0
        
        # Warnings
        warnings = metadata.warnings if metadata.warnings else []
        
        # Failure flags
        has_title = any(t.role == TextRole.TITLE.value for t in metadata.texts)
        has_ticks = any(
            t.role in (TextRole.X_TICK.value, TextRole.Y_TICK.value)
            for t in metadata.texts
        )
        
        result = ChartAuditResult(
            file_name=image_path.name,
            true_type=true_type,
            predicted_type=predicted_type,
            classification_correct=classification_correct,
            classification_confidence=class_conf,
            ocr_count=ocr_count,
            ocr_mean_confidence=ocr_mean_conf,
            ocr_roles_found=ocr_roles,
            element_count=element_count,
            element_types_found=element_types,
            bars_found=bars_found,
            markers_found=markers_found,
            slices_found=slices_found,
            axis_detected=axis_detected,
            x_axis_detected=x_axis,
            y_axis_detected=y_axis,
            axis_calibration_confidence=axis_conf,
            overall_confidence=overall_conf,
            warnings=warnings,
            processing_time_ms=elapsed,
        )
        
        # Set failure flags
        result.classification_failed = not classification_correct
        result.ocr_failed = ocr_count == 0
        result.no_elements = element_count == 0
        result.axis_failed = not (x_axis or y_axis)
        result.low_confidence = overall_conf < 0.5
        
        return result
        
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return ChartAuditResult(
            file_name=image_path.name,
            true_type=true_type,
            predicted_type="ERROR",
            classification_correct=False,
            classification_confidence=0.0,
            ocr_count=0,
            ocr_mean_confidence=0.0,
            ocr_roles_found=[],
            element_count=0,
            element_types_found=[],
            bars_found=0,
            markers_found=0,
            slices_found=0,
            axis_detected=False,
            x_axis_detected=False,
            y_axis_detected=False,
            axis_calibration_confidence=0.0,
            overall_confidence=0.0,
            warnings=[],
            error=f"{type(e).__name__}: {str(e)[:200]}",
            processing_time_ms=elapsed,
            classification_failed=True,
        )


def compute_type_summary(results: List[ChartAuditResult], chart_type: str) -> TypeSummary:
    """Compute summary statistics for a chart type."""
    summary = TypeSummary(chart_type=chart_type)
    summary.total = len(results)
    
    if not results:
        return summary
    
    processing_times = []
    overall_confs = []
    ocr_confs = []
    
    for r in results:
        if r.error:
            summary.error_count += 1
            continue
        
        processing_times.append(r.processing_time_ms)
        overall_confs.append(r.overall_confidence)
        
        # Classification
        if r.classification_correct:
            summary.classification_correct += 1
        else:
            summary.type_mismatch_details.append(
                f"{r.file_name}: predicted={r.predicted_type}"
            )
        
        # OCR
        if r.ocr_count > 0:
            summary.ocr_success += 1
            ocr_confs.append(r.ocr_mean_confidence)
        else:
            summary.ocr_failure += 1
            summary.no_text_detected += 1
        
        # Title
        if TextRole.TITLE.value not in r.ocr_roles_found:
            summary.no_title_detected += 1
        
        # Tick labels
        if (TextRole.X_TICK.value not in r.ocr_roles_found and 
            TextRole.Y_TICK.value not in r.ocr_roles_found):
            summary.no_tick_labels += 1
        
        # Elements
        if r.element_count > 0:
            summary.element_success += 1
        else:
            summary.element_failure += 1
            summary.no_elements_detected += 1
        
        # Axis
        if r.x_axis_detected or r.y_axis_detected:
            summary.axis_success += 1
        else:
            summary.axis_failure += 1
        
        # Confidence
        if r.overall_confidence < 0.5:
            summary.low_confidence_count += 1
        
        # Warnings
        for w in r.warnings:
            # Normalize warning text
            key = w.split(":")[0] if ":" in w else w
            summary.common_warnings[key] = summary.common_warnings.get(key, 0) + 1
    
    # Compute averages
    valid = summary.total - summary.error_count
    if valid > 0:
        summary.classification_accuracy = summary.classification_correct / valid
        summary.avg_processing_time_ms = sum(processing_times) / len(processing_times) if processing_times else 0
        summary.avg_overall_confidence = sum(overall_confs) / len(overall_confs) if overall_confs else 0
        summary.ocr_mean_confidence = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0
    
    return summary


def generate_markdown_report(
    summaries: List[TypeSummary],
    all_results: Dict[str, List[ChartAuditResult]],
    total_time: float,
) -> str:
    """Generate a Markdown report from audit results."""
    lines = []
    lines.append("# Stage 3 Extraction -- Weakness Audit Report")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Samples per type:** {SAMPLES_PER_TYPE}")
    lines.append(f"**Total processing time:** {total_time:.1f}s")
    lines.append("")
    
    # Overview table
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Chart Type | Samples | Class. Acc | OCR Success | Elem. Found | Axis Cal. | Avg Conf | Avg Time (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    
    for s in summaries:
        valid = s.total - s.error_count
        if valid > 0:
            row = (
                f"| {s.chart_type} | {s.total} | "
                f"{s.classification_accuracy:.0%} | "
                f"{s.ocr_success}/{valid} ({s.ocr_success/valid:.0%}) | "
                f"{s.element_success}/{valid} ({s.element_success/valid:.0%}) | "
                f"{s.axis_success}/{valid} ({s.axis_success/valid:.0%}) | "
                f"{s.avg_overall_confidence:.2f} | "
                f"{s.avg_processing_time_ms:.0f} |"
            )
        else:
            row = f"| {s.chart_type} | {s.total} | N/A | N/A | N/A | N/A | N/A | N/A |"
        lines.append(row)
    
    # Critical weaknesses
    lines.append("")
    lines.append("## 2. Critical Weaknesses Identified")
    lines.append("")
    
    weakness_num = 1
    
    for s in summaries:
        valid = s.total - s.error_count
        if valid == 0:
            continue
        
        type_weaknesses = []
        
        # Classification failures
        if s.classification_accuracy < 0.8:
            type_weaknesses.append(
                f"**Classification weakness**: Only {s.classification_accuracy:.0%} accuracy. "
                f"Misclassified as: {', '.join(s.type_mismatch_details[:5])}"
            )
        
        # OCR failures
        ocr_fail_rate = s.ocr_failure / valid if valid > 0 else 0
        if ocr_fail_rate > 0.2:
            type_weaknesses.append(
                f"**OCR failure**: {s.ocr_failure}/{valid} ({ocr_fail_rate:.0%}) charts had ZERO text detected."
            )
        
        if s.ocr_mean_confidence < 0.7 and s.ocr_success > 0:
            type_weaknesses.append(
                f"**Low OCR confidence**: Mean OCR confidence = {s.ocr_mean_confidence:.2f} (below 0.70 threshold)."
            )
        
        # No title/ticks
        no_title_rate = s.no_title_detected / valid if valid > 0 else 0
        if no_title_rate > 0.4:
            type_weaknesses.append(
                f"**Title detection failure**: {s.no_title_detected}/{valid} ({no_title_rate:.0%}) charts had no title detected."
            )
        
        no_tick_rate = s.no_tick_labels / valid if valid > 0 else 0
        if no_tick_rate > 0.3:
            type_weaknesses.append(
                f"**Tick label failure**: {s.no_tick_labels}/{valid} ({no_tick_rate:.0%}) charts had no tick labels detected."
            )
        
        # Element detection
        elem_fail_rate = s.element_failure / valid if valid > 0 else 0
        if elem_fail_rate > 0.3:
            type_weaknesses.append(
                f"**Element detection failure**: {s.element_failure}/{valid} ({elem_fail_rate:.0%}) charts had ZERO elements detected."
            )
        
        # Axis calibration
        axis_fail_rate = s.axis_failure / valid if valid > 0 else 0
        if axis_fail_rate > 0.3:
            type_weaknesses.append(
                f"**Axis calibration failure**: {s.axis_failure}/{valid} ({axis_fail_rate:.0%}) charts failed axis calibration."
            )
        
        # Low confidence
        low_conf_rate = s.low_confidence_count / valid if valid > 0 else 0
        if low_conf_rate > 0.3:
            type_weaknesses.append(
                f"**Low overall confidence**: {s.low_confidence_count}/{valid} ({low_conf_rate:.0%}) charts had confidence < 0.50."
            )
        
        if type_weaknesses:
            lines.append(f"### 2.{weakness_num}. {s.chart_type.upper()} Charts")
            lines.append("")
            for tw in type_weaknesses:
                lines.append(f"- {tw}")
            lines.append("")
            weakness_num += 1
    
    # Specific failure analysis per type
    lines.append("## 3. Detailed Failure Analysis")
    lines.append("")
    
    for s in summaries:
        valid = s.total - s.error_count
        if valid == 0:
            continue
        
        lines.append(f"### {s.chart_type.upper()}")
        lines.append("")
        lines.append(f"- **Classification accuracy:** {s.classification_accuracy:.0%}")
        lines.append(f"- **OCR success rate:** {s.ocr_success}/{valid}")
        lines.append(f"- **OCR mean confidence:** {s.ocr_mean_confidence:.2f}")
        lines.append(f"- **Element detection success:** {s.element_success}/{valid}")
        lines.append(f"- **Axis calibration success:** {s.axis_success}/{valid}")
        lines.append(f"- **Average overall confidence:** {s.avg_overall_confidence:.2f}")
        lines.append(f"- **No title detected:** {s.no_title_detected}/{valid}")
        lines.append(f"- **No tick labels:** {s.no_tick_labels}/{valid}")
        lines.append(f"- **Errors:** {s.error_count}")
        
        if s.type_mismatch_details:
            lines.append(f"- **Misclassifications:** {', '.join(s.type_mismatch_details[:5])}")
        
        if s.common_warnings:
            lines.append("- **Common warnings:**")
            for w, count in sorted(s.common_warnings.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {w}: {count}x")
        
        lines.append("")
    
    # OCR Role Classification analysis
    lines.append("## 4. OCR Spatial Role Classification Analysis")
    lines.append("")
    lines.append("The current heuristic-based role classification uses hardcoded position thresholds:")
    lines.append("")
    lines.append("| Role | Rule | Known Weakness |")
    lines.append("|---|---|---|")
    lines.append("| Title | Top 15% + centered + long text | Fails when title is absent or very short |")
    lines.append("| Y-Axis Label | Left 15% + not numeric | Fails for right-side Y-axis (dual-axis charts) |")
    lines.append("| Y-Tick | Left 25% + numeric | Mis-classifies data labels near left edge |")
    lines.append("| X-Axis Label | Bottom 15% + centered | Fails when xlabel is above legend at bottom |")
    lines.append("| X-Tick | Bottom 25% + numeric/short | Overlaps with legend region |")
    lines.append("| Legend | Right 35% + short text | Fails for bottom-positioned legends |")
    lines.append("| Data Label | Inner region + numeric | Catches too many false positives |")
    lines.append("")
    
    # Pie chart specific analysis
    lines.append("## 5. Pie Chart Extraction Analysis")
    lines.append("")
    lines.append("The current pie chart detection algorithm relies on:")
    lines.append("")
    lines.append("1. **K-Means color clustering** to identify slice regions")
    lines.append("2. **Contour analysis** on color masks to find slice boundaries")
    lines.append("3. **Centroid-based angle calculation** for proportions")
    lines.append("")
    lines.append("**Known failure modes:**")
    lines.append("")
    lines.append("- Exploded pie charts: Slices are spatially separated, breaking contour connectivity")
    lines.append("- Donut charts: `radius_inner=0.0` is hardcoded, never detecting the hole")
    lines.append("- 3D perspective pies: Elliptical projection breaks circular assumptions")
    lines.append("- Monochrome pies: K-Means cannot separate slices without color contrast")
    lines.append("- Overlapping legends: Color patches classified as slices")
    lines.append("")
    
    # Geometric mapper specific analysis
    lines.append("## 6. Geometric Mapping (Pixel-to-Value) Analysis")
    lines.append("")
    lines.append("The calibration pipeline is:")
    lines.append("")
    lines.append("```")
    lines.append("OCR tick values -> (pixel_position, numeric_value) pairs -> RANSAC/Theil-Sen fit -> slope + intercept")
    lines.append("```")
    lines.append("")
    lines.append("**Critical dependency chain:**")
    lines.append("")
    lines.append("1. OCR must correctly **read** tick label text (error if '100' -> '10')")
    lines.append("2. OCR must correctly **classify** text as tick label vs data label")
    lines.append("3. Tick positions must be accurately mapped to pixel centers")
    lines.append("4. Linear regression must have minimum 2 valid points")
    lines.append("")
    lines.append("**Single-point-of-failure:** If OCR misreads ONE tick label, the linear")
    lines.append("calibration slope changes drastically, making ALL extracted values wrong.")
    lines.append("RANSAC helps but requires >= 3 points to be effective.")
    lines.append("")
    
    # Recommendations
    lines.append("## 7. Prioritized Recommendations")
    lines.append("")
    lines.append("### Priority 1: Instance Segmentation for Pie Chart Slices")
    lines.append("")
    lines.append("Replace K-Means color clustering with YOLOv8-seg trained on pie_slice class.")
    lines.append("This would solve: exploded pie, donut, 3D perspective, monochrome issues.")
    lines.append("**Impact:** High | **Feasibility:** Medium (requires labeled data)")
    lines.append("")
    lines.append("### Priority 2: Line Removal Before OCR")
    lines.append("")
    lines.append("Subtract skeleton mask from image before running PaddleOCR.")
    lines.append("Grid lines cutting through digits cause systematic OCR errors ('8' -> '3').")
    lines.append("**Impact:** Medium | **Feasibility:** High (algorithm exists in preprocessor)")
    lines.append("")
    lines.append("### Priority 3: ML-Based Role Classification")
    lines.append("")
    lines.append("Replace hardcoded position % thresholds with a small classifier (YOLOv8-nano OBB).")
    lines.append("Train on: Title, X-Label, Y-Label, Legend, Data-Label bounding boxes.")
    lines.append("**Impact:** High | **Feasibility:** Medium (requires annotation)")
    lines.append("")
    lines.append("### Priority 4: Confidence-Aware AI Routing")
    lines.append("")
    lines.append("Use ExtractionConfidence scores to control Stage 4 behavior.")
    lines.append("When confidence < threshold, prompt should instruct LLM to re-analyze image.")
    lines.append("**Impact:** Medium | **Feasibility:** High (AIRouter already exists)")
    lines.append("")
    
    return "\n".join(lines)


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    """Run the Stage 3 weakness audit."""
    logger.info("=" * 60)
    logger.info("Stage 3 Extraction -- Weakness Audit")
    logger.info("=" * 60)
    
    # Initialize Stage 3 with default config (OCR disabled for speed, use easyocr)
    config = ExtractionConfig(
        ocr_engine="easyocr",
        enable_vectorization=True,
        enable_element_detection=True,
        enable_ocr=True,
        enable_classification=True,
        use_resnet_classifier=True,
        use_ml_classifier=True,
        use_color_segmentation=True,
    )
    
    logger.info("Initializing Stage3Extraction...")
    stage3 = Stage3Extraction(config)
    logger.info("Stage3Extraction initialized")
    
    all_results: Dict[str, List[ChartAuditResult]] = {}
    summaries: List[TypeSummary] = []
    
    total_start = time.time()
    
    for chart_type in CHART_TYPES:
        logger.info(f"--- Auditing: {chart_type} ---")
        
        images = sample_charts(chart_type, SAMPLES_PER_TYPE)
        if not images:
            logger.warning(f"No images found for {chart_type}")
            continue
        
        logger.info(f"Sampled {len(images)} images for {chart_type}")
        
        results = []
        for i, img_path in enumerate(images):
            result = audit_single_chart(stage3, img_path, chart_type)
            results.append(result)
            
            status = "OK" if not result.error else f"ERR: {result.error[:50]}"
            conf = f"conf={result.overall_confidence:.2f}" if not result.error else ""
            cls = f"cls={result.predicted_type}" if not result.error else ""
            logger.info(
                f"  [{i+1}/{len(images)}] {img_path.name[:40]:<40} | "
                f"{status} | {cls} | {conf}"
            )
        
        all_results[chart_type] = results
        summary = compute_type_summary(results, chart_type)
        summaries.append(summary)
        
        logger.info(
            f"  Summary: acc={summary.classification_accuracy:.0%} | "
            f"ocr={summary.ocr_success}/{summary.total} | "
            f"elem={summary.element_success}/{summary.total} | "
            f"axis={summary.axis_success}/{summary.total} | "
            f"conf={summary.avg_overall_confidence:.2f}"
        )
    
    total_time = time.time() - total_start
    
    # Generate reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    json_report = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "samples_per_type": SAMPLES_PER_TYPE,
            "total_time_seconds": round(total_time, 1),
            "chart_types": CHART_TYPES,
        },
        "summaries": {},
        "detailed_results": {},
    }
    
    for s in summaries:
        json_report["summaries"][s.chart_type] = {
            "total": s.total,
            "classification_accuracy": round(s.classification_accuracy, 3),
            "ocr_success_rate": round(s.ocr_success / max(s.total - s.error_count, 1), 3),
            "ocr_mean_confidence": round(s.ocr_mean_confidence, 3),
            "element_success_rate": round(s.element_success / max(s.total - s.error_count, 1), 3),
            "axis_success_rate": round(s.axis_success / max(s.total - s.error_count, 1), 3),
            "avg_overall_confidence": round(s.avg_overall_confidence, 3),
            "avg_processing_time_ms": round(s.avg_processing_time_ms, 1),
            "errors": s.error_count,
            "no_text_detected": s.no_text_detected,
            "no_title_detected": s.no_title_detected,
            "no_tick_labels": s.no_tick_labels,
            "no_elements_detected": s.no_elements_detected,
            "low_confidence_count": s.low_confidence_count,
            "misclassifications": s.type_mismatch_details[:10],
            "common_warnings": dict(sorted(s.common_warnings.items(), key=lambda x: -x[1])[:10]),
        }
    
    for chart_type, results in all_results.items():
        json_report["detailed_results"][chart_type] = [
            {
                "file": r.file_name,
                "true_type": r.true_type,
                "predicted_type": r.predicted_type,
                "classification_correct": r.classification_correct,
                "classification_confidence": round(r.classification_confidence, 3),
                "ocr_count": r.ocr_count,
                "ocr_mean_confidence": round(r.ocr_mean_confidence, 3),
                "ocr_roles": r.ocr_roles_found,
                "element_count": r.element_count,
                "element_types": r.element_types_found,
                "bars": r.bars_found,
                "markers": r.markers_found,
                "slices": r.slices_found,
                "axis_detected": r.axis_detected,
                "x_axis": r.x_axis_detected,
                "y_axis": r.y_axis_detected,
                "axis_confidence": round(r.axis_calibration_confidence, 3),
                "overall_confidence": round(r.overall_confidence, 3),
                "warnings": r.warnings,
                "error": r.error,
                "time_ms": round(r.processing_time_ms, 1),
            }
            for r in results
        ]
    
    json_path = OUTPUT_DIR / "stage3_weakness_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON report saved: {json_path}")
    
    # Markdown report
    md_report = generate_markdown_report(summaries, all_results, total_time)
    md_path = OUTPUT_DIR / "stage3_weakness_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"Markdown report saved: {md_path}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("STAGE 3 WEAKNESS AUDIT -- SUMMARY")
    print("=" * 70)
    print(f"{'Type':<12} {'Acc':<8} {'OCR':<8} {'Elem':<8} {'Axis':<8} {'Conf':<8} {'Time':<10}")
    print("-" * 70)
    
    for s in summaries:
        valid = s.total - s.error_count
        if valid == 0:
            continue
        print(
            f"{s.chart_type:<12} "
            f"{s.classification_accuracy:>5.0%}   "
            f"{s.ocr_success:>2}/{valid:<2}   "
            f"{s.element_success:>2}/{valid:<2}   "
            f"{s.axis_success:>2}/{valid:<2}   "
            f"{s.avg_overall_confidence:>5.2f}   "
            f"{s.avg_processing_time_ms:>6.0f}ms"
        )
    
    print("-" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Reports: {json_path}")
    print(f"         {md_path}")


if __name__ == "__main__":
    main()
