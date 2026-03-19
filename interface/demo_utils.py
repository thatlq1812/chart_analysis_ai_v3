"""
Demo utilities for Gradio chart analysis app.

Helper functions that wrap the pipeline for interactive demo use.
These should NOT contain business logic -- only format/display adapters.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)


def run_pipeline_for_demo(
    image_path: str,
    backend: str = "deplot",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run chart analysis pipeline on a single image for demo display.

    Args:
        image_path: Path to chart image file.
        backend: VLM extractor backend (deplot, matcha, pix2struct).
        device: Compute device (cpu, cuda, auto).

    Returns:
        Dict with keys: json_output, summary, table_html, timing, error
    """
    result = {
        "json_output": "{}",
        "summary": "",
        "table_html": "",
        "timing": {},
        "error": None,
    }

    try:
        import cv2

        t_start = time.time()

        # Stage 3: VLM Extraction
        t_extract = time.time()
        from core_engine.stages.s3_extraction.extractors import create_extractor

        extractor = create_extractor(backend, device=device)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            result["error"] = f"Failed to load image: {image_path}"
            return result

        chart_id = Path(image_path).stem
        pix2struct_result = extractor.extract(image_bgr, chart_id=chart_id)
        t_extract = time.time() - t_extract

        # Stage 3b: Classification
        t_classify = time.time()
        chart_type = _classify_chart(image_bgr, device)
        t_classify = time.time() - t_classify

        # Build output
        output_data = {
            "chart_id": chart_id,
            "chart_type": chart_type,
            "backend": backend,
            "extraction": {
                "headers": pix2struct_result.headers,
                "rows": pix2struct_result.rows,
                "records": pix2struct_result.records,
                "confidence": pix2struct_result.extraction_confidence,
                "model": pix2struct_result.model_name,
            },
        }

        t_total = time.time() - t_start

        result["json_output"] = json.dumps(output_data, indent=2, ensure_ascii=False)
        result["summary"] = _generate_summary(chart_type, pix2struct_result)
        result["table_html"] = _records_to_html(pix2struct_result.headers, pix2struct_result.rows)
        result["timing"] = {
            "extraction_s": round(t_extract, 2),
            "classification_s": round(t_classify, 2),
            "total_s": round(t_total, 2),
        }

    except ImportError as e:
        result["error"] = f"Missing dependency: {e}. Install with: pip install -e '.[dev]'"
    except Exception as e:
        result["error"] = f"Pipeline error: {e}"
        logger.exception(f"Demo pipeline failed | image={image_path}")

    return result


def _classify_chart(image_bgr: Any, device: str) -> str:
    """Classify chart type using EfficientNet or fallback."""
    try:
        from core_engine.stages.s3_extraction.resnet_classifier import EfficientNetClassifier

        classifier = EfficientNetClassifier(device=device)
        chart_type, confidence = classifier.predict_with_confidence(image_bgr)
        return chart_type
    except Exception:
        return "unknown"


def _generate_summary(chart_type: str, result: Any) -> str:
    """Generate human-readable summary from extraction result."""
    n_rows = len(result.rows) if result.rows else 0
    n_cols = len(result.headers) if result.headers else 0
    conf = result.extraction_confidence

    lines = [
        f"Chart Type: {chart_type}",
        f"Extracted Table: {n_rows} rows x {n_cols} columns",
        f"Extraction Confidence: {conf:.0%}",
        f"Model: {result.model_name}",
    ]

    if result.headers:
        lines.append(f"Headers: {', '.join(result.headers[:5])}")

    if conf < 0.3:
        lines.append("\nWARNING: Low confidence extraction. Results may be inaccurate.")

    return "\n".join(lines)


def _records_to_html(headers: List[str], rows: List[List[str]]) -> str:
    """Convert table data to HTML table string."""
    if not headers and not rows:
        return "<p>No data extracted</p>"

    html = ['<table style="border-collapse: collapse; width: 100%;">']
    if headers:
        html.append("<thead><tr>")
        for h in headers:
            html.append(f'<th style="border: 1px solid #ddd; padding: 8px; '
                        f'background: #f2f2f2; text-align: left;">{h}</th>')
        html.append("</tr></thead>")

    html.append("<tbody>")
    for row in rows:
        html.append("<tr>")
        for cell in row:
            html.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{cell}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")

    return "\n".join(html)


def format_json_output(data: Dict) -> str:
    """Pretty-format JSON for display."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def get_sample_images(samples_dir: Optional[Path] = None) -> List[str]:
    """Get list of sample chart images for the gallery."""
    if samples_dir is None:
        samples_dir = PROJECT_ROOT / "data" / "samples"

    if not samples_dir.exists():
        # Try benchmark images as fallback
        samples_dir = PROJECT_ROOT / "data" / "benchmark" / "images"

    if not samples_dir.exists():
        return []

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(str(p) for p in sorted(samples_dir.glob(ext))[:12])
    return images
