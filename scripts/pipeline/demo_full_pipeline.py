#!/usr/bin/env python3
"""
Demo Full Pipeline: Image → Analysis → Report

This script demonstrates the complete chart analysis pipeline:
1. Stage 3: Extract features (OCR, elements, classification)
2. Stage 4: Reasoning (value mapping, description generation)
3. Output: Formatted report with insights

Usage:
    python scripts/pipeline/demo_full_pipeline.py path/to/chart.png
    python scripts/pipeline/demo_full_pipeline.py --random  # Pick random chart from dataset
    python scripts/pipeline/demo_full_pipeline.py --chart-type bar --random

Output:
    Console: Formatted report with chart analysis
    JSON: Structured output (optional with --output-json)
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress verbose warnings
logging.getLogger("ppocr").setLevel(logging.ERROR)


def load_image(image_path: Path):
    """Load image and basic validation."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def run_stage3(image_path: Path, image, use_cache: bool = True) -> dict:
    """Run Stage 3: Extraction."""
    from core_engine.stages.s3_extraction import Stage3Extraction, ExtractionConfig
    
    # Configure for PaddleOCR with cache support
    config = ExtractionConfig(
        ocr_engine="paddleocr",
        use_resnet_classifier=True,  # Use ResNet-18 classifier
        enable_vectorization=True,
        enable_element_detection=True,
        enable_ocr=True,
        enable_classification=True,
    )
    config.ocr.engine = "paddleocr"
    config.ocr.use_gpu = False
    
    # Configure cache
    if use_cache:
        cache_file = PROJECT_ROOT / "data" / "cache" / "ocr_cache.json"
        cache_base_dir = PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts"
        if cache_file.exists():
            config.ocr.use_cache = True
            config.ocr.cache_file = cache_file
            config.ocr.cache_base_dir = cache_base_dir
            logger.info(f"OCR cache enabled: {cache_file}")
    
    stage3 = Stage3Extraction(config)
    
    # Process using process_image method, pass image_path for cache lookup
    chart_id = image_path.stem
    result = stage3.process_image(image, chart_id=chart_id, image_path=image_path)
    
    # Convert RawMetadata to dict for easier handling
    return {
        "chart_id": result.chart_id,
        "chart_type": result.chart_type.value if hasattr(result.chart_type, 'value') else str(result.chart_type),
        "texts": [
            {
                "text": t.text,
                "role": t.role.value if hasattr(t.role, 'value') else str(t.role) if t.role else "unknown",
                "confidence": t.confidence,
                "bbox": {"x_min": t.bbox.x_min, "y_min": t.bbox.y_min, "x_max": t.bbox.x_max, "y_max": t.bbox.y_max} if t.bbox else None,
            }
            for t in result.texts
        ],
        "elements": [
            {
                "type": e.element_type.value if hasattr(e.element_type, 'value') else str(e.element_type),
                "bbox": {"x_min": e.bbox.x_min, "y_min": e.bbox.y_min, "x_max": e.bbox.x_max, "y_max": e.bbox.y_max} if e.bbox else None,
                "color": {"r": e.color.r, "g": e.color.g, "b": e.color.b} if e.color else None,
                "center": {"x": (e.bbox.x_min + e.bbox.x_max) // 2, "y": (e.bbox.y_min + e.bbox.y_max) // 2} if e.bbox else None,
            }
            for e in result.elements
        ],
        "axis_info": result.axis_info.model_dump() if result.axis_info else {},
        "confidence": result.confidence.model_dump() if result.confidence else {},
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
    }


def extract_numeric_values(texts: list) -> dict:
    """
    Extract numeric values from OCR text.
    Returns dict with categorized values.
    """
    import re
    
    values = {
        "percentages": [],  # Values with %
        "integers": [],     # Whole numbers
        "decimals": [],     # Decimal numbers
        "currencies": [],   # Values with $, etc.
        "raw_numbers": [],  # All numeric values
    }
    
    for t in texts:
        text = t.get("text", "").strip()
        bbox = t.get("bbox")
        confidence = t.get("confidence", 0)
        
        # Skip low confidence
        if confidence < 0.5:
            continue
        
        # Percentage pattern: 12.5%, 100%
        pct_match = re.match(r'^([\d,.]+)\s*%$', text)
        if pct_match:
            try:
                val = float(pct_match.group(1).replace(',', ''))
                values["percentages"].append({
                    "value": val,
                    "text": text,
                    "bbox": bbox,
                })
                values["raw_numbers"].append(val)
                continue
            except ValueError:
                pass
        
        # Currency pattern: $1,234 or $1.5M
        currency_match = re.match(r'^[\$£€]([\d,.]+)([KMB])?$', text, re.IGNORECASE)
        if currency_match:
            try:
                val = float(currency_match.group(1).replace(',', ''))
                multiplier = {"k": 1e3, "m": 1e6, "b": 1e9}.get(
                    (currency_match.group(2) or "").lower(), 1
                )
                values["currencies"].append({
                    "value": val * multiplier,
                    "text": text,
                    "bbox": bbox,
                })
                values["raw_numbers"].append(val * multiplier)
                continue
            except ValueError:
                pass
        
        # Integer pattern: 100, 1,234
        int_match = re.match(r'^([\d,]+)$', text)
        if int_match:
            try:
                val = int(int_match.group(1).replace(',', ''))
                values["integers"].append({
                    "value": val,
                    "text": text,
                    "bbox": bbox,
                })
                values["raw_numbers"].append(float(val))
                continue
            except ValueError:
                pass
        
        # Decimal pattern: 0.5, 12.34
        dec_match = re.match(r'^([\d,]*\.\d+)$', text)
        if dec_match:
            try:
                val = float(dec_match.group(1).replace(',', ''))
                values["decimals"].append({
                    "value": val,
                    "text": text,
                    "bbox": bbox,
                })
                values["raw_numbers"].append(val)
                continue
            except ValueError:
                pass
    
    return values


def remap_element_types(elements: list, chart_type: str) -> list:
    """
    Remap detected element types based on chart classification.
    
    Element detector may misclassify elements (e.g., detect bars as points).
    This function corrects based on the known chart type.
    """
    # Define expected element types per chart type
    type_mapping = {
        "bar": "bar",
        "histogram": "bar",
        "line": "point",
        "scatter": "point",
        "area": "point",
        "pie": "slice",
        "box": "box",
        "heatmap": "cell",
    }
    
    expected_type = type_mapping.get(chart_type, None)
    
    if not expected_type:
        return elements
    
    remapped = []
    for elem in elements:
        elem_copy = elem.copy()
        original_type = elem.get("type", "unknown")
        
        # For pie charts, remap bars to slices
        if chart_type == "pie" and original_type in ["bar", "point"]:
            elem_copy["type"] = "slice"
            elem_copy["original_type"] = original_type
        
        # For line/scatter, remap bars to points
        elif chart_type in ["line", "scatter", "area"] and original_type == "bar":
            elem_copy["type"] = "point"
            elem_copy["original_type"] = original_type
        
        # For bar/histogram, keep bars
        elif chart_type in ["bar", "histogram"]:
            if original_type == "point":
                # Small rectangles might be detected as points
                elem_copy["type"] = "bar"
                elem_copy["original_type"] = original_type
        
        remapped.append(elem_copy)
    
    return remapped


def parse_numeric_text(text: str) -> Optional[float]:
    """
    Parse numeric value from text with various formats.
    
    Handles: 100, 1,234, 1.5, -50, 1.5K, 2M, 10%, $100, etc.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Remove common prefixes
    text = text.lstrip('$£€¥')
    
    # Check for percentage (but keep the number)
    is_percentage = text.endswith('%')
    if is_percentage:
        text = text[:-1].strip()
    
    # Handle K/M/B suffixes
    multiplier = 1.0
    if text and text[-1].upper() in 'KMB':
        suffix = text[-1].upper()
        multiplier = {'K': 1e3, 'M': 1e6, 'B': 1e9}[suffix]
        text = text[:-1].strip()
    
    # Remove commas (thousand separators)
    text = text.replace(',', '')
    
    # Try to parse
    try:
        value = float(text)
        return value * multiplier
    except ValueError:
        return None


def calibrate_y_axis(texts: list, elements: list, image_size: dict) -> dict:
    """
    Calibrate Y-axis from OCR tick labels.
    
    Returns calibration info for pixel-to-value mapping.
    """
    image_height = image_size.get("height", 400)
    image_width = image_size.get("width", 600)
    
    # Find Y-axis tick labels (numbers on left side of image)
    y_ticks = []
    
    for t in texts:
        bbox = t.get("bbox")
        if not bbox:
            continue
        
        # Y-axis ticks: on left 20% of image
        center_x = (bbox["x_min"] + bbox["x_max"]) / 2
        if center_x > image_width * 0.25:
            continue
        
        # Try to parse as number
        text = t.get("text", "").strip()
        value = parse_numeric_text(text)
        
        if value is not None:
            center_y = (bbox["y_min"] + bbox["y_max"]) / 2
            y_ticks.append({
                "pixel_y": center_y,
                "value": value,
                "text": t.get("text"),
            })
    
    if len(y_ticks) < 2:
        return {"calibrated": False, "reason": "insufficient_ticks"}
    
    # Sort by pixel position (top to bottom)
    y_ticks.sort(key=lambda t: t["pixel_y"])
    
    # Get min/max
    pixels = [t["pixel_y"] for t in y_ticks]
    values = [t["value"] for t in y_ticks]
    
    pixel_min, pixel_max = min(pixels), max(pixels)
    value_min, value_max = min(values), max(values)
    
    # Calculate scale factor (pixels per value unit)
    # Note: In image coordinates, Y increases downward but values increase upward
    if abs(value_max - value_min) < 1e-10:
        return {"calibrated": False, "reason": "zero_range"}
    
    # Pixels per value unit (negative because Y is inverted)
    scale_factor = (pixel_max - pixel_min) / (value_max - value_min)
    
    return {
        "calibrated": True,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "value_min": value_min,
        "value_max": value_max,
        "scale_factor": abs(scale_factor),
        "y_inverted": True,  # Image Y is inverted
        "ticks_found": len(y_ticks),
        "tick_values": values,
    }


def calibrate_x_axis(texts: list, elements: list, image_size: dict) -> dict:
    """
    Calibrate X-axis from OCR tick labels.
    
    Returns calibration info for pixel-to-value mapping.
    Also handles categorical X-axis labels.
    """
    image_height = image_size.get("height", 400)
    image_width = image_size.get("width", 600)
    
    # Find X-axis tick labels (text on bottom of image)
    x_ticks = []
    categorical_labels = []
    
    for t in texts:
        bbox = t.get("bbox")
        if not bbox:
            continue
        
        # X-axis ticks: on bottom 25% of image
        center_y = (bbox["y_min"] + bbox["y_max"]) / 2
        if center_y < image_height * 0.75:
            continue
        
        # Skip if too far left (might be Y-axis)
        center_x = (bbox["x_min"] + bbox["x_max"]) / 2
        if center_x < image_width * 0.1:
            continue
        
        text = t.get("text", "").strip()
        value = parse_numeric_text(text)
        
        if value is not None:
            x_ticks.append({
                "pixel_x": center_x,
                "value": value,
                "text": text,
            })
        else:
            # Categorical label
            if len(text) > 0 and len(text) < 30:  # Reasonable label length
                categorical_labels.append({
                    "pixel_x": center_x,
                    "label": text,
                })
    
    # If we have numeric X-axis
    if len(x_ticks) >= 2:
        x_ticks.sort(key=lambda t: t["pixel_x"])
        
        pixels = [t["pixel_x"] for t in x_ticks]
        values = [t["value"] for t in x_ticks]
        
        pixel_min, pixel_max = min(pixels), max(pixels)
        value_min, value_max = min(values), max(values)
        
        if abs(value_max - value_min) < 1e-10:
            return {"calibrated": False, "reason": "zero_range", "is_categorical": False}
        
        scale_factor = (pixel_max - pixel_min) / (value_max - value_min)
        
        return {
            "calibrated": True,
            "is_categorical": False,
            "pixel_min": pixel_min,
            "pixel_max": pixel_max,
            "value_min": value_min,
            "value_max": value_max,
            "scale_factor": abs(scale_factor),
            "ticks_found": len(x_ticks),
            "tick_values": values,
        }
    
    # If we have categorical labels
    elif len(categorical_labels) >= 2:
        categorical_labels.sort(key=lambda t: t["pixel_x"])
        
        return {
            "calibrated": True,
            "is_categorical": True,
            "labels": [c["label"] for c in categorical_labels],
            "pixel_positions": [c["pixel_x"] for c in categorical_labels],
            "ticks_found": len(categorical_labels),
        }
    
    return {"calibrated": False, "reason": "insufficient_ticks", "is_categorical": False}


def map_element_values(elements: list, y_calibration: dict, x_calibration: dict, chart_type: str) -> list:
    """
    Map element pixel positions to actual data values using axis calibration.
    """
    mapped_elements = []
    
    for elem in elements:
        elem_copy = elem.copy()
        bbox = elem.get("bbox")
        center = elem.get("center")
        
        if not bbox:
            mapped_elements.append(elem_copy)
            continue
        
        # Get center position
        if center:
            center_x = center["x"]
            center_y = center["y"]
        else:
            center_x = (bbox["x_min"] + bbox["x_max"]) / 2
            center_y = (bbox["y_min"] + bbox["y_max"]) / 2
        
        # Map Y value
        if y_calibration.get("calibrated") and chart_type not in ["pie"]:
            pixel_min = y_calibration["pixel_min"]
            pixel_max = y_calibration["pixel_max"]
            value_min = y_calibration["value_min"]
            value_max = y_calibration["value_max"]
            
            # Determine which pixel to use for value mapping
            if chart_type in ["bar", "histogram"]:
                # For bars, use top of bar (y_min in image coords = max value)
                pixel_y = bbox["y_min"]
            else:
                # For points, use center
                pixel_y = center_y
            
            # Map pixel to value (inverted Y)
            if abs(pixel_max - pixel_min) > 1e-10:
                normalized = (pixel_max - pixel_y) / (pixel_max - pixel_min)
                mapped_value = value_min + normalized * (value_max - value_min)
                
                # Clamp to valid range with small tolerance
                tolerance = (value_max - value_min) * 0.1
                clamped = mapped_value < value_min - tolerance or mapped_value > value_max + tolerance
                mapped_value = max(value_min - tolerance, min(value_max + tolerance, mapped_value))
                
                elem_copy["mapped_y"] = round(mapped_value, 2)
                elem_copy["y_confidence"] = 0.8 if 0 <= normalized <= 1 else 0.5
                elem_copy["y_clamped"] = clamped
        
        # Map X value
        if x_calibration.get("calibrated"):
            if x_calibration.get("is_categorical"):
                # Map to nearest categorical label
                pixel_positions = x_calibration.get("pixel_positions", [])
                labels = x_calibration.get("labels", [])
                
                if pixel_positions and labels:
                    # Find nearest label
                    min_dist = float('inf')
                    nearest_label = None
                    nearest_idx = 0
                    
                    for i, px in enumerate(pixel_positions):
                        dist = abs(center_x - px)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_label = labels[i]
                            nearest_idx = i
                    
                    elem_copy["mapped_x_label"] = nearest_label
                    elem_copy["x_category_index"] = nearest_idx
                    elem_copy["x_confidence"] = 0.9 if min_dist < 50 else 0.6
            else:
                # Numeric X axis
                pixel_min = x_calibration["pixel_min"]
                pixel_max = x_calibration["pixel_max"]
                value_min = x_calibration["value_min"]
                value_max = x_calibration["value_max"]
                
                if abs(pixel_max - pixel_min) > 1e-10:
                    normalized = (center_x - pixel_min) / (pixel_max - pixel_min)
                    mapped_value = value_min + normalized * (value_max - value_min)
                    
                    tolerance = (value_max - value_min) * 0.1
                    mapped_value = max(value_min - tolerance, min(value_max + tolerance, mapped_value))
                    
                    elem_copy["mapped_x"] = round(mapped_value, 2)
                    elem_copy["x_confidence"] = 0.8 if 0 <= normalized <= 1 else 0.5
        
        # Legacy field for backward compatibility
        if "mapped_y" in elem_copy:
            elem_copy["mapped_value"] = elem_copy["mapped_y"]
            elem_copy["mapping_confidence"] = elem_copy.get("y_confidence", 0.5)
        
        mapped_elements.append(elem_copy)
    
    return mapped_elements


def analyze_trend(values: list) -> dict:
    """
    Analyze trend in numeric values.
    Returns trend direction and statistics.
    """
    if len(values) < 2:
        return {"trend": "insufficient_data", "direction": None}
    
    # Calculate statistics
    sorted_vals = sorted(values)
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)
    
    # Calculate variance
    variance = sum((v - avg_val) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5
    
    # Simple trend: compare first half vs second half
    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
    second_half_avg = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0
    
    # Determine trend
    if len(values) >= 3:
        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
            direction = "up"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"
            direction = "down"
        else:
            trend = "stable"
            direction = "flat"
    else:
        trend = "unknown"
        direction = None
    
    return {
        "trend": trend,
        "direction": direction,
        "min": min_val,
        "max": max_val,
        "average": round(avg_val, 2),
        "std_dev": round(std_dev, 2),
        "range": round(max_val - min_val, 2),
        "count": len(values),
    }


def analyze_distribution(percentages: list) -> dict:
    """
    Analyze distribution for pie charts.
    """
    if not percentages:
        return {"analysis": "no_data"}
    
    values = [p["value"] for p in percentages]
    total = sum(values)
    
    # Check if percentages sum to ~100
    is_valid_pie = 95 <= total <= 105
    
    # Find dominant and minor segments
    sorted_segs = sorted(percentages, key=lambda x: x["value"], reverse=True)
    
    dominant = sorted_segs[0] if sorted_segs else None
    minor = sorted_segs[-1] if len(sorted_segs) > 1 else None
    
    return {
        "is_valid_percentage": is_valid_pie,
        "total_percentage": round(total, 1),
        "segment_count": len(percentages),
        "dominant_segment": {
            "value": dominant["value"],
            "text": dominant["text"],
        } if dominant else None,
        "smallest_segment": {
            "value": minor["value"],
            "text": minor["text"],
        } if minor else None,
        "even_distribution": max(values) - min(values) < 10 if values else False,
    }


def map_legend_to_colors(elements: list, texts: list) -> list:
    """
    Attempt to map legend text to element colors.
    """
    # Get legend texts
    legend_texts = [t for t in texts if t.get("role") == "legend"]
    
    # Get unique colors from elements
    color_elements = []
    seen_colors = set()
    for e in elements:
        if e.get("color"):
            color_tuple = (e["color"]["r"], e["color"]["g"], e["color"]["b"])
            if color_tuple not in seen_colors:
                seen_colors.add(color_tuple)
                color_elements.append({
                    "color": e["color"],
                    "hex": f"#{e['color']['r']:02x}{e['color']['g']:02x}{e['color']['b']:02x}",
                })
    
    # Simple mapping by order (legend often ordered same as elements)
    mappings = []
    for i, legend in enumerate(legend_texts):
        mapping = {
            "label": legend.get("text", ""),
            "color": color_elements[i] if i < len(color_elements) else None,
        }
        mappings.append(mapping)
    
    return mappings


def run_stage4(stage3_result: dict, image_path: Path) -> dict:
    """Run Stage 4: Reasoning (enhanced rule-based analysis with geometric mapping)."""
    
    # Extract key info from stage3
    chart_type = stage3_result.get("chart_type", "unknown")
    texts = stage3_result.get("texts", [])
    elements = stage3_result.get("elements", [])
    axis_info = stage3_result.get("axis_info", {})
    image_size = stage3_result.get("image_size", {"width": 600, "height": 400})
    
    # Step 1: Remap element types based on chart classification
    elements = remap_element_types(elements, chart_type)
    
    # Step 2: Calibrate both axes from OCR tick labels
    y_calibration = calibrate_y_axis(texts, elements, image_size)
    x_calibration = calibrate_x_axis(texts, elements, image_size)
    
    # Step 3: Map element pixel positions to actual values (both X and Y)
    if chart_type not in ["pie"]:
        elements = map_element_values(elements, y_calibration, x_calibration, chart_type)
    
    # Extract numeric values from text
    numeric_values = extract_numeric_values(texts)
    
    # Group texts by role
    title = None
    x_label = None
    y_label = None
    data_labels = []
    legend_items = []
    x_ticks = []
    y_ticks = []
    
    for t in texts:
        role = t.get("role", "unknown")
        text = t.get("text", "")
        bbox = t.get("bbox")
        
        if role == "title":
            title = text
        elif role in ["x_label", "xlabel"]:
            x_label = text
        elif role in ["y_label", "ylabel"]:
            y_label = text
        elif role == "data_label":
            data_labels.append({"text": text, "bbox": bbox})
        elif role == "legend":
            legend_items.append({"text": text, "bbox": bbox})
        elif role == "x_tick":
            x_ticks.append({"text": text, "bbox": bbox})
        elif role == "y_tick":
            y_ticks.append({"text": text, "bbox": bbox})
    
    # Chart-type specific analysis
    analysis = {}
    
    # Use mapped values if available
    mapped_values = [e.get("mapped_value") for e in elements if e.get("mapped_value") is not None]
    
    if chart_type in ["line", "bar", "area", "histogram"]:
        # Trend analysis - prefer mapped values over OCR values
        if mapped_values:
            analysis["trend"] = analyze_trend(mapped_values)
            analysis["value_source"] = "geometric_mapping"
        else:
            analysis["trend"] = analyze_trend(numeric_values["raw_numbers"])
            analysis["value_source"] = "ocr_extraction"
    
    elif chart_type == "pie":
        # Distribution analysis for pie charts
        analysis["distribution"] = analyze_distribution(numeric_values["percentages"])
        analysis["value_source"] = "ocr_percentages"
    
    elif chart_type == "scatter":
        # Point analysis for scatter plots
        analysis["scatter"] = {
            "point_count": len([e for e in elements if e.get("type") in ["point", "scatter"]]),
            "has_clusters": len(elements) > 10,
            "mapped_values": mapped_values[:10] if mapped_values else [],
        }
        analysis["value_source"] = "geometric_mapping" if mapped_values else "none"
    
    # Add calibration info to analysis
    analysis["y_calibration"] = y_calibration
    analysis["x_calibration"] = x_calibration
    
    # Legend-color mapping
    legend_mapping = map_legend_to_colors(elements, texts)
    
    # Generate description based on what we found
    description = generate_description(
        chart_type=chart_type,
        title=title,
        x_label=x_label,
        y_label=y_label,
        num_elements=len(elements),
        data_labels=[d["text"] for d in data_labels],
        analysis=analysis,
    )
    
    # Generate insights
    insights = generate_insights(
        chart_type=chart_type,
        elements=elements,
        texts=texts,
        axis_info=axis_info,
        numeric_values=numeric_values,
        analysis=analysis,
    )
    
    # Build mapped data points with both X and Y values
    data_points = []
    for i, elem in enumerate(elements):
        point = {
            "index": i,
            "type": elem.get("type"),
            "color": elem.get("color"),
        }
        
        # Y value
        if elem.get("mapped_y") is not None:
            point["y_value"] = elem.get("mapped_y")
            point["y_confidence"] = elem.get("y_confidence", 0.5)
        elif elem.get("mapped_value") is not None:
            point["y_value"] = elem.get("mapped_value")
            point["y_confidence"] = elem.get("mapping_confidence", 0.5)
        
        # X value (numeric or categorical)
        if elem.get("mapped_x") is not None:
            point["x_value"] = elem.get("mapped_x")
            point["x_confidence"] = elem.get("x_confidence", 0.5)
        elif elem.get("mapped_x_label") is not None:
            point["x_label"] = elem.get("mapped_x_label")
            point["x_category_index"] = elem.get("x_category_index")
            point["x_confidence"] = elem.get("x_confidence", 0.5)
        
        # Only add if we have at least Y value
        if "y_value" in point:
            data_points.append(point)
    
    return {
        "chart_type": chart_type,
        "title": title,
        "x_axis_label": x_label,
        "y_axis_label": y_label,
        "x_ticks": [t["text"] for t in x_ticks],
        "y_ticks": [t["text"] for t in y_ticks],
        "data_labels": [d["text"] for d in data_labels],
        "legend_items": [l["text"] for l in legend_items],
        "legend_mapping": legend_mapping,
        "num_elements": len(elements),
        "elements_with_values": elements,  # Elements now include mapped values
        "data_points": data_points,  # Data points with X and Y values
        "numeric_values": numeric_values,
        "analysis": analysis,
        "description": description,
        "insights": insights,
        "raw_stage3": stage3_result,
    }


def generate_description(
    chart_type: str,
    title: Optional[str],
    x_label: Optional[str],
    y_label: Optional[str],
    num_elements: int,
    data_labels: list,
    analysis: dict = None,
) -> str:
    """Generate human-readable description of the chart."""
    parts = []
    
    # Chart type
    type_desc = {
        "bar": "bar chart",
        "line": "line chart",
        "pie": "pie chart",
        "scatter": "scatter plot",
        "histogram": "histogram",
        "area": "area chart",
        "heatmap": "heatmap",
        "box": "box plot",
    }
    chart_desc = type_desc.get(chart_type, f"{chart_type} chart")
    
    if title:
        parts.append(f"This is a {chart_desc} titled '{title}'.")
    else:
        parts.append(f"This is a {chart_desc}.")
    
    # Axes
    if x_label or y_label:
        axis_parts = []
        if x_label:
            axis_parts.append(f"x-axis represents '{x_label}'")
        if y_label:
            axis_parts.append(f"y-axis represents '{y_label}'")
        parts.append(f"The {' and '.join(axis_parts)}.")
    
    # Elements
    if num_elements > 0:
        if chart_type == "bar":
            parts.append(f"The chart contains {num_elements} bars.")
        elif chart_type == "line":
            parts.append(f"The chart shows data points connected by lines.")
        elif chart_type == "pie":
            parts.append(f"The chart has {num_elements} segments.")
        elif chart_type == "scatter":
            parts.append(f"The chart displays {num_elements} data points.")
        elif chart_type == "histogram":
            parts.append(f"The histogram shows {num_elements} bins.")
        elif chart_type == "box":
            parts.append(f"The plot shows {num_elements} box-and-whisker elements.")
    
    # Add analysis-based description
    if analysis:
        # Trend analysis
        if "trend" in analysis and analysis["trend"].get("trend") not in ["insufficient_data", "unknown"]:
            trend_info = analysis["trend"]
            trend_word = {
                "increasing": "an increasing",
                "decreasing": "a decreasing",
                "stable": "a stable",
            }.get(trend_info["trend"], "")
            if trend_word:
                parts.append(f"The data shows {trend_word} trend.")
            if trend_info.get("min") is not None and trend_info.get("max") is not None:
                parts.append(f"Values range from {trend_info['min']} to {trend_info['max']}.")
        
        # Distribution analysis for pie
        if "distribution" in analysis:
            dist_info = analysis["distribution"]
            if dist_info.get("dominant_segment"):
                dom = dist_info["dominant_segment"]
                parts.append(f"The largest segment is {dom['text']} ({dom['value']}%).")
            if dist_info.get("even_distribution"):
                parts.append("The segments are relatively evenly distributed.")
    
    # Data labels (limited)
    if data_labels and len(data_labels) <= 5:
        labels_str = ", ".join(str(l) for l in data_labels[:5])
        parts.append(f"Visible labels include: {labels_str}.")
    elif data_labels:
        parts.append(f"The chart contains {len(data_labels)} labeled data points.")
    
    return " ".join(parts)


def generate_insights(
    chart_type: str,
    elements: list,
    texts: list,
    axis_info: dict,
    numeric_values: dict = None,
    analysis: dict = None,
) -> list:
    """Generate detailed insights about the chart."""
    insights = []
    
    # Basic insight about structure
    if elements:
        element_types = {}
        for e in elements:
            t = e.get("type", "unknown")
            element_types[t] = element_types.get(t, 0) + 1
        
        insights.append({
            "type": "structure",
            "text": f"Chart contains {sum(element_types.values())} visual elements: {element_types}",
            "confidence": 0.9,
        })
    
    # Numeric values insight
    if numeric_values:
        value_summary = []
        if numeric_values.get("percentages"):
            value_summary.append(f"{len(numeric_values['percentages'])} percentage values")
        if numeric_values.get("integers"):
            value_summary.append(f"{len(numeric_values['integers'])} integer values")
        if numeric_values.get("decimals"):
            value_summary.append(f"{len(numeric_values['decimals'])} decimal values")
        if numeric_values.get("currencies"):
            value_summary.append(f"{len(numeric_values['currencies'])} currency values")
        
        if value_summary:
            insights.append({
                "type": "data_values",
                "text": f"Extracted numeric data: {', '.join(value_summary)}",
                "confidence": 0.85,
            })
    
    # Trend insight (for line/bar/area charts)
    if analysis and "trend" in analysis:
        trend_info = analysis["trend"]
        if trend_info.get("trend") not in ["insufficient_data", "unknown"]:
            trend_desc = {
                "increasing": "Values show an upward trend",
                "decreasing": "Values show a downward trend",
                "stable": "Values remain relatively stable",
            }.get(trend_info["trend"], "")
            
            if trend_desc:
                detail = f"{trend_desc} (range: {trend_info['min']}-{trend_info['max']}, avg: {trend_info['average']})"
                insights.append({
                    "type": "trend",
                    "text": detail,
                    "confidence": 0.75,
                })
    
    # Distribution insight (for pie charts)
    if analysis and "distribution" in analysis:
        dist_info = analysis["distribution"]
        if dist_info.get("is_valid_percentage"):
            insights.append({
                "type": "distribution",
                "text": f"Valid pie chart with {dist_info['segment_count']} segments totaling {dist_info['total_percentage']}%",
                "confidence": 0.9,
            })
        
        if dist_info.get("dominant_segment"):
            dom = dist_info["dominant_segment"]
            insights.append({
                "type": "dominant",
                "text": f"Largest segment: {dom['text']} at {dom['value']}%",
                "confidence": 0.85,
            })
        
        if dist_info.get("even_distribution"):
            insights.append({
                "type": "balance",
                "text": "Segments are evenly distributed (difference < 10%)",
                "confidence": 0.8,
            })
    
    # OCR quality insight
    text_confidences = [t.get("confidence", 0) for t in texts if t.get("confidence")]
    if text_confidences:
        avg_conf = sum(text_confidences) / len(text_confidences)
        quality = "high" if avg_conf > 0.9 else "medium" if avg_conf > 0.7 else "low"
        insights.append({
            "type": "ocr_quality",
            "text": f"Text extraction: {quality} quality ({avg_conf:.1%} avg confidence, {len(texts)} regions)",
            "confidence": avg_conf,
        })
    
    # Axis insight
    if axis_info and axis_info.get("y_axis_detected"):
        y_range = axis_info.get("y_range", [None, None])
        if y_range[0] is not None:
            insights.append({
                "type": "axis",
                "text": f"Y-axis scale: {y_range[0]} to {y_range[1]}",
                "confidence": 0.8,
            })
    
    # Color insight
    colors_found = set()
    for e in elements:
        if e.get("color"):
            c = e["color"]
            colors_found.add(f"#{c['r']:02x}{c['g']:02x}{c['b']:02x}")
    
    if len(colors_found) > 1:
        insights.append({
            "type": "colors",
            "text": f"Chart uses {len(colors_found)} distinct colors for data representation",
            "confidence": 0.9,
        })
    
    return insights


def format_report(result: dict, image_path: Path, processing_time: float) -> str:
    """Format analysis result as readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("CHART ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Source info
    lines.append(f"Source: {image_path.name}")
    lines.append(f"Processing time: {processing_time:.2f}s")
    lines.append("")
    
    # Chart info
    lines.append("-" * 40)
    lines.append("CHART INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Type: {result['chart_type'].upper()}")
    if result.get('title'):
        lines.append(f"Title: {result['title']}")
    if result.get('x_axis_label'):
        lines.append(f"X-Axis: {result['x_axis_label']}")
    if result.get('y_axis_label'):
        lines.append(f"Y-Axis: {result['y_axis_label']}")
    lines.append(f"Elements detected: {result['num_elements']}")
    lines.append("")
    
    # Description
    lines.append("-" * 40)
    lines.append("DESCRIPTION")
    lines.append("-" * 40)
    lines.append(result['description'])
    lines.append("")
    
    # Extracted numeric values
    numeric = result.get('numeric_values', {})
    has_numeric = any(numeric.get(k) for k in ['percentages', 'integers', 'decimals', 'currencies'])
    if has_numeric:
        lines.append("-" * 40)
        lines.append("EXTRACTED NUMERIC VALUES")
        lines.append("-" * 40)
        
        if numeric.get('percentages'):
            pct_vals = [f"{v['value']}%" for v in numeric['percentages'][:10]]
            lines.append(f"  Percentages: {', '.join(pct_vals)}")
        
        if numeric.get('integers'):
            int_vals = [str(v['value']) for v in numeric['integers'][:10]]
            lines.append(f"  Integers: {', '.join(int_vals)}")
        
        if numeric.get('decimals'):
            dec_vals = [str(v['value']) for v in numeric['decimals'][:10]]
            lines.append(f"  Decimals: {', '.join(dec_vals)}")
        
        if numeric.get('currencies'):
            cur_vals = [f"${v['value']:,.0f}" for v in numeric['currencies'][:10]]
            lines.append(f"  Currencies: {', '.join(cur_vals)}")
        
        lines.append("")
    
    # Analysis results
    analysis = result.get('analysis', {})
    if analysis:
        lines.append("-" * 40)
        lines.append("DATA ANALYSIS")
        lines.append("-" * 40)
        
        # Trend analysis
        if 'trend' in analysis and analysis['trend'].get('trend') not in ['insufficient_data', 'unknown']:
            trend = analysis['trend']
            lines.append(f"  Trend: {trend['trend'].upper()}")
            lines.append(f"  Range: {trend['min']} - {trend['max']}")
            lines.append(f"  Average: {trend['average']}")
            lines.append(f"  Std Dev: {trend['std_dev']}")
        
        # Distribution analysis
        if 'distribution' in analysis:
            dist = analysis['distribution']
            lines.append(f"  Segments: {dist.get('segment_count', 'N/A')}")
            lines.append(f"  Total %: {dist.get('total_percentage', 'N/A')}")
            if dist.get('dominant_segment'):
                dom = dist['dominant_segment']
                lines.append(f"  Largest: {dom['text']} ({dom['value']}%)")
            if dist.get('even_distribution'):
                lines.append("  Distribution: Even (< 10% variance)")
        
        # Value source info
        if analysis.get('value_source'):
            lines.append(f"  Value Source: {analysis['value_source']}")
        
        # Y-axis calibration info
        if analysis.get('y_calibration', {}).get('calibrated'):
            cal = analysis['y_calibration']
            lines.append(f"  Y-Axis Calibration: {cal['value_min']} to {cal['value_max']} ({cal['ticks_found']} ticks)")
        
        # X-axis calibration info
        x_cal = analysis.get('x_calibration', {})
        if x_cal.get('calibrated'):
            if x_cal.get('is_categorical'):
                labels = x_cal.get('labels', [])
                lines.append(f"  X-Axis: Categorical ({len(labels)} labels)")
                if labels:
                    lines.append(f"    Labels: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
            else:
                lines.append(f"  X-Axis Calibration: {x_cal['value_min']} to {x_cal['value_max']} ({x_cal['ticks_found']} ticks)")
        
        lines.append("")
    
    # Mapped Data Points (Geometric Value Mapping)
    data_points = result.get('data_points', [])
    if data_points:
        lines.append("-" * 40)
        lines.append("MAPPED DATA VALUES (Geometric)")
        lines.append("-" * 40)
        # Sort by Y value descending
        sorted_points = sorted(data_points, key=lambda p: p.get('y_value', p.get('value', 0)), reverse=True)
        for i, point in enumerate(sorted_points[:15]):
            y_val = point.get('y_value', point.get('value', 0))
            y_conf = point.get('y_confidence', point.get('confidence', 0))
            ptype = point.get('type', 'unknown')
            color = point.get('color')
            
            # Build X part
            x_part = ""
            if point.get('x_label'):
                x_part = f"x='{point['x_label']}'"
            elif point.get('x_value') is not None:
                x_part = f"x={point['x_value']:.1f}"
            
            # Build color part
            color_str = f" [{color['r']},{color['g']},{color['b']}]" if color else ""
            
            if x_part:
                lines.append(f"  {i+1}. {ptype}: y={y_val:.2f}, {x_part} ({y_conf:.0%}){color_str}")
            else:
                lines.append(f"  {i+1}. {ptype}: y={y_val:.2f} ({y_conf:.0%}){color_str}")
        
        if len(data_points) > 15:
            lines.append(f"  ... and {len(data_points) - 15} more elements")
        lines.append("")
    
    # Legend mapping
    legend_mapping = result.get('legend_mapping', [])
    if legend_mapping:
        lines.append("-" * 40)
        lines.append("LEGEND MAPPING")
        lines.append("-" * 40)
        for mapping in legend_mapping[:10]:
            label = mapping.get('label', 'Unknown')
            color = mapping.get('color')
            if color:
                lines.append(f"  {label}: {color['hex']}")
            else:
                lines.append(f"  {label}: (no color)")
        lines.append("")
    
    # Insights
    if result.get('insights'):
        lines.append("-" * 40)
        lines.append("INSIGHTS")
        lines.append("-" * 40)
        for insight in result['insights']:
            conf = f"({insight['confidence']:.0%})" if insight.get('confidence') else ""
            lines.append(f"[{insight['type'].upper()}] {insight['text']} {conf}")
        lines.append("")
    
    # X/Y tick labels
    if result.get('x_ticks') or result.get('y_ticks'):
        lines.append("-" * 40)
        lines.append("AXIS TICK LABELS")
        lines.append("-" * 40)
        if result.get('x_ticks'):
            x_vals = result['x_ticks'][:10]
            lines.append(f"  X-axis: {', '.join(x_vals)}")
        if result.get('y_ticks'):
            y_vals = result['y_ticks'][:10]
            lines.append(f"  Y-axis: {', '.join(y_vals)}")
        lines.append("")
    
    # Legend items
    if result.get('legend_items'):
        lines.append("-" * 40)
        lines.append("LEGEND ITEMS")
        lines.append("-" * 40)
        for item in result['legend_items'][:15]:
            lines.append(f"  - {item}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def get_random_chart(chart_type: Optional[str] = None) -> Path:
    """Get random chart from classified_charts dataset."""
    base_dir = PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts"
    
    if chart_type:
        chart_dir = base_dir / chart_type
        if not chart_dir.exists():
            raise ValueError(f"Chart type directory not found: {chart_dir}")
        images = list(chart_dir.glob("*.png"))
    else:
        images = list(base_dir.glob("*/*.png"))
    
    if not images:
        raise ValueError("No images found in dataset")
    
    return random.choice(images)


def main():
    parser = argparse.ArgumentParser(description="Demo full chart analysis pipeline")
    parser.add_argument(
        "image_path",
        type=Path,
        nargs="?",
        help="Path to chart image",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick random chart from dataset",
    )
    parser.add_argument(
        "--chart-type",
        type=str,
        choices=["bar", "line", "pie", "scatter", "histogram", "area", "heatmap", "box"],
        help="Filter random selection by chart type",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable OCR cache (force live OCR)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save structured output to JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed processing info",
    )
    args = parser.parse_args()
    
    # Get image path
    if args.random or args.image_path is None:
        image_path = get_random_chart(args.chart_type)
        print(f"\nSelected: {image_path}\n")
    else:
        image_path = args.image_path
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load image
        print("Loading image...")
        image = load_image(image_path)
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        start_time = time.time()
        
        # Stage 3: Extraction
        print("\n[Stage 3] Extracting features...")
        use_cache = not args.no_cache
        stage3_result = run_stage3(image_path, image, use_cache=use_cache)
        s3_time = time.time() - start_time
        print(f"  - Chart type: {stage3_result.get('chart_type', 'unknown')}")
        print(f"  - Texts found: {len(stage3_result.get('texts', []))}")
        print(f"  - Elements found: {len(stage3_result.get('elements', []))}")
        print(f"  - Time: {s3_time:.2f}s" + (" (cached)" if use_cache else ""))
        
        # Stage 4: Reasoning
        print("\n[Stage 4] Generating analysis...")
        s4_start = time.time()
        final_result = run_stage4(stage3_result, image_path)
        s4_time = time.time() - s4_start
        print(f"  - Time: {s4_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Generate report
        print("\n")
        report = format_report(final_result, image_path, total_time)
        print(report)
        
        # Save JSON if requested
        if args.output_json:
            # Remove raw_stage3 for cleaner output
            output_data = {k: v for k, v in final_result.items() if k != "raw_stage3"}
            output_data["source_image"] = str(image_path)
            output_data["processing_time_seconds"] = total_time
            
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nJSON output saved to: {args.output_json}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
