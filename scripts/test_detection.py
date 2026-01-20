#!/usr/bin/env python3
"""
Test Chart Detection on PDF Files

This script runs the trained YOLO model on PDF files and visualizes results.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Test chart detection on PDFs")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained YOLO weights (best.pt)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input PDF or image file (default: use sample PDF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PDF render DPI (default: 150)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results interactively",
    )

    args = parser.parse_args()

    # Validate weights
    if not args.weights.exists():
        logger.error(f"Weights not found | path={args.weights}")
        sys.exit(1)

    # Find input file
    if args.input is None:
        # Use sample PDF
        sample_pdfs = list((PROJECT_ROOT / "data" / "raw_pdfs").glob("*.pdf"))
        if sample_pdfs:
            args.input = sample_pdfs[0]
            logger.info(f"Using sample PDF | path={args.input.name}")
        else:
            logger.error("No input file specified and no sample PDFs found")
            sys.exit(1)

    if not args.input.exists():
        logger.error(f"Input file not found | path={args.input}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Chart Detection Test")
    logger.info("=" * 60)
    logger.info(f"Weights:    {args.weights}")
    logger.info(f"Input:      {args.input}")
    logger.info(f"Output:     {args.output_dir}")
    logger.info(f"Confidence: {args.conf}")
    logger.info("=" * 60)

    try:
        from ultralytics import YOLO
        import fitz
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install: pip install ultralytics pymupdf pillow")
        sys.exit(1)

    # Load model
    logger.info("Loading YOLO model...")
    model = YOLO(str(args.weights))

    # Process input
    if args.input.suffix.lower() == ".pdf":
        logger.info("Processing PDF...")
        doc = fitz.open(args.input)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            zoom = args.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append((page_num + 1, img))
        
        doc.close()
        logger.info(f"Extracted {len(pages)} pages")
    else:
        # Single image
        img = Image.open(args.input).convert("RGB")
        pages = [(1, img)]

    # Run detection
    total_detections = 0
    results_summary = []

    for page_num, img in pages:
        logger.info(f"Processing page {page_num}...")
        
        # Convert to numpy
        img_np = np.array(img)
        
        # Run inference
        results = model.predict(
            img_np,
            conf=args.conf,
            verbose=False,
        )
        
        # Process results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                })
        
        total_detections += len(detections)
        results_summary.append({
            "page": page_num,
            "detections": len(detections),
            "details": detections,
        })
        
        # Visualize
        draw = ImageDraw.Draw(img)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label = f"chart {conf:.2f}"
            draw.rectangle([x1, y1 - 20, x1 + 100, y1], fill="red")
            draw.text((x1 + 5, y1 - 18), label, fill="white")
        
        # Save
        output_path = args.output_dir / f"page_{page_num:03d}_detected.jpg"
        img.save(output_path, "JPEG", quality=95)
        
        # Crop detected charts
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            cropped = Image.open(args.output_dir / f"page_{page_num:03d}_detected.jpg").crop(
                (x1, y1, x2, y2)
            )
            crop_path = args.output_dir / f"page_{page_num:03d}_chart_{i+1}.jpg"
            cropped.save(crop_path, "JPEG", quality=95)

    # Summary
    logger.info("=" * 60)
    logger.info("Detection Results")
    logger.info("=" * 60)
    logger.info(f"Pages processed:     {len(pages)}")
    logger.info(f"Total detections:    {total_detections}")
    logger.info("")

    for res in results_summary:
        status = "DETECTED" if res["detections"] > 0 else "NO CHARTS"
        logger.info(f"  Page {res['page']:3d}: {res['detections']} charts [{status}]")
        for det in res["details"]:
            bbox = det["bbox"]
            logger.info(f"           -> bbox={bbox}, conf={det['confidence']:.3f}")

    logger.info("")
    logger.info(f"Results saved to: {args.output_dir}")

    # Create HTML viewer
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Detection Results - {args.input.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .page {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
        .page img {{ max-width: 100%; height: auto; }}
        .stats {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Chart Detection Results</h1>
    <div class="stats">
        <p><strong>Input:</strong> {args.input.name}</p>
        <p><strong>Pages:</strong> {len(pages)}</p>
        <p><strong>Total Detections:</strong> {total_detections}</p>
        <p><strong>Confidence Threshold:</strong> {args.conf}</p>
    </div>
"""
    
    for res in results_summary:
        html_content += f"""
    <div class="page">
        <h2>Page {res['page']} - {res['detections']} chart(s)</h2>
        <img src="page_{res['page']:03d}_detected.jpg" alt="Page {res['page']}">
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    html_path = args.output_dir / "results.html"
    html_path.write_text(html_content)
    logger.info(f"HTML viewer: {html_path}")


if __name__ == "__main__":
    main()
