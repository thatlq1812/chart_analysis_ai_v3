"""
Pipeline Report Generator

Runs Stage 3 extraction on sample images and generates a summary report.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from core_engine.stages.s3_extraction import ExtractionConfig, Stage3Extraction


def generate_report(
    input_dir: Path,
    output_dir: Path,
    max_images: int = 20,
    enable_ocr: bool = True,
) -> dict:
    """
    Run Stage 3 on sample images and generate report.
    
    Args:
        input_dir: Directory containing chart images
        output_dir: Directory to save report
        max_images: Maximum images to process
        enable_ocr: Whether to enable OCR (slower)
    
    Returns:
        Report dictionary
    """
    # Configuration
    config = ExtractionConfig(
        use_ml_classifier=True,
        enable_ocr=enable_ocr,
        ocr_engine="easyocr",
        enable_vectorization=True,
        enable_element_detection=True,
    )
    
    stage = Stage3Extraction(config)
    
    # Find images
    image_extensions = [".png", ".jpg", ".jpeg"]
    images = []
    for ext in image_extensions:
        images.extend(input_dir.glob(f"*{ext}"))
    
    images = sorted(images)[:max_images]
    
    print(f"Processing {len(images)} images...")
    print(f"OCR enabled: {enable_ocr}")
    print("-" * 60)
    
    # Process images
    results = []
    type_counts = {"bar": 0, "line": 0, "pie": 0, "other": 0, "unknown": 0}
    
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}...", end=" ", flush=True)
        
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print("SKIP (cannot read)")
                continue
            
            result = stage.process_image(image, chart_id=img_path.stem)
            
            chart_type = result.chart_type.value if hasattr(result.chart_type, "value") else str(result.chart_type)
            type_counts[chart_type] = type_counts.get(chart_type, 0) + 1
            
            text_count = len(result.texts) if result.texts else 0
            element_count = len(result.elements) if result.elements else 0
            
            # Get confidence if available
            confidence = getattr(result, "classification_confidence", None)
            
            results.append({
                "image": img_path.name,
                "chart_type": chart_type,
                "confidence": round(confidence, 2) if confidence else None,
                "text_count": text_count,
                "element_count": element_count,
                "texts_preview": [t.text for t in (result.texts or [])[:5]],
            })
            
            print(f"type={chart_type}, texts={text_count}, elements={element_count}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "image": img_path.name,
                "error": str(e),
            })
    
    # Generate report
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "enable_ocr": enable_ocr,
            "ocr_engine": config.ocr_engine,
            "use_ml_classifier": config.use_ml_classifier,
        },
        "summary": {
            "total_images": len(images),
            "processed": len([r for r in results if "error" not in r]),
            "errors": len([r for r in results if "error" in r]),
            "type_distribution": type_counts,
        },
        "results": results,
    }
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"stage3_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("-" * 60)
    print(f"Report saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  Total: {report['summary']['total_images']}")
    print(f"  Processed: {report['summary']['processed']}")
    print(f"  Errors: {report['summary']['errors']}")
    print(f"  Types: {type_counts}")
    
    return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Stage 3 pipeline report")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/academic_dataset/images"),
        help="Input image directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Output directory for report",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum images to process",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR (faster)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / args.input_dir if not args.input_dir.is_absolute() else args.input_dir
    output_dir = script_dir / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    
    generate_report(
        input_dir=input_dir,
        output_dir=output_dir,
        max_images=args.max_images,
        enable_ocr=not args.no_ocr,
    )


if __name__ == "__main__":
    main()
