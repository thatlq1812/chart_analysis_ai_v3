"""Test OCR engine with EasyOCR."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from core_engine.stages.s3_extraction.ocr_engine import OCRConfig, OCREngine


def main():
    """Test EasyOCR integration."""
    # Test EasyOCR
    print("Testing EasyOCR...")
    config = OCRConfig(engine="easyocr", languages=["en"])
    ocr = OCREngine(config)

    # Load test image
    img_path = Path(__file__).parent.parent / "data/academic_dataset/images/plotqa_paper_p02_img00.png"
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    image = cv2.imread(str(img_path))
    print(f"Image shape: {image.shape}")

    # Extract text
    result = ocr.extract_text(image, chart_id="test")
    print(f"Texts found: {len(result.texts)}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    
    print("\nExtracted texts:")
    for text in result.texts[:15]:
        print(f"  - [{text.role}] {text.text!r} (conf={text.confidence:.2f})")


if __name__ == "__main__":
    main()
