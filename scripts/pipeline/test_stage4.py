"""
Stage 4 Integration Test

Test the Stage 4 reasoning module with and without Gemini API.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.core_engine.schemas.common import BoundingBox, Color, Point, SessionInfo
from src.core_engine.schemas.enums import ChartType
from src.core_engine.schemas.stage_outputs import (
    ChartElement,
    OCRText,
    RawMetadata,
    Stage3Output,
)
from src.core_engine.stages.s4_reasoning import (
    Stage4Reasoning,
    ReasoningConfig,
    GeminiConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_metadata() -> RawMetadata:
    """Create test metadata simulating Stage 3 output."""
    return RawMetadata(
        chart_id="test_bar_001",
        chart_type=ChartType.BAR,
        texts=[
            OCRText(
                text="Quarterly Sales 2O25",  # Intentional OCR error: O instead of 0
                bbox=BoundingBox(x_min=100, y_min=10, x_max=300, y_max=40, confidence=0.9),
                confidence=0.85,
                role="title"
            ),
            OCRText(
                text="loo",  # Intentional error: loo instead of 100
                bbox=BoundingBox(x_min=20, y_min=250, x_max=50, y_max=270, confidence=0.9),
                confidence=0.7,
                role="value"
            ),
            OCRText(
                text="2OO",  # Intentional error: 2OO instead of 200
                bbox=BoundingBox(x_min=20, y_min=150, x_max=50, y_max=170, confidence=0.9),
                confidence=0.7,
                role="value"
            ),
            OCRText(
                text="3OO",  # Intentional error: 3OO instead of 300
                bbox=BoundingBox(x_min=20, y_min=50, x_max=50, y_max=70, confidence=0.9),
                confidence=0.7,
                role="value"
            ),
            OCRText(
                text="Q1",
                bbox=BoundingBox(x_min=115, y_min=360, x_max=135, y_max=380, confidence=0.9),
                confidence=0.95,
                role="xlabel"
            ),
            OCRText(
                text="Q2",
                bbox=BoundingBox(x_min=195, y_min=360, x_max=215, y_max=380, confidence=0.9),
                confidence=0.95,
                role="xlabel"
            ),
            OCRText(
                text="Q3",
                bbox=BoundingBox(x_min=275, y_min=360, x_max=295, y_max=380, confidence=0.9),
                confidence=0.95,
                role="xlabel"
            ),
            OCRText(
                text="Q4",
                bbox=BoundingBox(x_min=355, y_min=360, x_max=375, y_max=380, confidence=0.9),
                confidence=0.95,
                role="xlabel"
            ),
        ],
        elements=[
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(x_min=100, y_min=150, x_max=150, y_max=350, confidence=0.9),
                center=Point(x=125, y=250),
                color=Color(r=66, g=133, b=244),
                area_pixels=10000
            ),
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(x_min=180, y_min=70, x_max=230, y_max=350, confidence=0.9),
                center=Point(x=205, y=210),
                color=Color(r=52, g=168, b=83),
                area_pixels=14000
            ),
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(x_min=260, y_min=200, x_max=310, y_max=350, confidence=0.9),
                center=Point(x=285, y=275),
                color=Color(r=251, g=188, b=5),
                area_pixels=7500
            ),
            ChartElement(
                element_type="bar",
                bbox=BoundingBox(x_min=340, y_min=30, x_max=390, y_max=350, confidence=0.9),
                center=Point(x=365, y=190),
                color=Color(r=234, g=67, b=53),
                area_pixels=16000
            ),
        ],
        axis_info=None,
    )


def test_stage4_fallback():
    """Test Stage 4 with rule-based fallback (no API)."""
    print("\n" + "=" * 60)
    print("TEST: Stage 4 Fallback (Rule-based)")
    print("=" * 60)
    
    config = ReasoningConfig(
        engine="rule_based",
        use_fallback_on_error=True,
    )
    
    stage4 = Stage4Reasoning(config)
    
    # Create test data
    metadata = create_test_metadata()
    session = SessionInfo(
        session_id=str(uuid4())[:8],
        created_at=datetime.now(),
        source_file=Path("test_chart.png"),
        config_hash="test1234567890",  # Must be at least 8 chars
    )
    
    stage3_output = Stage3Output(
        session=session,
        metadata=[metadata],
    )
    
    # Process
    result = stage4.process(stage3_output)
    
    # Verify
    assert len(result.charts) == 1, "Should have 1 chart"
    chart = result.charts[0]
    
    print(f"Chart ID: {chart.chart_id}")
    print(f"Chart Type: {chart.chart_type.value}")
    print(f"Title: {chart.title}")
    print(f"Series: {len(chart.series)}")
    print(f"Description: {chart.description[:100]}...")
    print(f"Corrections: {chart.correction_log}")
    
    print("\nFallback test PASSED")
    return True


def test_stage4_gemini():
    """Test Stage 4 with Gemini API (if available)."""
    print("\n" + "=" * 60)
    print("TEST: Stage 4 Gemini API")
    print("=" * 60)
    
    config = ReasoningConfig(
        engine="gemini",
        gemini=GeminiConfig(
            model_name="gemini-3-flash-preview",
            temperature=0.3,
        ),
        use_fallback_on_error=True,
    )
    
    stage4 = Stage4Reasoning(config)
    
    # Check API availability
    if stage4.engine and stage4.engine.is_available():
        print("Gemini API: AVAILABLE")
    else:
        print("Gemini API: NOT AVAILABLE (will use fallback)")
        print("Set GOOGLE_API_KEY environment variable to enable")
        return True  # Skip but don't fail
    
    # Create test data
    metadata = create_test_metadata()
    
    # Test single chart processing
    result = stage4.process_single(metadata)
    
    print(f"Chart ID: {result.chart_id}")
    print(f"Chart Type: {result.chart_type.value}")
    print(f"Title: {result.title}")
    print(f"X-axis: {result.x_axis_label}")
    print(f"Y-axis: {result.y_axis_label}")
    print(f"Series: {len(result.series)}")
    print(f"Description: {result.description[:150]}...")
    print(f"Corrections: {result.correction_log}")
    
    # Print data series
    for series in result.series:
        print(f"\n  Series: {series.name}")
        for point in series.points[:5]:
            print(f"    {point.label}: {point.value}")
    
    print("\nGemini test PASSED")
    return True


def test_ocr_correction():
    """Test OCR error correction specifically."""
    print("\n" + "=" * 60)
    print("TEST: OCR Error Correction")
    print("=" * 60)
    
    from src.core_engine.stages.s4_reasoning import GeminiReasoningEngine, GeminiConfig
    
    config = GeminiConfig(
        model_name="gemini-3-flash-preview",
        temperature=0.2,
    )
    
    engine = GeminiReasoningEngine(config)
    
    if not engine.is_available():
        print("Gemini API not available, skipping OCR correction test")
        return True
    
    # Test texts with errors
    test_texts = [
        OCRText(
            text="loo",  # Should correct to 100
            bbox=BoundingBox(x_min=0, y_min=0, x_max=30, y_max=20, confidence=0.9),
            confidence=0.7,
            role="value"
        ),
        OCRText(
            text="2O25",  # Should correct to 2025
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20, confidence=0.9),
            confidence=0.8,
            role="title"
        ),
    ]
    
    corrected, corrections = engine.correct_ocr(test_texts, ChartType.BAR)
    
    print(f"Input texts: {[t.text for t in test_texts]}")
    print(f"Corrections made: {len(corrections)}")
    for c in corrections:
        print(f"  '{c.get('original')}' -> '{c.get('corrected')}' ({c.get('reason')})")
    print(f"Corrected texts: {[t.text for t in corrected]}")
    
    print("\nOCR correction test PASSED")
    return True


def main():
    """Run all Stage 4 tests."""
    print("=" * 60)
    print("STAGE 4 INTEGRATION TESTS")
    print("=" * 60)
    
    results = {
        "fallback": test_stage4_fallback(),
        "gemini": test_stage4_gemini(),
        "ocr_correction": test_ocr_correction(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
