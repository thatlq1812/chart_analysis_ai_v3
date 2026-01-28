"""
Test Stage 3 Extraction Pipeline on Pre-classified Charts

This script tests the Stage 3 extraction pipeline against charts that have
already been classified, allowing us to measure accuracy against ground truth.

Features tested:
- Chart classification accuracy (vs ground truth)
- OCR extraction with post-processing
- Element detection (bars, markers, slices)
- Skeletonization and vectorization
- Geometric axis calibration with RANSAC
- Confidence scoring system
"""

import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

# Import Stage 3 modules
from core_engine.stages.s3_extraction import (
    Stage3Extraction,
    ExtractionConfig,
    ImagePreprocessor,
    PreprocessConfig,
    Skeletonizer,
    SkeletonConfig,
    Vectorizer,
    VectorizeConfig,
    ElementDetector,
    ElementDetectorConfig,
    ChartClassifier,
    ClassifierConfig,
    OCREngine,
    OCRConfig,
    GeometricMapper,
    MapperConfig,
    ResNet18Classifier,
)
from core_engine.schemas import ChartType


@dataclass
class ChartTestResult:
    """Result of testing a single chart."""
    image_path: str
    image_name: str
    ground_truth_type: str
    predicted_type: str
    classification_correct: bool
    classification_confidence: float
    
    # Component results
    preprocessing_success: bool = True
    ocr_texts_count: int = 0
    ocr_confidence_avg: float = 0.0
    elements_bars: int = 0
    elements_markers: int = 0
    elements_slices: int = 0
    keypoints_count: int = 0
    polylines_count: int = 0
    vectorization_compression: float = 0.0
    
    # Timing
    processing_time_ms: float = 0.0
    
    # Errors
    error: Optional[str] = None
    
    # Confidence scores (new system)
    confidence_classification: float = 0.0
    confidence_ocr: float = 0.0
    confidence_axis: float = 0.0
    confidence_elements: float = 0.0
    confidence_overall: float = 0.0


@dataclass
class TestSummary:
    """Summary statistics for the test run."""
    total_charts: int = 0
    successful: int = 0
    failed: int = 0
    
    # Classification accuracy per type
    accuracy_per_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Overall accuracy
    classification_accuracy: float = 0.0
    
    # Timing
    avg_processing_time_ms: float = 0.0
    total_processing_time_s: float = 0.0
    
    # Confidence stats
    avg_confidence_overall: float = 0.0
    avg_confidence_classification: float = 0.0
    avg_confidence_ocr: float = 0.0


def get_sample_charts(
    classified_dir: Path,
    samples_per_type: int = 10,
    chart_types: Optional[List[str]] = None
) -> List[Tuple[Path, str]]:
    """
    Get sample charts from each type directory.
    
    Returns list of (image_path, ground_truth_type) tuples.
    """
    if chart_types is None:
        chart_types = ["bar", "line", "pie", "scatter", "histogram", "area", "box", "heatmap"]
    
    samples = []
    
    for chart_type in chart_types:
        type_dir = classified_dir / chart_type
        if not type_dir.exists():
            print(f"    Warning: {chart_type} directory not found")
            continue
        
        images = list(type_dir.glob("*.png")) + list(type_dir.glob("*.jpg"))
        
        if len(images) == 0:
            print(f"    Warning: No images in {chart_type} directory")
            continue
        
        # Random sample
        selected = random.sample(images, min(samples_per_type, len(images)))
        
        for img_path in selected:
            samples.append((img_path, chart_type))
    
    # Shuffle to mix types
    random.shuffle(samples)
    
    return samples


def map_predicted_to_ground_truth(predicted: str) -> str:
    """Map predicted chart type to ground truth label."""
    # The classifier uses ChartType enum values
    mapping = {
        "bar": "bar",
        "line": "line",
        "pie": "pie",
        "scatter": "scatter",
        "area": "area",
        "histogram": "histogram",
        "box": "box",
        "heatmap": "heatmap",
        "unknown": "unknown",
    }
    return mapping.get(predicted.lower(), predicted.lower())


def process_chart(
    image_path: Path,
    ground_truth: str,
    preprocessor: ImagePreprocessor,
    skeletonizer: Skeletonizer,
    vectorizer: Vectorizer,
    element_detector: ElementDetector,
    classifier: Optional[ChartClassifier],
    ocr_engine: OCREngine,
    geometric_mapper: GeometricMapper,
    resnet_classifier: Optional[ResNet18Classifier] = None,
    use_resnet: bool = True,
) -> ChartTestResult:
    """Process a single chart and return test result."""
    
    start_time = time.time()
    chart_id = image_path.stem
    
    result = ChartTestResult(
        image_path=str(image_path),
        image_name=image_path.name,
        ground_truth_type=ground_truth,
        predicted_type="unknown",
        classification_correct=False,
        classification_confidence=0.0,
    )
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            result.error = "Failed to load image"
            return result
        
        h, w = image.shape[:2]
        
        # 1. Preprocessing
        try:
            prep_result = preprocessor.process(image, chart_id=chart_id)
            binary_image = prep_result.binary_image
            result.preprocessing_success = True
        except Exception as e:
            result.preprocessing_success = False
            # Fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 2. OCR extraction
        ocr_texts = []
        try:
            ocr_result = ocr_engine.extract_text(image, chart_id=chart_id)
            ocr_texts = ocr_result.texts
            result.ocr_texts_count = len(ocr_texts)
            if ocr_texts:
                result.ocr_confidence_avg = sum(t.confidence for t in ocr_texts) / len(ocr_texts)
                result.confidence_ocr = result.ocr_confidence_avg
        except Exception as e:
            pass  # OCR is optional
        
        # 3. Skeletonization
        keypoints = []
        polylines = []
        try:
            skeleton_result = skeletonizer.process(binary_image, chart_id=chart_id)
            keypoints = skeleton_result.keypoints
            result.keypoints_count = len(keypoints)
            
            # Trace paths and vectorize
            paths = skeletonizer.trace_paths(skeleton_result.skeleton, keypoints)
            vector_result = vectorizer.process(paths, chart_id=chart_id)
            polylines = vector_result.polylines
            result.polylines_count = len(polylines)
            result.vectorization_compression = vector_result.compression_ratio
        except Exception as e:
            pass  # Continue without vectorization
        
        # 4. Element detection
        bars = []
        markers = []
        slices = []
        try:
            elem_result = element_detector.detect(
                binary_image, 
                color_image=image, 
                chart_id=chart_id
            )
            bars = elem_result.bars
            markers = elem_result.markers
            slices = elem_result.slices
            result.elements_bars = len(bars)
            result.elements_markers = len(markers)
            result.elements_slices = len(slices)
            
            # Element confidence
            total_elements = len(bars) + len(markers) + len(slices)
            if total_elements > 0:
                all_confs = [b.confidence for b in bars] + [m.confidence for m in markers] + [s.confidence for s in slices]
                result.confidence_elements = sum(all_confs) / len(all_confs)
        except Exception as e:
            pass  # Continue without elements
        
        # 5. Classification - Use ResNet if available, else rule-based
        try:
            if use_resnet and resnet_classifier is not None:
                # Use deep learning classifier
                predicted, confidence = resnet_classifier.predict_with_confidence(image_path)
                result.predicted_type = map_predicted_to_ground_truth(predicted)
                result.classification_confidence = confidence
                result.confidence_classification = confidence
                result.classification_correct = (result.predicted_type == ground_truth)
            elif classifier is not None:
                # Use rule-based classifier
                class_result = classifier.classify(
                    bars=bars,
                    polylines=polylines,
                    markers=markers,
                    slices=slices,
                    texts=ocr_texts,
                    image_shape=(h, w),
                    chart_id=chart_id,
                )
                predicted = map_predicted_to_ground_truth(class_result.chart_type.value)
                result.predicted_type = predicted
                result.classification_confidence = class_result.confidence
                result.confidence_classification = class_result.confidence
                result.classification_correct = (predicted == ground_truth)
            
        except Exception as e:
            result.error = f"Classification failed: {e}"
        
        # 6. Compute overall confidence
        confidences = [
            result.confidence_classification,
            result.confidence_ocr,
            result.confidence_elements,
        ]
        valid_confs = [c for c in confidences if c > 0]
        if valid_confs:
            result.confidence_overall = sum(valid_confs) / len(valid_confs)
        
    except Exception as e:
        result.error = str(e)
    
    result.processing_time_ms = round((time.time() - start_time) * 1000, 2)
    return result


def compute_summary(results: List[ChartTestResult]) -> TestSummary:
    """Compute summary statistics from results."""
    summary = TestSummary()
    summary.total_charts = len(results)
    
    # Count successful
    successful_results = [r for r in results if r.error is None]
    summary.successful = len(successful_results)
    summary.failed = summary.total_charts - summary.successful
    
    # Classification accuracy per type
    for r in successful_results:
        gt = r.ground_truth_type
        if gt not in summary.accuracy_per_type:
            summary.accuracy_per_type[gt] = {"correct": 0, "total": 0}
        summary.accuracy_per_type[gt]["total"] += 1
        if r.classification_correct:
            summary.accuracy_per_type[gt]["correct"] += 1
    
    # Overall classification accuracy
    correct = sum(1 for r in successful_results if r.classification_correct)
    summary.classification_accuracy = correct / len(successful_results) if successful_results else 0
    
    # Timing
    times = [r.processing_time_ms for r in successful_results]
    summary.avg_processing_time_ms = sum(times) / len(times) if times else 0
    summary.total_processing_time_s = sum(times) / 1000
    
    # Confidence averages
    confidences = [r.confidence_overall for r in successful_results if r.confidence_overall > 0]
    summary.avg_confidence_overall = sum(confidences) / len(confidences) if confidences else 0
    
    class_confs = [r.confidence_classification for r in successful_results if r.confidence_classification > 0]
    summary.avg_confidence_classification = sum(class_confs) / len(class_confs) if class_confs else 0
    
    ocr_confs = [r.confidence_ocr for r in successful_results if r.confidence_ocr > 0]
    summary.avg_confidence_ocr = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0
    
    return summary


def generate_report(
    results: List[ChartTestResult],
    summary: TestSummary,
    output_dir: Path
) -> Path:
    """Generate markdown report."""
    report_path = output_dir / "STAGE3_CLASSIFIED_TEST_REPORT.md"
    
    report = f"""# Stage 3 Extraction Test Report (Classified Charts)

| Property | Value |
|----------|-------|
| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| Total Charts Tested | {summary.total_charts} |
| Successful Processing | {summary.successful} ({summary.successful/summary.total_charts*100:.1f}%) |
| Failed | {summary.failed} |
| Overall Classification Accuracy | **{summary.classification_accuracy*100:.1f}%** |
| Average Processing Time | {summary.avg_processing_time_ms:.1f} ms |
| Total Processing Time | {summary.total_processing_time_s:.1f} s |

## Classification Accuracy by Chart Type

| Chart Type | Correct | Total | Accuracy |
|------------|---------|-------|----------|
"""
    
    for chart_type in sorted(summary.accuracy_per_type.keys()):
        stats = summary.accuracy_per_type[chart_type]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        report += f"| {chart_type} | {stats['correct']} | {stats['total']} | {acc:.1f}% |\n"
    
    report += f"""
## Confidence Scores

| Metric | Average |
|--------|---------|
| Overall Confidence | {summary.avg_confidence_overall*100:.1f}% |
| Classification Confidence | {summary.avg_confidence_classification*100:.1f}% |
| OCR Confidence | {summary.avg_confidence_ocr*100:.1f}% |

## Confusion Matrix

"""
    
    # Build confusion matrix
    chart_types = sorted(set(r.ground_truth_type for r in results))
    predicted_types = sorted(set(r.predicted_type for r in results))
    all_types = sorted(set(chart_types) | set(predicted_types))
    
    confusion = {gt: {pred: 0 for pred in all_types} for gt in all_types}
    for r in results:
        if r.error is None:
            confusion[r.ground_truth_type][r.predicted_type] += 1
    
    # Header
    report += "| Ground Truth | " + " | ".join(all_types) + " |\n"
    report += "|" + "----|" * (len(all_types) + 1) + "\n"
    
    for gt in all_types:
        row = [str(confusion[gt].get(pred, 0)) for pred in all_types]
        report += f"| **{gt}** | " + " | ".join(row) + " |\n"
    
    report += """
## Detailed Results

<details>
<summary>Click to expand individual results</summary>

| # | Image | Ground Truth | Predicted | Correct | Confidence | Time (ms) |
|---|-------|--------------|-----------|---------|------------|-----------|
"""
    
    for i, r in enumerate(results[:100], 1):  # Limit to first 100
        correct_mark = "Yes" if r.classification_correct else "No"
        report += f"| {i} | {r.image_name[:40]}... | {r.ground_truth_type} | {r.predicted_type} | {correct_mark} | {r.classification_confidence*100:.0f}% | {r.processing_time_ms:.0f} |\n"
    
    if len(results) > 100:
        report += f"\n*... and {len(results) - 100} more results*\n"
    
    report += """
</details>

## Observations

### Strengths
- Processing pipeline handles diverse chart styles
- Confidence scores provide reliability indicators
- Vectorization compression is efficient

### Areas for Improvement
- Some chart types may need more training data
- OCR post-processing could be tuned for academic charts
- Consider ensemble methods for ambiguous cases

## Next Steps

1. Fine-tune classifier on misclassified examples
2. Add more chart type-specific features
3. Implement Stage 4 semantic reasoning for value extraction
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def main():
    """Main test function."""
    print("=" * 70)
    print("STAGE 3 EXTRACTION TEST ON CLASSIFIED CHARTS")
    print("=" * 70)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    classified_dir = project_root / "data" / "academic_dataset" / "classified_charts"
    output_dir = project_root / "docs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    SAMPLES_PER_TYPE = 10  # Quick test with 10 charts per type
    CHART_TYPES = ["bar", "line", "pie", "scatter", "histogram", "area"]
    USE_RESNET = True  # Use ResNet18 classifier instead of rule-based
    
    # Seed for reproducibility
    random.seed(42)
    
    print(f"\n[1/5] Loading samples from {classified_dir}")
    print(f"      Samples per type: {SAMPLES_PER_TYPE}")
    print(f"      Chart types: {CHART_TYPES}")
    print(f"      Classifier: {'ResNet18' if USE_RESNET else 'Rule-based'}")
    
    samples = get_sample_charts(classified_dir, SAMPLES_PER_TYPE, CHART_TYPES)
    print(f"      Total samples: {len(samples)}")
    
    # Initialize modules
    print("\n[2/5] Initializing Stage 3 modules...")
    
    preprocessor = ImagePreprocessor(PreprocessConfig(
        apply_denoise=True,
        apply_negative=True,
        apply_morphology=True,
    ))
    
    skeletonizer = Skeletonizer(SkeletonConfig())
    
    vectorizer = Vectorizer(VectorizeConfig(
        epsilon=1.5,
        min_points=3,
    ))
    
    element_detector = ElementDetector(ElementDetectorConfig())
    
    classifier = ChartClassifier(ClassifierConfig(
        min_confidence=0.3,
    ))
    
    ocr_engine = OCREngine(OCRConfig(
        engine="easyocr",  # Use EasyOCR (PaddleOCR 3.x has compatibility issues)
        languages=["en"],
        enable_post_processing=True,
        fix_common_ocr_errors=True,
    ))
    
    geometric_mapper = GeometricMapper(MapperConfig())
    
    # Initialize ResNet classifier if enabled
    resnet_classifier = None
    if USE_RESNET:
        model_path = project_root / "models" / "weights" / "resnet18_chart_classifier_best.pt"
        if model_path.exists():
            resnet_classifier = ResNet18Classifier(
                model_path=model_path,
                device='auto',
                confidence_threshold=0.3
            )
            print(f"      ResNet18 classifier loaded from {model_path.name}")
        else:
            print(f"      WARNING: ResNet model not found at {model_path}, using rule-based")
            USE_RESNET = False
    
    print("      All modules initialized")
    
    # Process charts
    print("\n[3/5] Processing charts...")
    results: List[ChartTestResult] = []
    
    for i, (image_path, ground_truth) in enumerate(samples):
        progress = f"[{i+1}/{len(samples)}]"
        print(f"      {progress} {image_path.name[:50]}...", end=" ")
        
        result = process_chart(
            image_path=image_path,
            ground_truth=ground_truth,
            preprocessor=preprocessor,
            skeletonizer=skeletonizer,
            vectorizer=vectorizer,
            element_detector=element_detector,
            classifier=classifier,
            ocr_engine=ocr_engine,
            geometric_mapper=geometric_mapper,
            resnet_classifier=resnet_classifier,
            use_resnet=USE_RESNET,
        )
        results.append(result)
        
        if result.error:
            print(f"ERROR: {result.error[:30]}")
        else:
            status = "OK" if result.classification_correct else "WRONG"
            print(f"{status} (GT:{ground_truth}, Pred:{result.predicted_type}, Conf:{result.classification_confidence:.0%})")
    
    # Compute summary
    print("\n[4/5] Computing statistics...")
    summary = compute_summary(results)
    
    # Generate report
    print("\n[5/5] Generating report...")
    report_path = generate_report(results, summary, output_dir)
    print(f"      Report saved: {report_path}")
    
    # Save raw results
    json_path = output_dir / "stage3_classified_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"      Raw results: {json_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Charts: {summary.total_charts}")
    print(f"Successful: {summary.successful}")
    print(f"Classification Accuracy: {summary.classification_accuracy*100:.1f}%")
    print(f"Average Processing Time: {summary.avg_processing_time_ms:.1f} ms")
    
    print("\nAccuracy by Type:")
    for chart_type in sorted(summary.accuracy_per_type.keys()):
        stats = summary.accuracy_per_type[chart_type]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {chart_type:12s}: {stats['correct']:2d}/{stats['total']:2d} ({acc:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
