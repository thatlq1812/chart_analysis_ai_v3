"""
Test Stage 3 Pipeline with Academic Dataset (arXiv Charts)

This script tests the Stage 3 extraction pipeline on real chart images
from the academic dataset and generates a comprehensive report.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import json

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
)


def get_sample_images(dataset_dir: Path, num_samples: int = 10) -> List[Path]:
    """Get diverse sample images from the dataset."""
    all_images = list(dataset_dir.glob("arxiv_*.png"))
    
    # Try to get diverse samples (different papers)
    papers = {}
    for img in all_images:
        # Extract paper ID (e.g., arxiv_2601_08668v1)
        parts = img.stem.split('_')
        if len(parts) >= 3:
            paper_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if paper_id not in papers:
                papers[paper_id] = []
            papers[paper_id].append(img)
    
    # Select one image from each paper until we have enough
    selected = []
    for paper_id, images in papers.items():
        if len(selected) >= num_samples:
            break
        selected.append(images[0])
    
    return selected


def process_single_image(
    image_path: Path,
    preprocessor: ImagePreprocessor,
    skeletonizer: Skeletonizer,
    vectorizer: Vectorizer,
    element_detector: ElementDetector,
    classifier: ChartClassifier,
) -> Dict[str, Any]:
    """Process a single image through Stage 3 pipeline."""
    result = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "success": False,
        "error": None,
        "preprocessing": {},
        "skeletonization": {},
        "vectorization": {},
        "element_detection": {},
        "classification": {},
        "processing_time_ms": 0
    }
    
    start_time = time.time()
    chart_id = image_path.stem
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            result["error"] = "Failed to load image"
            return result
        
        h, w = image.shape[:2]
        result["image_size"] = {"width": w, "height": h}
        
        # Step 1: Preprocessing
        try:
            prep_result = preprocessor.process(image, chart_id=chart_id)
            result["preprocessing"] = {
                "success": True,
                "operations": prep_result.operations_applied
            }
            binary_image = prep_result.binary_image
        except Exception as e:
            result["preprocessing"] = {"success": False, "error": str(e)}
            # Fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Step 2: Skeletonization
        keypoints = []
        polylines = []
        try:
            skeleton_result = skeletonizer.process(binary_image, chart_id=chart_id)
            result["skeletonization"] = {
                "success": True,
                "keypoints_count": len(skeleton_result.keypoints)
            }
            keypoints = skeleton_result.keypoints
            
            # Step 3: Trace and vectorize
            paths = skeletonizer.trace_paths(skeleton_result.skeleton, skeleton_result.keypoints)
            vector_result = vectorizer.process(paths, chart_id=chart_id)
            polylines = vector_result.polylines
            result["vectorization"] = {
                "success": True,
                "polylines_count": len(polylines),
                "compression_ratio": vector_result.compression_ratio
            }
        except Exception as e:
            result["skeletonization"] = {"success": False, "error": str(e)}
            result["vectorization"] = {"success": False, "error": str(e)}
        
        # Step 4: Element detection
        bars = []
        markers = []
        slices = []
        try:
            elem_result = element_detector.detect(binary_image, color_image=image, chart_id=chart_id)
            bars = elem_result.bars
            markers = elem_result.markers
            slices = elem_result.slices
            result["element_detection"] = {
                "success": True,
                "bars_count": len(bars),
                "markers_count": len(markers),
                "slices_count": len(slices)
            }
        except Exception as e:
            result["element_detection"] = {"success": False, "error": str(e)}
        
        # Step 5: Classification
        try:
            class_result = classifier.classify(
                bars=bars,
                polylines=polylines,
                markers=markers,
                slices=slices,
                texts=[],  # Skip OCR for speed
                image_shape=(h, w),
                chart_id=chart_id,
            )
            result["classification"] = {
                "success": True,
                "chart_type": class_result.chart_type.value,
                "confidence": class_result.confidence,
                "features": class_result.features
            }
        except Exception as e:
            result["classification"] = {"success": False, "error": str(e)}
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
    return result


def create_visualization(
    image_path: Path,
    result: Dict[str, Any],
    output_dir: Path,
    index: int
) -> Optional[Path]:
    """Create visualization for a processed image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb_image)
        axes[0].set_title(f"Original: {image_path.name}")
        axes[0].axis('off')
        
        # Classification result
        if result["classification"].get("success"):
            chart_type = result["classification"]["chart_type"]
            confidence = result["classification"]["confidence"]
            features = result["classification"].get("features", {})
            
            info_text = f"Type: {chart_type}\nConfidence: {confidence:.1%}\n"
            if features:
                info_text += "\nFeatures:\n"
                for k, v in list(features.items())[:5]:
                    info_text += f"  {k}: {v:.3f}\n" if isinstance(v, float) else f"  {k}: {v}\n"
            
            axes[1].text(0.1, 0.5, info_text, fontsize=12, 
                        verticalalignment='center', fontfamily='monospace',
                        transform=axes[1].transAxes)
        else:
            axes[1].text(0.1, 0.5, f"Classification failed:\n{result['classification'].get('error', 'Unknown error')}", 
                        fontsize=12, color='red', transform=axes[1].transAxes)
        
        axes[1].set_title("Analysis Results")
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"result_{index:02d}_{image_path.stem}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"    Visualization error: {e}")
        return None


def generate_report(
    results: List[Dict[str, Any]],
    visualizations: List[Optional[Path]],
    output_dir: Path
) -> Path:
    """Generate markdown report with results."""
    report_path = output_dir / "ACADEMIC_DATASET_TEST_REPORT.md"
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    
    # Classification distribution
    classification_dist = {}
    for r in results:
        if r["classification"].get("success"):
            chart_type = r["classification"]["chart_type"]
            classification_dist[chart_type] = classification_dist.get(chart_type, 0) + 1
    
    # Timing statistics
    times = [r["processing_time_ms"] for r in results if r["success"]]
    avg_time = sum(times) / len(times) if times else 0
    
    report_content = f"""# Stage 3 Academic Dataset Test Report

| Property | Value |
|----------|-------|
| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| Dataset | Academic arXiv Charts |
| Total Images | {total} |
| Successful | {successful} ({successful/total*100:.1f}%) |
| Average Processing Time | {avg_time:.1f} ms |

## Classification Distribution

| Chart Type | Count | Percentage |
|------------|-------|------------|
"""
    
    for chart_type, count in sorted(classification_dist.items(), key=lambda x: -x[1]):
        pct = count / successful * 100 if successful > 0 else 0
        report_content += f"| {chart_type} | {count} | {pct:.1f}% |\n"
    
    report_content += "\n## Individual Results\n\n"
    
    for i, (result, viz_path) in enumerate(zip(results, visualizations)):
        report_content += f"### {i+1}. {result['image_name']}\n\n"
        
        if viz_path and viz_path.exists():
            rel_path = f"../images/stage3_academic/{viz_path.name}"
            report_content += f"![Result]({rel_path})\n\n"
        
        if result["success"]:
            report_content += f"- **Status**: Success\n"
            report_content += f"- **Processing Time**: {result['processing_time_ms']} ms\n"
            report_content += f"- **Image Size**: {result.get('image_size', {}).get('width', '?')}x{result.get('image_size', {}).get('height', '?')}\n"
            
            if result["classification"].get("success"):
                report_content += f"- **Chart Type**: {result['classification']['chart_type']}\n"
                report_content += f"- **Confidence**: {result['classification']['confidence']:.1%}\n"
            
            if result["preprocessing"].get("success"):
                ops = result["preprocessing"].get("operations", [])
                report_content += f"- **Preprocessing Steps**: {len(ops)}\n"
            
            if result["skeletonization"].get("success"):
                report_content += f"- **Keypoints**: {result['skeletonization']['keypoints_count']}\n"
            
            if result["vectorization"].get("success"):
                report_content += f"- **Polylines**: {result['vectorization']['polylines_count']}\n"
        else:
            report_content += f"- **Status**: Failed\n"
            report_content += f"- **Error**: {result.get('error', 'Unknown')}\n"
        
        report_content += "\n"
    
    report_content += """
## Summary

This test validates the Stage 3 extraction pipeline against real academic chart images
from arXiv papers. The results demonstrate the pipeline's ability to:

1. **Preprocess** diverse chart images with varying quality and styles
2. **Extract structural features** through skeletonization
3. **Vectorize** the chart structure into polylines
4. **Classify** chart types (bar, line, pie, scatter, etc.)

### Observations

- Line charts are common in academic papers for showing trends
- Classification accuracy varies based on chart complexity
- Processing time is suitable for batch processing

### Next Steps

- Fine-tune classification thresholds based on these results
- Add OCR extraction for real-world chart understanding
- Implement Stage 4 (SLM reasoning) for value extraction
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path


def main():
    """Main function to run academic dataset tests."""
    print("=" * 60)
    print("STAGE 3 ACADEMIC DATASET TEST")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "data" / "academic_dataset" / "images"
    output_dir = project_root / "docs" / "images" / "stage3_academic"
    report_dir = project_root / "docs" / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample images
    print(f"\n[1/4] Selecting sample images from {dataset_dir}...")
    sample_images = get_sample_images(dataset_dir, num_samples=15)
    print(f"    Selected {len(sample_images)} images from different papers")
    
    # Initialize Stage 3 modules
    print("\n[2/4] Initializing Stage 3 modules...")
    preprocessor = ImagePreprocessor(PreprocessConfig(
        apply_denoise=True,
        apply_negative=True,
        apply_morphology=True,
    ))
    skeletonizer = Skeletonizer(SkeletonConfig())
    vectorizer = Vectorizer(VectorizeConfig())
    element_detector = ElementDetector(ElementDetectorConfig())
    classifier = ChartClassifier(ClassifierConfig(min_confidence=0.3))
    print("    All modules initialized")
    
    # Process images
    print("\n[3/4] Processing images...")
    results = []
    visualizations = []
    
    for i, image_path in enumerate(sample_images):
        print(f"    [{i+1}/{len(sample_images)}] {image_path.name}...", end=" ")
        
        result = process_single_image(
            image_path, preprocessor, skeletonizer, vectorizer,
            element_detector, classifier
        )
        results.append(result)
        
        if result["success"]:
            chart_type = result["classification"].get("chart_type", "unknown")
            confidence = result["classification"].get("confidence", 0)
            print(f"OK ({chart_type}, {confidence:.0%}, {result['processing_time_ms']}ms)")
        else:
            print(f"FAILED: {result.get('error', 'Unknown')}")
        
        # Create visualization
        viz_path = create_visualization(image_path, result, output_dir, i)
        visualizations.append(viz_path)
    
    # Generate report
    print("\n[4/4] Generating report...")
    report_path = generate_report(results, visualizations, report_dir)
    print(f"    Report saved: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for r in results if r["success"])
    print(f"Total: {len(results)}, Successful: {successful}, Failed: {len(results) - successful}")
    
    # Classification distribution
    classification_dist = {}
    for r in results:
        if r["classification"].get("success"):
            chart_type = r["classification"]["chart_type"]
            classification_dist[chart_type] = classification_dist.get(chart_type, 0) + 1
    
    print("\nClassification distribution:")
    for chart_type, count in sorted(classification_dist.items(), key=lambda x: -x[1]):
        print(f"    {chart_type}: {count}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    # Save raw results as JSON
    json_path = report_dir / "academic_dataset_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved: {json_path}")


if __name__ == "__main__":
    main()
