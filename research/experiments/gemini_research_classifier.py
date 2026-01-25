"""
Research Script: Gemini-powered Analysis for Classifier Improvement

Uses Gemini API to research and recommend solutions for improving
chart classification accuracy on real academic charts.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Use Deep Research model for comprehensive analysis
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "deep-research-pro-preview-12-2025")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

# Problem context
PROBLEM_CONTEXT = """
# Chart Classification Problem

## Current Situation
We are building a chart analysis system (Geo-SLM) that combines computer vision with geometric analysis.

### Current Classifier: SimpleChartClassifier (Rule-based)
- **Method**: Image-based heuristics
- **Features**:
  1. Edge orientation analysis (Sobel + Hough lines)
  2. Color distribution (saturation variance)
  3. Circular detection (Hough circles for pie charts)
  4. Aspect ratio
  
- **Performance**:
  - Synthetic data: 100% accuracy [PASS]
  - Real academic charts: 37.5% accuracy [FAIL]
  
### Failure Analysis (80 real charts):
- Bar charts: 40% (8/20) [FAIL]
- Line charts: 0% (0/20) [CATASTROPHIC FAIL - all classified as scatter]
- Pie charts: 25% (5/20) [FAIL]
- Scatter plots: 85% (17/20) [PASS but causes false positives]

### Key Problem: 
**SEVERE BIAS toward "scatter" classification** - most charts misclassified as scatter.

### Root Causes Identified:
1. Real charts have noise, grid lines, text, legends → rule-based features fail
2. Synthetic data is too clean → overfitting to synthetic patterns
3. Line charts have dense markers → confused with scatter plots
4. Current features not discriminative enough

## Available Resources
1. **Data**: 2,852 academic chart images with ground truth labels
2. **Hardware**: Local GPU (CUDA available)
3. **Models Available**:
   - SimpleChartClassifier (current, rule-based)
   - MLChartClassifier (skeleton, not trained)
   - YOLO for chart detection (trained, working)
   - PaddleOCR for text extraction (working)
   
4. **Tech Stack**: Python, PyTorch, OpenCV, scikit-learn

## Constraints
- Must run locally (no cloud API dependencies for core inference)
- Inference speed: < 500ms per chart (current: ~50ms)
- Model size: < 500MB (for deployment)
- Maintain explainability (for academic thesis)

## Objective
Need a classification approach that achieves **>80% accuracy** on real academic charts.
"""

RESEARCH_QUERY = """
Based on the problem context above, please research and recommend:

## Part 1: Literature Review
What are the state-of-the-art methods for chart classification in recent papers (2020-2024)?
Include:
- Method names and key papers
- Approaches (CNN-based, transformer-based, hybrid)
- Benchmark datasets used
- Reported accuracies

## Part 2: Feature Engineering
What visual features are most effective for distinguishing chart types?
Consider:
- Low-level features (edges, colors, shapes)
- Mid-level features (layout, spatial patterns)
- High-level features (semantic understanding)
- How to handle noise, text, grid lines?

## Part 3: Architecture Recommendations
For our constraint (local inference, <500ms, <500MB), which architecture would you recommend?
Compare:
1. **Improved Rule-based**: Better features + ensemble
2. **Lightweight CNN**: ResNet-18, MobileNet, EfficientNet-Lite
3. **Hybrid**: Rules for easy cases + CNN for hard cases
4. **Transfer Learning**: Fine-tune pretrained model

## Part 4: Training Strategy
Given 2,852 labeled charts, what's the optimal training approach?
Consider:
- Data augmentation techniques
- Train/val/test split
- Handling class imbalance (line: 31%, bar: 21%, pie: 3.4%)
- Few-shot learning for rare types
- Curriculum learning (easy→hard)

## Part 5: Implementation Roadmap
Provide a step-by-step plan to achieve >80% accuracy within 2 weeks.
Priority: high-impact, low-effort improvements first.

## Part 6: Error Analysis Insights
How to diagnose why line charts are ALL misclassified as scatter?
What discriminative features separate line vs scatter?

Please provide actionable, concrete recommendations with code examples where applicable.
"""

def research_with_gemini(query: str, context: str, model_name: str = None) -> dict:
    """Query Gemini for research insights."""
    
    model_name = model_name or GEMINI_MODEL
    print(f"[INFO] Querying {model_name}...")
    print(f"[INFO] Query length: {len(query)} chars")
    
    model = genai.GenerativeModel(model_name)
    
    # Combine context and query
    full_prompt = f"{context}\n\n{'='*60}\n\n{query}"
    
    # Generate response
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,  # Lower for factual research
            max_output_tokens=32000,  # Deep Research can output more
        )
    )
    
    return {
        "model": model_name,
        "prompt_length": len(full_prompt),
        "response": response.text,
        "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
    }

def save_research_results(results: dict, output_path: Path):
    """Save research results to markdown and JSON."""
    
    # Save markdown report
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gemini Research: Chart Classifier Improvement\n\n")
        f.write(f"**Model**: {results['model']}\n\n")
        f.write(f"**Generated**: {results.get('timestamp', 'N/A')}\n\n")
        f.write("---\n\n")
        f.write(results["response"])
    
    # Save JSON metadata
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": results["model"],
            "prompt_length": results["prompt_length"],
            "finish_reason": str(results["finish_reason"]),
            "timestamp": results.get("timestamp"),
        }, f, indent=2)
    
    print(f"\n[SUCCESS] Research saved:")
    print(f"   Report: {md_path}")
    print(f"   Metadata: {json_path}")

def main():
    """Run Gemini research."""
    from datetime import datetime
    
    print("=" * 70)
    print("GEMINI-POWERED RESEARCH: CLASSIFIER IMPROVEMENT")
    print("=" * 70)
    
    # Use Gemini 3 Pro for advanced reasoning
    model_name = "gemini-3-pro-preview"
    print(f"\nUsing Model: {model_name}")
    
    # Run research
    results = research_with_gemini(RESEARCH_QUERY, PROBLEM_CONTEXT, model_name=model_name)
    results["timestamp"] = datetime.now().isoformat()
    
    # Save results
    output_path = project_root / "research" / "experiments" / f"gemini_classifier_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_research_results(results, output_path)
    
    print("\n" + "=" * 70)
    print("Review the research report and implement recommendations!")
    print("=" * 70)

if __name__ == "__main__":
    main()
