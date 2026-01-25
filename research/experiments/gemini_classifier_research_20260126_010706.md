# Gemini Research: Chart Classifier Improvement

**Model**: gemini-3-flash-preview

**Generated**: 2026-01-26T01:07:06.239712

---

This analysis addresses the "Synthetic-to-Real" gap in your Geo-SLM system. The 37.5% accuracy on real charts is a classic case of **domain shift**: rule-based systems designed for "clean" synthetic data cannot handle the entropy (grid lines, anti-aliasing, overlapping text) of academic publications.

---

## Part 1: Literature Review (2020-2024)

Recent SOTA methods have moved away from pure heuristics toward **Vision Transformers (ViT)** and **Multi-modal encoders**.

| Method / Paper | Approach | Key Innovation | Accuracy (Avg) |
| :--- | :--- | :--- | :--- |
| **ChartViT** (2023) | Transformer | Uses a Vision Transformer backbone specifically fine-tuned on chart structures. | ~93% on ChartQA |
| **UniChart** (2023) | Multi-task | A universal encoder for cropping, classification, and data extraction. | ~90%+ |
| **MatplotAgent** (2024) | LLM-Vision | Uses code-generation feedback loops to understand chart structure. | High (but slow) |
| **Revision** (2020) | CNN | Uses a modified ResNet architecture with a "Chart-specific" preprocessing layer. | ~88% |

**Key Takeaway:** The industry has converged on **Transfer Learning** using ImageNet-pretrained models (ResNet/EfficientNet) fine-tuned on large-scale synthetic datasets (like FigureQA or PlotQA) before being exposed to real-world data.

---

## Part 2: Feature Engineering

To move beyond simple Sobel filters, you need features that capture **topology** rather than just **edges**.

1.  **Connectivity Analysis (The Line vs. Scatter Fix):**
    *   **Feature:** Ratio of "Number of connected components" to "Total black pixels."
    *   **Logic:** A scatter plot has many small, isolated components. A line chart has a few very long, continuous components.
2.  **Frequency Domain (FFT):**
    *   **Feature:** Magnitude spectrum.
    *   **Logic:** Bar charts have strong periodic peaks in the FFT due to regular spacing of bars.
3.  **Text Masking (Crucial):**
    *   Use your existing **PaddleOCR** to generate bounding boxes for text. **Black out these regions** before passing the image to the classifier. This prevents the classifier from "learning" that certain fonts or words (e.g., "Figure 1") correlate with chart types.
4.  **Hough Transform Refinement:**
    *   Instead of just counting lines, calculate the **ratio of horizontal to vertical lines**. Bar charts are dominated by one or the other; line charts have diagonal components.

---

## Part 3: Architecture Recommendations

**Recommended: Hybrid Transfer Learning (Fine-tuned EfficientNet-B0)**

*   **Why:** 
    *   **Size:** ~20MB (well under 500MB).
    *   **Speed:** <30ms on GPU (well under 500ms).
    *   **Explainability:** Use **Grad-CAM** (Gradient-weighted Class Activation Mapping) to produce a heatmap showing *which pixels* led to the classification (e.g., highlighting the bars or the line).

| Architecture | Pros | Cons |
| :--- | :--- | :--- |
| **Improved Rules** | Fully explainable | Brittle, fails on noisy academic PDFs. |
| **EfficientNet-B0** | **Best balance of speed/accuracy** | Requires GPU training. |
| **Vision Transformer** | Highest accuracy | Slower inference, larger memory footprint. |

---

## Part 4: Training Strategy

With 2,852 images, you are in the "Goldilocks" zone for fine-tuning.

1.  **Data Augmentation (The "Real-World" Simulator):**
    *   **Gaussian Noise & Blur:** Simulate low-quality PDF renders.
    *   **Random Erasing:** Mimic overlapping labels or legends.
    *   **Grid-line Injection:** Procedurally add light gray horizontal/vertical lines to your synthetic data to "teach" the model to ignore them.
2.  **Handling Imbalance (Pie Charts 3.4%):**
    *   **Oversampling:** Duplicate pie chart samples.
    *   **Weighted Cross-Entropy Loss:** Penalize the model more for missing a Pie chart than a Scatter plot.
3.  **Split:** 80% Train / 10% Val / 10% Test (Stratified by class).

---

## Part 5: Implementation Roadmap (2-Week Sprint)

### Week 1: Data & Baseline (High Impact)
1.  **Pre-processing Pipeline:** Create a script that: (a) Resizes to 224x224, (b) Masks OCR text, (c) Converts to Grayscale (optional, but helps focus on structure).
2.  **Baseline Model:** Fine-tune `torchvision.models.efficientnet_b0` (pretrained on ImageNet).
3.  **Synthetic Mixing:** Mix your 2,852 real charts with 5,000 synthetic charts (easily generated via Matplotlib) to stabilize training.

### Week 2: Refinement & Explainability
1.  **Error Analysis:** Run the model on the 62.5% failed cases.
2.  **Grad-CAM Integration:** Implement Grad-CAM to visualize failures.
3.  **Optimization:** Quantize the model to `INT8` using PyTorch Mobile/ONNX for even faster local inference.

---

## Part 6: Error Analysis Insights (Line vs. Scatter)

**The Problem:** Your rule-based system sees "markers" (the dots on a line chart) as isolated circles. Because there are many of them, the "Scatter" rule triggers first.

**The Solution (Code Logic):**
To separate them, use **Morphological Closing** to bridge the gaps between markers.

```python
import cv2
import numpy as np

def distinguish_line_vs_scatter(image):
    # 1. Preprocess: Threshold to binary
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Count initial components
    num_labels_orig, _ = cv2.connectedComponents(binary)
    
    # 3. Apply Morphological Closing (connects dots near each other)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. Count components after closing
    num_labels_closed, _ = cv2.connectedComponents(closed)
    
    # 5. Logic:
    # If the number of components drops drastically (e.g., > 70% reduction),
    # it means many small dots were connected into a few lines.
    reduction_ratio = num_labels_closed / num_labels_orig
    
    if reduction_ratio < 0.3:
        return "Line Chart (Markers Connected)"
    else:
        return "Scatter Plot (Isolated Points)"
```

### Final Recommendation for Geo-SLM:
Don't rely on a single "Mega-Model." Use a **Two-Stage Pipeline**:
1.  **Stage 1 (CNN):** EfficientNet-B0 provides a probability distribution (e.g., Line: 80%, Scatter: 20%).
2.  **Stage 2 (Geometric Verification):** If the CNN is "unsure" (top two classes within 15%), trigger the **Connectivity Analysis** code above to break the tie. This maintains explainability for your thesis while hitting the accuracy target.