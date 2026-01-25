# Gemini Research: Chart Classifier Improvement

**Model**: gemini-3-flash-preview

**Generated**: 2026-01-26T01:09:49.252684

---

This analysis provides a comprehensive strategy to transition from a failing rule-based system to a robust, production-grade chart classifier for the Geo-SLM project.

---

## Part 1: Literature Review (2020-2024)

Recent research has shifted from pure geometric analysis to **Deep Visual-Semantic** approaches.

1.  **Chart-to-Text (2022):** (Kantharaj et al.) Uses a hybrid approach. It highlights that chart classification is the foundational step for downstream summarization.
2.  **UniChart (2023):** A universal vision-language model for charts. While too large for your constraints, its pre-training strategy (using synthetic data to learn structure) is a gold standard.
3.  **Matplotlib-based Synthetic Generation (2021):** (Luo et al.) Demonstrated that training on diverse synthetic data with "realistic noise" (varying fonts, grid styles) is better than "clean" synthetic data.
4.  **Key Architectures:**
    *   **CNNs:** ResNet and EfficientNet remain the workhorses for classification due to speed.
    *   **Vision Transformers (ViT):** (e.g., DeiT) Show higher accuracy on complex layouts but require more data and are slower.
    *   **Accuracies:** State-of-the-art (SOTA) on datasets like *PMC-Figures* or *Chart-to-Text* typically ranges from **92% to 96%**.

---

## Part 2: Feature Engineering

To move beyond 37%, we must address the "Scatter Bias."

*   **Low-Level (Connectivity):** The primary difference between a line chart and a scatter plot is **pixel connectivity**. Use morphological "Closing" operations to see if markers merge into a continuous path.
*   **Mid-Level (Layout):**
    *   **Axis Detection:** Bar charts usually have one categorical axis and one numerical axis.
    *   **Legend Analysis:** Pie charts rarely have axes but almost always have a legend with color patches.
*   **High-Level (OCR Semantic Features):**
    *   Extract text via PaddleOCR. If the X-axis contains "Years" or "Months," it is likely a **Line/Bar** chart. If both axes are numeric with high variance, it's likely a **Scatter** plot.
*   **Noise Handling:** Use **Canny Edge Detection** followed by a **Hough Transform** to identify and "mask out" grid lines before classification.

---

## Part 3: Architecture Recommendations

**Recommendation: Hybrid Lightweight CNN (EfficientNet-B0 + Metadata)**

| Architecture | Speed | Size | Accuracy Potential | Explainability |
| :--- | :--- | :--- | :--- | :--- |
| **Improved Rule-based** | <10ms | <1MB | Low (60%) | High |
| **EfficientNet-B0** | ~30ms | ~20MB | **High (90%+)** | Med (Grad-CAM) |
| **ViT-Tiny** | ~60ms | ~30MB | High (88%) | Low |

**Why EfficientNet-B0?**
- It fits well within your <500MB and <500ms constraints.
- It handles the "texture" of academic charts (stipple patterns, cross-hatching) better than ResNet.
- **Explainability:** Use **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize which part of the chart (e.g., the bars or the legend) the model is looking at.

---

## Part 4: Training Strategy

With 2,852 images, you are in the "Data Hungry" zone.

1.  **Data Augmentation (Crucial):**
    *   **Geometric:** Random cropping (to simulate zoomed-in charts), slight rotations.
    *   **Photometric:** Add Gaussian noise and "Salt & Pepper" noise to mimic scanned paper textures.
    *   **Simulated Grid Lines:** Randomly overlay grid lines on synthetic data to force the model to ignore them.
2.  **Handling Imbalance (Pie Charts at 3.4%):**
    *   Use **WeightedRandomSampler** in PyTorch to oversample Pie charts during training.
    *   Use **Focal Loss** instead of Cross-Entropy to penalize errors on the rare classes more heavily.
3.  **Transfer Learning:**
    *   Start with weights pre-trained on ImageNet.
    *   **Fine-tuning:** Freeze the first 70% of layers and train the "head" first, then unfreeze all layers with a very low learning rate ($1e-5$).

---

## Part 5: Implementation Roadmap (2-Week Sprint)

### Week 1: The Baseline Shift
*   **Day 1-2:** Data Cleaning. Remove corrupted images. Split data: 70% Train / 15% Val / 15% Test (Stratified).
*   **Day 3-4:** Implement **EfficientNet-B0** in PyTorch. Set up a training pipeline with `torchvision.transforms`.
*   **Day 5:** Run baseline training. Target: >70% accuracy.

### Week 2: Optimization & Refinement
*   **Day 6-7:** Error Analysis. Use a Confusion Matrix. Identify why Line/Scatter are still mixing.
*   **Day 8-9:** **Feature Fusion.** Feed OCR metadata (e.g., "is_numeric_x_axis") into the final fully connected layer of the CNN.
*   **Day 10-11:** Hyperparameter tuning (Learning rate, Batch size).
*   **Day 12:** Implement Grad-CAM for the thesis "Explainability" requirement.
*   **Day 14:** Final Evaluation and Export to ONNX for <50ms inference.

---

## Part 6: Error Analysis Insights (Line vs. Scatter)

**The Problem:** Line charts with markers look identical to scatter plots to a global CNN.

**The Solution (Discriminative Features):**
1.  **Skeletonization:** Apply a skeletonization algorithm. A line chart will result in a long, continuous skeleton. A scatter plot will result in many disjointed points.
2.  **Hough Line Transform:** Line charts have strong local gradients connecting points.
3.  **Code Snippet for Diagnosis:**

```python
import cv2
import numpy as np

def check_connectivity(image_path):
    # Load, grayscale, and threshold
    img = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing to bridge small gaps in lines
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Count connected components
    num_labels, _ = cv2.connectedComponents(closed)
    
    # Line charts: Few, large components
    # Scatter plots: Many, small components
    return num_labels

# If num_labels > 50 and avg_component_size is small -> Scatter
# If num_labels < 10 and components are long -> Line
```

### Final Recommendation for Geo-SLM:
Don't rely on rules alone. Use **EfficientNet-B0** as your backbone. It will learn the "concept" of a line vs. a bar much better than a Hough transform ever will. Use the rule-based logic only as a **pre-processing filter** to remove grid lines and text before the image hits the CNN.