# Gemini Research: Chart Classifier Improvement

**Model**: gemini-3-pro-preview

**Generated**: 2026-01-26T01:11:17.337416

---

This is a classic computer vision transition problem: moving from brittle heuristics (rule-based) to robust statistical learning (deep learning) when facing real-world noise.

Given your constraints (local GPU, <500ms latency, <500MB size) and your dataset size (~2.8k images), **Transfer Learning with a Lightweight CNN** is the optimal path. You do not need a massive Transformer model to distinguish a pie chart from a bar chart.

Here is the detailed analysis and roadmap.

---

## Part 1: Literature Review (2020-2024)

Recent research has moved almost entirely away from rule-based systems for chart classification due to the exact issues you are facing (noise, grid lines, styling variations).

1.  **Deep Learning Baselines (The Standard):**
    *   **Paper:** *DocFigure* and *Chart-Text* benchmarks.
    *   **Method:** Standard CNNs (ResNet-50, VGG-16, DenseNet-121) pretrained on ImageNet and fine-tuned on chart data.
    *   **Accuracy:** Consistently **90-95%** on datasets like PMC-OA (PubMed Central).
    *   **Relevance:** This confirms that standard CNNs are sufficient for this task.

2.  **Multimodal Approaches (SOTA):**
    *   **Paper:** *ChartBERT* or *DoT (Document Layout Analysis with Transformers)*.
    *   **Method:** Combines image data (pixels) with OCR data (text labels/legends).
    *   **Accuracy:** **96-98%**.
    *   **Relevance:** While more accurate, these are slower and heavier. Given your 80% target, a visual-only approach is faster and sufficient.

3.  **Object Detection Approaches:**
    *   **Method:** Using YOLO/Faster-RCNN to detect chart *components* (bars, axis, legend) to infer type.
    *   **Relevance:** You already have YOLO. If YOLO detects "bars," it's a bar chart. If it detects "slice," it's a pie chart. However, training YOLO for fine-grained elements (like "line segment") is harder than simple image classification.

---

## Part 2: Feature Engineering (Why Rules Fail)

Your current failure is due to **hand-crafted features** failing to generalize.

### Why Line Charts Fail (The "Scatter" Bias)
*   **Current Feature:** Likely looking for "connected components" or "hough lines."
*   **Reality:** Real line charts often have **markers** (dots, triangles) at data points. A rule-based system sees 20 dots and thinks "Scatter." It fails to weigh the thin connecting lines heavily enough against the grid lines (noise).

### Effective Visual Features (Learned by CNNs)
Instead of manually coding Sobel filters, a CNN learns:
1.  **Connectivity (Line vs. Scatter):** Convolutional filters detect continuous paths (lines) versus isolated blobs (scatter).
2.  **Rectilinearity (Bar vs. Line):** Detects solid blocks of uniform color/texture (bars) vs. thin paths.
3.  **Polarity (Pie vs. Others):** Detects curvature and radial edges.

### Handling Noise
*   **Grid Lines:** A CNN learns to ignore high-frequency, repetitive background lines because they don't correlate with the class label during training.
*   **Text:** CNNs generally treat text as texture.

---

## Part 3: Architecture Recommendations

**Recommendation: Transfer Learning with ResNet-18 or EfficientNet-B0.**

| Feature | **ResNet-18** (Recommended) | EfficientNet-B0 | Rule-Based (Current) |
| :--- | :--- | :--- | :--- |
| **Accuracy Potential** | High (>90%) | High (>92%) | Low (<40% on real) |
| **Inference Speed** | ~10-20ms (GPU) | ~15-25ms (GPU) | ~50ms |
| **Model Size** | ~45 MB | ~20 MB | < 1 MB |
| **Training Data** | Needs ~1k+ images | Needs ~1k+ images | N/A |
| **Explainability** | Medium (Grad-CAM) | Medium (Grad-CAM) | High |

**Why ResNet-18?**
1.  **Speed:** Well under your 500ms limit (usually <20ms on a modern GPU).
2.  **Size:** ~45MB file size (fits your <500MB constraint easily).
3.  **Availability:** Native in PyTorch (`torchvision.models`).
4.  **Robustness:** The residual connections make it very good at learning geometric shapes without vanishing gradients.

---

## Part 4: Training Strategy

You have 2,852 images. This is enough for Transfer Learning, but you must handle the class imbalance.

### 1. Data Split
*   **Train:** 70% (~2,000 images)
*   **Validation:** 15% (~425 images) - *Use this to tune hyperparameters.*
*   **Test:** 15% (~425 images) - *Do not touch until final evaluation.*

### 2. Handling Imbalance (Line: 31% vs Pie: 3.4%)
If you don't fix this, the model will ignore Pie charts.
*   **Technique A: Weighted Random Sampling (Recommended):** Over-sample the minority classes (Pie) in the DataLoader so the model sees an equal number of Line and Pie charts per epoch.
*   **Technique B: Loss Weighting:** Penalize the model more for missing a Pie chart than a Line chart in the CrossEntropyLoss.

### 3. Data Augmentation (Crucial for Generalization)
Since you only have ~2.8k images, you must augment to prevent overfitting to specific fonts or colors.
*   **Grayscale:** Convert 20% of images to B/W during training (academic papers are often B/W).
*   **Gaussian Blur:** Simulate low-resolution scans.
*   **Color Jitter:** Change brightness/contrast slightly.
*   **Affine:** Slight rotations (+/- 5 degrees) and shearing. *Do not flip* (charts have direction).

---

## Part 5: Implementation Roadmap (2 Weeks)

### Week 1: The Pivot to Deep Learning

**Day 1: Data Prep**
*   Organize data into folders: `train/bar`, `train/line`, etc.
*   Implement the PyTorch `Dataset` and `DataLoader` with **WeightedRandomSampler**.

**Day 2: Model Setup (The "MLChartClassifier")**
*   Load `resnet18(pretrained=True)`.
*   Replace the final Fully Connected layer: `model.fc = nn.Linear(512, num_classes)`.
*   Freeze the early layers (feature extractor) and train only the head for 5 epochs.

**Day 3: Full Training**
*   Unfreeze all layers.
*   Train for 20-30 epochs with a low learning rate (e.g., 1e-4).
*   **Goal:** Reach >85% on Validation set.

**Day 4: Integration**
*   Wrap the model in a Python class.
*   Add image preprocessing (Resize to 224x224, Normalize).
*   Run inference on your 80 "hard" real charts.

**Day 5: Error Analysis & Tuning**
*   Check Confusion Matrix.
*   If Line/Scatter confusion persists, increase resolution (e.g., 448x448) or add specific augmentations (erasing random patches).

### Week 2: Refinement & Explainability

**Day 6-7: Optimization**
*   Convert model to ONNX format (faster inference on CPU/GPU).
*   Ensure latency is <50ms.

**Day 8-9: Explainability (Thesis Requirement)**
*   Implement **Grad-CAM**.
*   This generates a heatmap overlay showing *what* the model looked at.
*   *Example:* For a Pie chart, the heatmap should light up the circle. For a Bar chart, it should light up the rectangular bars. This restores the "explainability" lost by moving away from rules.

**Day 10: Final Benchmarking**
*   Run on the held-out Test set.
*   Document results.

---

## Part 6: Error Analysis Insights & Code

### Why Line Charts = Scatter? (Diagnosis)
In your current rule-based system, you likely use edge detection.
1.  **Scatter:** High edge density, no long continuous lines.
2.  **Line:** High edge density (markers), long continuous lines.
3.  **The Bug:** If the lines are thin or dashed, edge detection breaks them. The system sees "broken lines" + "markers" = "Scatter."

### The Solution Code (PyTorch Skeleton)

Here is the code to replace your `SimpleChartClassifier` with a robust CNN.

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class MLChartClassifier:
    def __init__(self, model_path=None, num_classes=4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.classes = ['bar', 'line', 'pie', 'scatter'] # Ensure order matches training
        
        # 1. Load Architecture (ResNet18)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Modify Head for Chart Classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # 3. Load Weights (if trained)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        # 4. Define Preprocessing (Must match training)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        """
        Returns class name and confidence score.
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        return self.classes[predicted_idx.item()], confidence.item()

# --- Training Snippet (Conceptual) ---
# criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Use WeightedRandomSampler in your DataLoader to handle the 3.4% Pie chart imbalance!
```

### Explainability (Grad-CAM)
To satisfy your thesis requirement, do not just output "Bar Chart." Output the image with a heatmap overlay.
*   **Library:** `pip install grad-cam`
*   **Usage:** It hooks into the last convolutional layer of ResNet.
*   **Result:** It proves the model isn't cheating (e.g., looking at the title text) but is actually looking at the geometric structure of the data.

### Summary Recommendation
Stop tuning the Sobel/Hough parameters. It is a dead end for noisy academic charts. Switch to the **ResNet-18** approach immediately. With 2,800 images, you will likely hit **90%+ accuracy** within the first few hours of training.