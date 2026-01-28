Stage 3 (Extraction) Enhancements

Stage 3 of the Geo-SLM pipeline converts a detected chart image into structured data. Current Stage 3 submodules include image preprocessing, skeletonization, vectorization, OCR with text-role tagging, geometric calibration, element detection, and chart classification. Below we review each submodule’s method, analyze its limitations, and propose improvements (symbolic, neural, or hybrid) to boost accuracy and robustness. We also discuss how these enhancements feed into Stage 4 (SLM reasoning) and leverage the ArXiv chart dataset (with provided type counts) for training/evaluation. Finally, we outline an evaluation strategy for overall extraction fidelity.

Image Preprocessing

Current Method: The pipeline converts the input chart to grayscale, optionally applies a negative transform, blurs to reduce noise, then applies adaptive thresholding and morphological operations to produce a binary image. (Specifically, it inverts “light-on-dark” charts by computing 255 - pixel, then uses Gaussian blur and adaptive thresholding followed by optional open/close operations.)

Limitations: This fixed recipe can break on diverse chart styles. For example, colored or gradient backgrounds, anti‑aliased lines, or low-contrast elements may not binarize cleanly. Global threshold parameters may fail on charts with uneven illumination. Rigid inversion logic (“apply negative if” by rule) may mis-handle charts with atypical color schemes. As Luo et al. note, pure rule-based chart preprocessing cannot handle wide style variation.

Proposed Enhancements: We recommend adding learned or adaptive components. Possible improvements include:

Neural segmentation: Train a U-Net or similar CNN to segment chart foreground (lines, shapes, text) from background, obviating manual threshold tuning. Such models can learn to ignore textured backgrounds and shadows.

Adaptive color processing: Instead of binary inversion, use color clustering or channel-wise thresholding to separate ink from paper. For example, detect bright vs dark themes automatically (e.g. by checking histogram) rather than fixed “top 15%” rules.

Denoising autoencoders: Apply CNN-based denoising (e.g. a small background-removal network) to remove artifacts before binarization.

Data augmentation: During training of any learned modules, include diverse chart images (different themes, noise, resolutions) to improve generalization.
These enhancements draw on hybrid approaches: ChartOCR showed that combining deep frameworks with rule-based logic generalizes well across chart styles, so a hybrid preprocessing model (deep+heuristic) could adapt to many charts.

Skeletonization

Current Method: The binary image is thinned to a one-pixel skeleton using Lee’s topology-preserving algorithm. This preserves connectivity, junctions, and endpoints (data points), with optional spur pruning.

Limitations: Simple thinning is sensitive to noise and can create spurious spurs from small artifacts. Thick or curved regions (e.g. filled pie slices or area plots) may skeletonize poorly or break. Any mis-binarization upstream (e.g. broken lines) leads to disconnected skeletons. Spur removal heuristics (minimum length 5px by default) may still leave clutter or inadvertently remove fine detail.

Proposed Enhancements: We suggest combining classical morphology with learned refinement. Ideas include:

Improved thinning algorithms: Explore other thinning variants (e.g. Guo-Hall) or the Zhang-Suen algorithm, which may better handle diagonal connectivity.

Morphological preprocessing: Before thinning, use morphological closing to fill tiny gaps in lines (small dilation followed by erosion) to stabilize skeleton continuity.

Neural skeleton refinement: Use a small CNN (or U-Net) trained on chart skeletons to clean up the result – e.g. remove noisy branches, enforce straightness for lines. Some research indicates that CNNs can learn to extract object skeletons more robustly than classical algorithms.

Graph optimization: After initial skeletonization, build a graph of the skeleton and optimize it by merging nearly collinear segments and removing tiny dead-ends based on a learned or statistical criterion.

These measures should reduce spurious branches and yield cleaner connectivity, improving the fidelity of subsequent vectorization.

Vectorization (Polyline Extraction)

Current Method: Each skeleton path (pixel polyline) is simplified to a piecewise-linear polyline using the Ramer–Douglas–Peucker (RDP) algorithm. An adaptive epsilon (simplification threshold) scales with path length, and sub-pixel refinement can smooth endpoints.

Limitations: RDP only produces straight segments, so curved data (e.g. circular arcs or sine-wave shapes) require many segments or significant approximation error. If ε is too large, fine detail is lost; if too small, noise is preserved. RDP is also relatively slow on very long polylines. In summary, RDP may distort curved elements and is sensitive to the chosen error threshold.

Proposed Enhancements: To better capture curve fidelity, consider:

Curve fitting: Detect if a polyline is approximately circular or elliptical (e.g. pie slice arcs) and fit a true circle/arc segment rather than many small lines. Similarly, use splines (e.g. B-splines) for smooth curves.

Hierarchical segmentation: First segment the polyline at junctions or high-curvature points, then fit lower-order curves (lines or arcs) to each segment.

Neural vectorization: Recently, learned approaches (e.g. via differentiable rendering of vector graphics) have been used to convert images to SVG-like representations. While complex, a lightweight neural model could refine the RDP output.

Refined epsilon selection: Dynamically adjust ε based on local curvature: allow tighter tolerance where curvature is high, looser where nearly straight.
Together, these methods would preserve the true shape of plotted curves (improving fidelity of lines and arcs in the vector output).

OCR and Text Role Classification

Current Method: Text regions are detected and recognized using PaddleOCR. Recognized texts are then assigned semantic roles (title, xlabel, ylabel, legend, value) by spatial heuristics: e.g. “top 15% of image → title” or “left 15% vertically → ylabel”.

Limitations: Off-the-shelf OCR can mis-recognize small, rotated, or stylized fonts common in charts. Spatial heuristics break down for atypical layouts (e.g. title at the bottom, vertical x-axis label on the right side, multi-line titles). Multi-language or symbol-rich legends may confuse default OCR. Moreover, as Kim et al. point out, even advanced text-role classifiers suffer accuracy drop on real chart images compared to synthetic charts.

Proposed Enhancements: We advocate a multimodal, trainable approach:

Advanced text detection: Use a state-of-the-art scene-text detector (e.g. CRAFT, or YOLOv7 as in ChartEye) to find text boxes robustly, including rotated text. Then apply OCR on those boxes.

Super-resolution OCR: For very small text, apply a lightweight super-resolution GAN before OCR (as ChartEye did) to boost recognition quality.

Language models for correction: Feed OCR outputs into a domain-specific language model or spell-checker to correct errors (especially for numeric values and scientific terms).

Learned role classification: Replace hard-coded rules with a multimodal classifier (e.g. finetune LayoutLMv3 or UDOP on chart data). These models use text content, layout, and image features jointly and outperform heuristics. For example, Kim et al. showed that a multimodal transformer (LayoutLMv3) significantly improves chart text-role F1 over simple rules.

Role candidate validation: Use Stage 4 reasoning to double-check roles (e.g. if text thought to be “y-label” does not appear numeric or unit-like, reconsider).

By leveraging vision-language models and AI-based correction, the OCR engine will better capture actual text and assign roles reliably, reducing mis-tagging of labels or values.

Geometric Calibration (Axis Mapping)

Current Method: Numeric axis labels identified by OCR (role=ylabel or xlabel) are parsed as numbers. The system fits a linear or logarithmic model (depending on tick spacing) from pixel coordinates to chart values. For example, it models value = slope*pixel + intercept for a linear axis.

Limitations: The approach assumes perfectly linear or log scales and evenly spaced ticks. Errors in OCR (mis-read numbers) or missing ticks can distort the fit. Nonlinearities (e.g. a truncated axis or a piecewise scale) are not handled. If the axis is incomplete (no “0” label shown), the intercept estimation can be poor.

Proposed Enhancements: To increase robustness:

Outlier rejection: Use RANSAC or robust regression when fitting the axis model, so a few OCR errors (e.g. mis-parsed “8” as “3”) don’t skew the result. Remove points with large residuals.

Multi-scale detection: Explicitly detect if the axis uses a log scale by pattern (e.g. labels 1, 10, 100…). Extend to detect other special scales (e.g. square-root, dual axis).

Axis line detection: Use image processing (e.g. Hough line detection) to find the actual axis lines, which constrains pixel spans. This helps infer “zero” if the axis line extends to origin.

Data-driven refinement: After an initial fit, check consistency with element positions. For example, if detected bars or points systematically violate the predicted values, iteratively adjust the fit. One could use the extracted data (Stage 3 elements) as additional “pseudo-observations” to improve calibration.

Contextual cues: Leverage chart text (Stage 4 can parse legend or title) to detect units or scale type (e.g. “logarithmic scale” in legend note).

These methods will ensure that the pixel-to-value mapping is accurate even in noisy or nonstandard charts, directly benefiting the numerical reasoning later.

Element Detection (Bars, Markers, Pies, etc.)

Current Method: Different heuristics identify chart elements: vertical/horizontal bars by aspect-ratio thresholds on contours; scatter-plot markers by HoughCircle detection of circular points; pie slices by finding large circular regions and color-segmenting their sectors.

Limitations: These heuristics can fail when elements overlap, vary in shape, or are occluded. For example, touching bars may appear as one contour, irregularly shaped 3D bars or patterned bars confuse simple aspect tests, and Hough detection misses square or triangular markers. Pie detection breaks on donuts (center hole) or when text overlaps the pie area. No support exists for more complex chart types (stacked bars, error bars, area fills, heatmap cells, etc.).

Proposed Enhancements: We propose moving to deep learning detectors, guided by context:

Object detection networks: Train a one-stage detector (like YOLOv7 or SSD) or a two-stage detector (Faster/Cascade R-CNN) to find bars, points, and pie slices. The Context-Aware Chart Element Detection (CACHED) model shows that a Cascade R-CNN with added visual and positional context modules dramatically improves detection by learning layout cues. We can adopt a similar architecture for bars, lines, points, and legends, using the refined 18-class taxonomy from CACHED.

Instance segmentation: For highly overlapping elements (clustered bars, dense points), a Mask R-CNN style model can output pixel masks for each element, yielding precise boundaries rather than just boxes. This would cleanly separate adjacent bars and detect arbitrarily shaped markers.

Keypoint-based extraction: Inspired by ChartOCR, predict keypoints that define each element (e.g. top and bottom of each bar, corners of each pie sector) using a neural network. Then reconstruct the element geometrically. This unifies different chart types into one keypoint framework.

Ensemble of methods: Combine classical and learned methods. For example, use morphological clustering for candidate bars and then a classifier to verify them; or refine Hough circles with a small CNN that rejects false positives.

Transfer learning: Leverage large synthetic datasets (e.g. ExcelChart400K) to pretrain detectors and then fine-tune on the ArXiv dataset.

By employing modern detection/segmentation models and contextual reasoning, element detection will be far more accurate and general. As Luo et al. demonstrated, combining deep keypoint detection with adjustable rules yields SOTA extraction for bars, pies, and lines.

Chart Classification (Type Recognition)

Current Method: Originally rule-based scoring on structural features (counts of bars, presence of lines, etc.), now replaced by an ML classifier. The system currently uses a ResNet-18 CNN that achieves ~94% accuracy on eight chart types (area, bar, box, heatmap, histogram, line, pie, scatter) as per initial tests.

Limitations: A single ResNet may still confuse visually similar types (e.g. horizontal vs. vertical bars, or area vs. line). It requires sufficient training samples per class; rare chart types or new variants may not be recognized. ResNet-18 is relatively small – deeper or transformer-based models could do better.

Proposed Enhancements: To bolster classification:

Vision Transformers: Use a pre-trained ViT or Swin Transformer and fine-tune on charts. Vision transformers often outperform CNNs on varied image classes when data is limited. Indeed, ChartEye used hierarchical vision transformers for chart-type and text-role classification, yielding F1 ≈0.97 on chart-type.

Metric learning / few-shot: Implement a Siamese network or prototypical network for chart type. Bajić and Job showed that a Siamese CNN can achieve near-perfect classification with few examples per class. This would help when adding new chart categories with limited samples.

Multimodal fusion: Combine visual cues with textual cues (e.g. the chart title or axis labels) for classification. For instance, an LLM could analyze the chart title (“Scatter of X vs Y”) to confirm the CNN’s guess.

Ensembles: Use both rule-based and ML scores: e.g. if ResNet is uncertain, fall back to rule heuristics or voting among multiple classifiers.

Data augmentation: Synthesize more chart images (vary colors, aspect ratios, clutter) to strengthen invariance.

These strategies will increase accuracy and adaptability of the chart classifier, ensuring the type label fed to Stage 4 is correct.

Integration with Stage 4 (SLM Reasoning)

The output of Stage 3 is raw metadata (chart type, texts, elements, polylines, calibrated axes) fed to Stage 4, where an LLM (e.g. Qwen) will perform reasoning and generate answers. Enhancing Stage 3 therefore directly improves Stage 4: cleaner, more accurate metadata means the LLM can reason more reliably. For example, precise numerical data from Stage 3 leads to correct answer generation. We propose the following integration points:

Structured schema: Ensure Stage 3 outputs a well-defined JSON (as in Stage3Output), including confidences for each extracted item. High-confidence values can be prioritized in reasoning, while low-confidence ones can trigger caveats.

Iterative feedback: Allow Stage 4 to flag ambiguous extractions. For instance, if the LLM’s reasoning (guided by Stage 3 data) yields a contradiction or low confidence, Stage 4 could request Stage 3 to retry OCR with different settings or check alternate calibration. This is similar to human-in-the-loop refinement.

Context passage: Pass relevant text (titles, axis labels, legend entries) along with data to the LLM, so it can cross-verify (e.g. using a title like “COVID Cases Over Time” to confirm a line chart of cases).

Chain-of-Thought guidance: The improved metadata serves as facts for the LLM’s chain-of-thought. If an axis unit is “%” and values range 0–100, the LLM will trust the scale accordingly. Robust extraction prevents garbage-in semantics.
By designing Stage 3 output for easy consumption by an LLM and including confidence annotations, we create a seamless pipeline: Stage 4 reasoning (planned for next phase) can fully leverage the richer, more accurate metadata.

ArXiv Dataset Utilization

The existing ArXiv-classified chart dataset (with provided chart-type counts) will be central to training and evaluation. We recommend:

Training data for ML models: Use the dataset to fine-tune the CNN/Transformer classifier and detection models. If class counts are imbalanced (e.g. far more line charts than pie charts), apply class weights or data augmentation to ensure minority types are learned.

Validation of extraction: Treat a labeled subset of ArXiv charts as a benchmark with ground-truth metadata. For example, if the data includes the original data values or chart annotations, we can measure how well Stage 3 recovers them.

Cross-dataset transfer: Combine the ArXiv dataset with public datasets (ExcelChart400K, ICPR CHART infographics, etc.) to improve generalization. For example, pretrain element detectors on large synthetic sets, then fine-tune on the ArXiv images to adapt to scientific chart styles.

Statistical analysis: Use the chart-type counts table (e.g. a table listing “Bar: N images, Line: M images, …”) to guide model selection. For abundant classes, aggressive augmentation is less needed, whereas for rare classes, consider few-shot techniques.

By fully leveraging this dataset, all Stage 3 submodules can be trained and tested on realistic scientific charts. For example, Luo et al.’s ExcelChart400K (∼387k charts, with ~173k bars and 116k lines) enabled state-of-the-art bar/line extraction; similarly, our curated ArXiv charts can bootstrap our models.

Evaluation Strategy

We will quantify extraction fidelity at multiple levels:

Text extraction: Compute character/word recognition accuracy of OCR on chart labels. Additionally, measure text-role classification accuracy against human-annotated roles. For example, we can use F1-score as in ICPR CHART challenges.

Element detection: Use precision/recall and F1 (or mAP) for bounding boxes of bars, points, and pie slices, comparing to annotated ground truth. We may adopt the chart-specific metrics of ChartOCR: for bar charts, a custom IoU-based distance; for line charts, point-set F1; for pie charts, sequence-matching F1. In practice, Table 2 of Luo et al. shows using a composite score (higher is better) for Bar/Pie/Line extraction.

Geometric accuracy: Measure the error in calibrated values. For numeric axes, compute RMSE or R² of recovered values vs. true values of selected tick marks. If raw data is known, we can measure the error of extracted data points (e.g. MAE of bar heights or line values).

Classification: Report standard classification accuracy on held-out charts. Also use confusion matrices to see which types are mistaken.

End-to-end quality: Finally, evaluate the impact on Stage 4: for instance, run question-answering tasks on charts and measure answer accuracy using the full pipeline. Improved Stage 3 should yield higher QA accuracy.

Where applicable, we will follow benchmarks from prior work. ChartOCR’s methodology suggests designing type-specific metrics, and ChartEye uses F1 and mAP for each subtask. We will adopt similar metrics so results are comparable to the literature. Any user-visible errors (e.g. wrong label text) should be logged to prioritize further improvements.

In summary, this report outlines a vision for upgrading every Stage 3 submodule: from smarter preprocessing and OCR to deep-learned detectors and classifiers. These enhancements, informed by recent chart-understanding research, will yield higher-fidelity metadata. Integrated with Stage 4’s SLM reasoning, they will enable robust chart explanation and QA from diverse scientific figures.

Sources: The above discussion builds on the current Stage 3 design and cites recent chart extraction literature. Each enhancement is grounded in these sources or standard computer-vision practice.

Comprehensive Enhancement Plan for Stage 3 Extraction in Geo-SLM
ChatGPT
I’m piecing together a detailed research report to enhance the accuracy, robustness, and fidelity of Stage 3 modules in the Geo-SLM chart analysis system. This includes an overview, critical analysis, proposed enhancements, and integration plans.

I’m working through the tasks by opening STAGE3_EXTRACTION.md to gather info on current methods for each submodule.

Read from STAGE3_EXTRACTION.md
ChatGPT
Piecing together the remaining sections of STAGE3_EXTRACTION.md to complete the content.

Read from STAGE3_EXTRACTION.md
ChatGPT
The Stage 3 pipeline includes components like image preprocessing, skeletonization, vectorization, OCR, geometric mapping, element detection, and chart classification, aimed at improving metadata accuracy.

Read from MASTER_CONTEXT.md
Read from MASTER_CONTEXT.md
ChatGPT
I'm piecing together the details of the Arxiv-classified dataset, suspecting it might be in a report or linked to academic datasets. The mention of chart type counts is a key hint.

Searched for Arxiv dataset chart types
ChatGPT
I’m confirming that the user has a dataset of charts from Arxiv, with specific chart type counts. Now, I’m mapping out where to find this dataset.

Read from MASTER_CONTEXT.md
ChatGPT
I'm exploring the folder structure of the academic dataset, particularly "Arxiv charts" and its relevance to the Arxiv classification dataset.

Searched for chart type counts provided
ChatGPT
The search results for chart types seem irrelevant. I'm now looking into datasets or Arxiv references in STAGE3_EXTRACTION.md.

Read from STAGE3_EXTRACTION.md
ChatGPT
Examining the source code reveals it might be test or training data.

Read from STAGE3_EXTRACTION.md
ChatGPT
OK, let me see. I have to suggest improvements for every Stage 3 submodule, aiming to boost accuracy, robustness, and extracted metadata fidelity to streamline Stage 4's reasoning.

Searched for skeletonization chart extraction
ChatGPT
I'm mapping out chart preprocessing, skeletonization, vectorization, OCR, role classification, geometric calibration, element detection, and classification. Evaluating extraction strategies with fidelity and sophisticated metrics.

Searched for Lee algorithm skeletonization chart
ChatGPT
I'm gathering info on chart recognition and skeletonization, focusing on the Ramer-Douglas-Peucker algorithm.

Searched for vectorization chart extraction RDP
ChatGPT
I reviewed a research paper on deep sketch vectorization via implicit surface extraction and another on vectorization using hierarchical line and elliptical arc fitting, both potentially relevant to chart sketches.

Searched for chart image OCR text classification
ChatGPT
I'm looking into two models: "Text Role Classification in Scientific Charts Using Multimodal" from arXiv and ChartOCR (possibly Microsoft). Clicking on "Text Role Classification" to further investigate it.


Read arxiv.org

ChatGPT
I'm working through fine-tuning LayoutLMv3 and UDOP on chart datasets, leveraging text, image, and layout for text role classification tasks. Data augmentation is beneficial, though synthetic accuracy is high, suggesting a need for further real chart adaptations.

ChartOCR methodology combines deep framework and rule-based methods, using keypoint detection for chart components, and is relevant for Stage3 improvements.

ChartOCR suggests keypoint-based detection, which could unify chart types effectively. Exploring hybrid approaches and recent advances in document analysis.

I'm looking at the ExcelChart400K dataset and its statistics. It includes different chart types and datasets, like Arxiv. ChartOCR is a good source for examining dataset extraction methods.

I'm considering whether to include metrics, focusing on ChartOCR and chart classification research. Discussing CNNs and resnet18 integration, as well as using large networks or ensembles.

Searched for chart type classification dataset CNN
ChatGPT
I’m working through CNN ensemble methods for chart classification. Considering rule-based and ML-based integration, possibly using Rule-based/ResNet architectures. Exploring Arxiv-classified dataset for training/testing.

Searched for chart type counts Arxiv dataset
ChatGPT
I’m confirming that this portion is from Stage3Extractions. It might be repeated, so I’ll search for "counts" or "charts" for the required information.

Searched for chart type counts academic dataset Arxiv
ChatGPT
I’m considering chart type counts and infographics, possibly from internal user datasets or Stage3Visualizations for further insights.

Read from MASTER_CONTEXT.md
ChatGPT
I'm noting that it involves chart images from arxiv papers, classified by type, and detailing its potential applications for machine learning or calibration.

Read from 03_stage3_extraction.ipynb
ChatGPT
I’m focusing on the limitations and potential improvements of each submodule in Stage3, including image preprocessing, skeletonization, vectorization, OCR, geometric calibration, element detection, and chart classification.

Searched for chart image data extraction evaluation metric
ChatGPT
I'm thinking through ChartOCR's three chart types. Curious about their analysis approach. Progressing through pages 5 or 6 for details.

ChartOCR's evaluation includes metrics for bar, pie, and line charts, using point matching or region IoU.


Read more from arxiv.org

ChatGPT
ChartEye (2024) reports high F1 scores using YOLOv7 for text detection, hierarchical vision transformers for classification, and super-resolution to boost OCR accuracy. Integrating relevant quotes is crucial for context.


Read more from arxiv.org

ChatGPT
I examined hierarchical vision transformers for chart classification against YOLOv7 for text detection, highlighting super-resolution networks for text OCR improvement.

Searched for skeletonization chart segmentation deep learning
ChatGPT
Refining the search term to "watershed chart segmentation bars" for more precise results.

Searched for watershed bar chart segmentation
ChatGPT
I’m considering chart bar detection using CNN. The goal is to refine my approach and gather relevant insights.

Searched for CNN bar chart element detection
ChatGPT
I'm thinking through the subject of "Context-Aware Chart Element Detection" as detailed in the Arxiv paper from May 2023.


Read more from arxiv.org

ChatGPT
OK, let me see. Cascade R-CNN with context fusion, focusing on position-sensitive elements, achieved top performance on ICDAR and PMC datasets.


Read ar5iv.labs.arxiv.org

ChatGPT
I'm thinking through integrating global visual and positional context with Cascade R-CNN to refine taxonomy and enhance chart element detection.


Read more from ar5iv.labs.arxiv.org

ChatGPT
I'm exploring advanced networks and modalities, like Vision Transformers, hierarchical classification, and interpretability, while ensuring compatibility with Stage4's structured inputs.

Searched for academic dataset chart class counts
ChatGPT
I’m compiling information on the Arxiv dataset, focusing on chart type counts and potential classes like area, bar, box, heatmap, histogram, line, pie, and scatter. Evaluating class balance, augmentation, and stratified splits.


Read pmc.ncbi.nlm.nih.gov

ChatGPT
Siamese CNN highlights 100% accuracy with 50 images per class, engaging small dataset challenges and few-shot learning for chart classification.

Searched for chart element detection IoU F1
ChatGPT
I’m assessing detection, OCR, numeric, and classification tasks, considering precision, recall, F1 score, IoU, word error rate, MAE, percentage error, and overall combined score, along with the arxiv reference.

I’m organizing the structure with titles, sub-titles, and bullet lists for clarity.

Searched for skeletonization noisy images morphological artifacts
ChatGPT
Describing how noise or thick shapes can cause skeletonization artifacts, which often don't need a citation since they are widely accepted. For vectorization limitations, consulting "Ramer-Douglas-Peucker limitations" is suggested.

Searched for Ramer-Douglas-Peucker limitation curvature
ChatGPT
I’m piecing together the practical applications of Rabe's work to see if it aligns uniquely with the problem at hand.


Read en.wikipedia.org

ChatGPT
I’m thinking through the structured report with specific sections for current methods, limitations, and proposed enhancements, using headings and bullet lists for clear organization.

I’m detailing the current grayscale, negative, blur, and threshold methods. Highlighting limitations like noise sensitivity and global parameter issues, I then propose deep learning techniques for robust segmentation and noise removal.

Searched for chart segmentation U-Net background
ChatGPT
I’m noting Marian's insights on improving chart interpretation. We’ll use semantic segmentation or inpainting for automated background isolation. For vectorization, exploring curve fitting, spline interpolation, and learning-based methods.

Searched for image to vector graphics neural network
ChatGPT
I'm thinking through stages for better OCR, including two-stage detection, language models for correction, finetuning on chart fonts, and using directional detection to better handle rotated text.

Read from MASTER_CONTEXT.md
ChatGPT
I'm gathering info on evaluating each submodule's accuracy, including OCR, element detection, calibration, and chart classification, using relevant metrics and comparisons.

Read from STAGE3_EXTRACTION.md
ChatGPT
I’m piecing together the key sections for each topic, including preprocessing, skeleton, vectorization, OCR, calibration, elements, and classification, from document [46], noting the specific line ranges for precise citations.

I'm mapping out 15 lines from the top of PDF page 1, focusing on lines 1-50.

It's interesting to see that ChartOCR merges deep and rule-based methods to manage chart variations, aiming for a more generalizable approach by leveraging key points to define components.


Read arxiv.org

ChatGPT
I’m gathering insights from several studies to highlight advancements in chart-type and text-role classification, including ChartEye's use of hierarchical vision transformers and YOLOv7. It’s interesting to see the progression in evaluation metrics.

I’m mapping out the necessary references, verifying accuracy and completeness. All references are carefully compiled, and now I’m ready to proceed with the writing phase.

Research completed in 7m · 12 sources

Citations
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
A[RGB Image] --> B[Grayscale] B --> C{Apply Negative?} C -->|Yes| D[Invert: 255 - pixel] C -->|No| E[Keep Original] D --> F[Gaussian Blur] E --> F F --> G[Adaptive Threshold] G --> H{Denoise?}
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
| Role | Location Heuristic | Content Pattern | | --- | --- | --- | | title | Top 15% of image, centered | Any text | | ylabel | Left 15%, vertically oriented | Numeric or category | | xlabel | Bottom 15%, horizontally aligned | Numeric or category | | legend | Top-right or bottom, near color blocks | Labels | | value | Near chart elements | Numeric only |

microsoft.com
in style (e.g. bar chart, line chart, pie chart and etc.), which makes pure rule-based data extraction methods difficult to handle. However, it is also improper to directly apply endto-end deep learning solutions since these methods usually deal with specific types of charts. In this paper, we propose an unified method ChartOCR to extract data from various types of charts. We show that by combining deep framework and rule-based methods, we can achieve a satisfying generalization ability and obtain accurate and semantic-rich
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
**Algorithm**: Lee (1994) - Topology-preserving thinning
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
**Algorithm**: Ramer-Douglas-Peucker (1973)
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
```python epsilon = base_epsilon * sqrt(path_length / reference_length) ```

arxiv.org
Text Role Classification in Scientific Charts Using Multimodal Transformers
classification (Wang et al., 2020 ; Poco and Heer, 44; Yan et al., 2023). Furthermore, while the performance of text role classification models on synthetically created datasets is near perfect, there is a drastic drop in performance on real datasets consisting of charts extracted from scientific publications.

arxiv.org
ChartEye: A Deep Learning Framework for Chart Information Extraction
information extraction pipeline. The proposed framework utilizes hierarchal vision transformers for the tasks of chart-type and text-role classification, while YOLOv7 for text detection. The detected text is then enhanced using Super Resolution Generative Adversarial Networks to improve the recognition output of the OCR. Experimental results on a benchmark dataset show that our proposed framework achieves excellent performance at every stage with F1-scores of 0.97 for chart-type classification, 0.91 for text-role classification, and a mean Average Precision of 0.95 for text detection.

arxiv.org
Text Role Classification in Scientific Charts Using Multimodal Transformers
classification (Wang et al., 2020 ; Poco and Heer, 44; Yan et al., 2023). Furthermore, while the performance of text role classification models on synthetically created datasets is near perfect, there is a drastic drop in performance on real datasets consisting of charts extracted from scientific publications.
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
1. Parse numeric labels from OCR (role=ylabel) 2. Get pixel Y-coordinates of labels 3. Fit linear/logarithmic model:
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
| Pattern | Detection | Model | | --- | --- | --- | | Linear | Equal spacing in pixels | y = mx + b | | Logarithmic | 1, 10, 100, 1000... pattern | y = a * log(x) + b | | Percentage | 0%, 25%, 50%, 75%, 100% | Normalized linear |
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
```mermaid flowchart LR A[Contours] --> B{Aspect Ratio} B -->|height > width * 1.5| C[Vertical Bar] B -->|width > height * 1.5| D[Horizontal Bar] B -->|else| E[Not a Bar] C --> F[Extract: x, y, width, height, color] D --> F ```
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
```python circles = cv2.HoughCircles( gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=3, maxRadius=20 )
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
**Pie Slice Detection**:

ar5iv.labs.arxiv.org
[2305.04151] Context-Aware Chart Element Detection
As a prerequisite of chart data extraction, the accurate detection of chart basic elements is essential and mandatory. In contrast to object detection in the general image domain, chart element detection relies heavily on context information as charts are highly structured data visualization formats. To address this, we propose a novel method CACHED, which stands for C ontext-A ware Ch art E lement D etection, by integrating a local-global context fusion module consisting of visual context enhancement and positional context encoding with the Cascade R-CNN framework. To improve the generalization of our method for broader applicability, we refine the existing chart element categorization and standardized 18 classes for chart basic elements, excluding plot elements. Our

microsoft.com
an unified method ChartOCR to extract data from various types of charts. We show that by combining deep framework and rule-based methods, we can achieve a satisfying generalization ability and obtain accurate and semantic-rich intermediate results. Our method extracts the key points that define the chart components. By adjusting the prior rules, the framework can be applied to different chart types. Experiments show that our method achieves state-of-theart performance with fast

microsoft.com
6. Evaluation Metric In previous works, researchers usually borrow evaluation metrics from other domains, e.g., object detection or information retrieval. Those methods do not take into account the specialty of chart data. In this paper, we propose three evaluation metrics for three chart types.
file_000000002b4871faad8d0fb59cdfdd69
STAGE3_EXTRACTION.md
**Scoring Algorithm**:
pmc.ncbi.nlm.nih.gov
Chart Classification Using Siamese CNN - PMC
using Few-shot learning (FSL). Many of described advantages of Siamese CNNs are shown in examples. In the end, we show that the Siamese CNN can work with one image per class, and a 100% average classification accuracy is achieved with 50 images per class, where the CNN achieves only average classification accuracy of 43% for the same dataset.
file_00000000ad9071fa9048445ee55dc1d2
MASTER_CONTEXT.md
| v (Refined Data) +-------------------+ | Stage 5: Report | --> Validate, Insights, Format Output +-------------------+ | v Output (JSON + Report)

