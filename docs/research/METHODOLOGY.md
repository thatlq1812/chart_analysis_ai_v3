# Research Methodology - Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-25 | That Le | Research methodology and hybrid approach rationale |

## 1. Research Problem

### 1.1. Problem Statement

Existing chart-to-data extraction methods suffer from:

| Issue | Description | Impact |
| --- | --- | --- |
| **Precision Drift** | Multimodal LLMs generate approximate values | 1-2% error in logarithmic scales = 10x data deviation |
| **Hallucination** | End-to-end models fabricate data points | False data in scientific analysis |
| **Structure Loss** | Smooth curve fitting loses actual data points | Missing inflection points and peaks |

### 1.2. Research Hypothesis

> A hybrid neuro-symbolic approach combining Computer Vision for detection, Classical Geometry for measurement, and SLM for semantic reasoning can achieve higher accuracy than pure end-to-end multimodal models while maintaining interpretability.

## 2. Proposed Solution: Geo-SLM

### 2.1. Key Innovation

**Treat charts as geometric entities, not images to describe.**

| Traditional Approach | Geo-SLM Approach |
| --- | --- |
| Image -> LLM -> Text -> Parse | Image -> Geometry -> Math -> SLM Verify |
| Probabilistic values | Measured coordinates |
| Black-box reasoning | Traceable calculations |
| Requires large models | Works with small models |

### 2.2. Core Techniques

#### 2.2.1. Negative Image Transformation

Most chart images have dark lines on light background. Inverting to light-on-dark:
- Optimizes morphological operations
- Enables Grassfire algorithm for skeleton extraction
- Preserves thin dashed lines that would be lost otherwise

```
I_neg(x,y) = 255 - I_src(x,y)
```

#### 2.2.2. Topology-Preserving Skeletonization

Using Lee (1994) algorithm instead of Zhang-Suen:
- Preserves connectivity at junctions
- No spurious branches
- Maintains endpoint positions

#### 2.2.3. Piecewise Linear Representation

Charts are NOT smooth curves - they are discrete data points connected by straight lines.

**RDP Algorithm Benefits**:
- Preserves local extrema (peaks, valleys)
- Guarantees maximum error bound
- No hallucinated interpolation

#### 2.2.4. Geometric Calibration

Direct measurement from axis scales:
- Extract tick values via OCR
- Fit linear regression for pixel-to-value mapping
- Apply inverse transform to all detected points

## 3. System Architecture

```mermaid
flowchart TB
    subgraph Traditional["Traditional: End-to-End"]
        T1[Chart Image] --> T2[Multimodal LLM]
        T2 --> T3[Text Description]
        T3 --> T4[Parse to Data]
        
        style T2 fill:#ffcccc
    end
    
    subgraph GeoSLM["Geo-SLM: Hybrid"]
        G1[Chart Image] --> G2[YOLO Detection]
        G2 --> G3[Geometric\nAnalysis]
        G3 --> G4[Measured\nCoordinates]
        G4 --> G5[SLM\nRefinement]
        G5 --> G6[Structured Data]
        
        style G3 fill:#ccffcc
        style G4 fill:#ccffcc
    end
```

## 4. Academic Contributions

### 4.1. Novel Contributions

| Area | Contribution | Comparison |
| --- | --- | --- |
| Pipeline Design | 5-stage hybrid architecture | vs. single-model approaches |
| Vectorization | Adaptive epsilon RDP | vs. fixed simplification |
| OCR Context | Spatial role classification | vs. raw text extraction |
| Value Mapping | Geometric calibration | vs. LLM estimation |
| Error Handling | Graceful degradation | vs. all-or-nothing |

### 4.2. Expected Results

| Metric | Target | Baseline (DePlot) |
| --- | --- | --- |
| Value Accuracy | >95% | ~85% |
| Point Detection | >90% | ~80% |
| Type Classification | >95% | ~90% |
| Processing Time | <5s/chart | ~10s/chart |

## 5. Evaluation Methodology

### 5.1. Datasets

| Dataset | Size | Source | Purpose |
| --- | --- | --- | --- |
| Chart QA (custom) | 2,852 charts | Arxiv papers | Training/Testing |
| Chart-to-Text | 10,000+ | Public benchmark | Comparison |
| Synthetic | TBD | Generated | Controlled testing |

### 5.2. Metrics

| Metric | Formula | Purpose |
| --- | --- | --- |
| Value Accuracy | MAE between predicted and ground truth | Core accuracy |
| Point Recall | Detected points / Actual points | Completeness |
| Type Accuracy | Correct classifications / Total | Classification |
| Structural Similarity | SSIM of reconstructed chart | Visual verification |

### 5.3. Ablation Studies

| Experiment | Variants | Purpose |
| --- | --- | --- |
| With/without negative transform | 2 | Validate preprocessing |
| RDP vs Visvalingam | 2 | Validate vectorization |
| With/without SLM | 2 | Validate reasoning layer |
| Different SLM sizes | 3 | Model size vs accuracy |

## 6. Implementation Progress

| Phase | Status | Deliverables |
| --- | --- | --- |
| Phase 1: Foundation | [COMPLETED] | Dataset, Stages 1-2 |
| Phase 2: Core Engine | [IN PROGRESS] | Stages 3-5 |
| Phase 3: Optimization | [PLANNED] | Benchmarking, fine-tuning |
| Phase 4: Presentation | [PLANNED] | Demo, thesis document |

## 7. Related Work

### 7.1. End-to-End Approaches

| Paper | Method | Limitation |
| --- | --- | --- |
| DePlot (2023) | Pix2Struct + LLM | Precision drift |
| MatCha (2023) | Chart-specific pretraining | Requires large model |
| ChartReader (2022) | Object detection + rules | Limited chart types |

### 7.2. Classical Approaches

| Paper | Method | Limitation |
| --- | --- | --- |
| ReVision (2011) | Hough transform | Only bar charts |
| Savva et al. (2011) | Template matching | Fixed templates |

### 7.3. Geo-SLM Position

Geo-SLM bridges the gap:
- Uses modern detection (YOLO) for robustness
- Applies classical geometry for precision
- Leverages SLM for semantic understanding
- Achieves accuracy of classical methods with generalization of learning methods

## 8. References

1. Liu et al. "DePlot: One-shot visual language reasoning by plot-to-table translation" (2023)
2. Lee et al. "MatCha: Enhancing Visual Language Pretraining with Math Reasoning" (2023)
3. Ramer, U. "An iterative procedure for the polygonal approximation of plane curves" (1972)
4. Douglas, D. & Peucker, T. "Algorithms for the reduction of the number of points..." (1973)
5. Lee et al. "Building skeleton models via 3-D medial surface/axis thinning algorithms" (1994)
