# Research Paper Novelty Assessment

## 1. Publication Feasibility

### 1.1. Verdict: FEASIBLE for Workshop/Short Paper

The Geo-SLM system has sufficient novelty for a **workshop paper** or **short paper** at venues like:
- ACL/EMNLP Workshop on Document Intelligence
- AAAI Workshop on Knowledge Discovery from Unstructured Data
- IEEE ICDAR (Document Analysis and Recognition)
- ECAI/IJCAI Workshop on Hybrid AI

A **full conference paper** would require completed SLM evaluation + benchmark comparison (Phase 3).

### 1.2. Assessment Matrix

| Criteria | Score (1-5) | Notes |
| --- | --- | --- |
| Technical novelty | 3.5 | Hybrid approach is novel; individual components are known |
| Empirical rigor | 2.5 | Strong internal metrics, no public benchmark yet |
| Reproducibility | 4.0 | Well-documented, config-driven, modular |
| Practical utility | 4.0 | Runs offline on consumer GPU (RTX 3060) |
| Writing readiness | 3.0 | Drafts ready; dataset stats strong |

## 2. Novelty Claims (Ranked by Strength)

### Claim 1: Hybrid Neuro-Symbolic Pipeline (STRONG)
**What**: 5-stage pipeline combining YOLO detection, classical geometry, and SLM reasoning -- each component handles what it does best.

**Why novel**: Existing work either uses pure end-to-end models (DePlot, MatCha) or pure rule-based systems (ReVision). No published system combines modern object detection + classical geometric measurement + task-specific SLM in a modular pipeline with per-stage schema contracts.

**Evidence**: 93.5% detection + 94.14% classification + 100% extraction + 92.6% pipeline confidence.

**Differentiation from DePlot (2023)**: DePlot feeds chart image directly to Pix2Struct then LLM. Values are estimated from visual appearance. Geo-SLM **measures** values via axis calibration and pixel-to-coordinate mapping. This is fundamentally different (measurement vs estimation).

---

### Claim 2: Geometric Calibration for Chart Value Extraction (STRONG)
**What**: RANSAC-based axis calibration + linear regression for pixel-to-value mapping. Values are computed, not predicted.

**Why novel**: Traditional approaches either use Hough line detection (brittle) or let LLMs estimate values (imprecise). The geometric calibration pipeline with outlier rejection is more robust.

**Evidence**: 69.9% axis info coverage across 32,364 academic charts (multiple types, multiple axis formats). Combined with OCR tick extraction.

**Key technique**: Detect OCR text near axis regions, parse numeric values, fit linreg model per axis, apply inverse transform to all pixel coordinates.

---

### Claim 3: Negative Image Transform for Skeleton Extraction (MODERATE)
**What**: Inverting chart images (light-on-dark) before skeletonization enables better morphological operations on thin lines and dashed patterns.

**Why novel**: Standard chart processing operates on original (dark-on-light). The negative transform specifically optimizes for Grassfire/Lee skeletonization by ensuring chart elements are the foreground.

**Formula**: `I_neg(x,y) = 255 - I_src(x,y)`

**Evidence**: Qualitative improvement in skeleton quality, especially for dashed and thin lines. No ablation study yet (planned).

---

### Claim 4: Adaptive Epsilon RDP Vectorization (MODERATE)
**What**: Douglas-Peucker vectorization with adaptive epsilon based on skeleton statistics, rather than a fixed simplification parameter.

**Why novel**: Charts are piecewise-linear by nature. Using adaptive RDP preserves local extrema while removing noise, without the hallucinated interpolation of smooth curve fitting.

**Differentiation**: Smooth spline fitting (common in chart digitization tools) introduces false inflection points. RDP guarantees maximum deviation bound and preserves actual data points.

---

### Claim 5: Large-Scale Academic Chart Dataset with Geometric Features (MODERATE)
**What**: 32,364 academic charts with structured Stage 3 features (axis calibration, OCR roles, element geometry, type classification) + 268,799 SLM training samples in ChatML format.

**Why novel**: Existing chart datasets (ChartQA, PlotQA) provide QA pairs but not geometric feature annotations. This dataset includes pixel-level measurements alongside semantic QA data.

**Limitation**: Dataset is not yet publicly released. Release would strengthen this claim significantly.

---

### Claim 6: Modular AI Routing with Fallback Chains (MODERATE-LOW)
**What**: Adapter pattern + Router with per-task fallback chains (local_slm -> gemini -> openai). Task type determines priority order.

**Why novel**: Practical engineering contribution. Most systems hardcode a single LLM. The routing pattern enables offline-first inference with graceful fallback to cloud APIs.

**Limitation**: More of a systems engineering contribution than a research novelty. Best suited for a "systems" track paper.

---

### Claim 7: Curriculum Learning for Chart SLM (LOW - PLANNED)
**What**: 4-stage curriculum (easy -> hard) for SLM fine-tuning on chart understanding tasks.

**Status**: Planned but not yet implemented or evaluated. Cannot claim as a contribution until experiments are complete.

> **[TODO - SUPPLEMENT AFTER SLM TRAINING]**
> - Update Claim 7 status after QLoRA training (Week 7-8)
> - Add SLM eval metrics to Section 2 claims
> - Revise feasibility verdict if benchmark results are strong

## 3. Weakness Analysis

### 3.1. Missing Elements for Strong Paper
| Gap | Impact | Remediation |
| --- | --- | --- |
| No public benchmark evaluation | Cannot claim SOTA | Evaluate on ChartQA / PlotQA (Phase 3) |
| SLM not yet trained | Core hypothesis partially unvalidated | Complete QLoRA training (Week 7-8) |
| No ablation study | Cannot prove each component's contribution | Run ablations (with/without neg. transform, RDP, SLM) |
| No comparison with GPT-4V | Missing strong baseline | Run GPT-4V on test set |
| Single data domain | Academic charts only | Plan business/dashboard eval |

### 3.2. What Can Be Claimed Now
Even without SLM training completion, the following are evidenced:
- The hybrid pipeline architecture is validated (100% extraction, 92.6% confidence)
- Geometric calibration works on 8 chart types at scale (32,364 charts)
- The data pipeline methodology produces quality training data (268,799 samples, 0% corruption)
- Modular design enables component replacement without pipeline changes

## 4. Recommended Paper Structure

### For Workshop/Short Paper (4-6 pages)
```
Title: "Geo-SLM: Hybrid Neuro-Symbolic Chart Analysis
        with Geometric Calibration and Small Language Models"

1. Introduction (0.5 page)
   - Problem: LLM hallucination in chart value extraction
   - Contribution: Hybrid approach with geometric precision

2. Related Work (0.5 page)
   - DePlot, MatCha, ChartReader, ReVision

3. Method (1.5 pages)
   - 5-stage pipeline overview
   - Key techniques: neg. transform, RANSAC calibration, adaptive RDP
   - AI routing architecture

4. Dataset (0.5 page)
   - 32,364 charts, 268,799 QA samples
   - Quality metrics

5. Experiments (1.0 page)
   - Detection: 93.5% mAP
   - Classification: 94.14%
   - Extraction: 100% success
   - End-to-end: 92.6% confidence
   - (If ready: SLM eval)

6. Discussion & Conclusion (0.5 page)
   - Measurement vs. estimation paradigm
   - Future: public benchmark, SLM distillation
```

### For Full Paper (8+ pages)
Same structure, expanded with:
- Ablation study (Section 5b)
- Public benchmark comparison (Section 5c)
- Per-type detailed analysis (Section 5d)
- SLM training methodology (Section 3b)
- Error analysis and failure cases (Section 5e)

## 5. Comparison Table for Thesis vs Paper

| Content | Thesis Report | Research Paper |
| --- | --- | --- |
| Background / Literature Review | 3-4 pages | 0.5 page |
| System Architecture | 3-4 pages (all stages) | 1.5 pages (key innovations only) |
| Dataset | 2 pages | 0.5 page |
| Results | 3-4 pages (all experiments) | 1 page (key results) |
| Discussion | 2 pages | 0.5 page |
| Appendix | Code listings, configs | None |
| **Total** | **40-60 pages** | **4-8 pages** |
| **Tone** | Comprehensive, educational | Concise, contribution-focused |
| **Audience** | Thesis committee | Peer researchers |

## 6. Action Items for Paper Readiness

| Priority | Action | Timeline | Dependency |
| --- | --- | --- | --- |
| P0 | Complete SLM QLoRA training | Week 7-8 | Training data ready |
| P0 | Evaluate SLM on held-out test set | Week 8 | Training complete |
| P1 | Run on ChartQA benchmark | Week 9 | SLM checkpoint |
| P1 | Run GPT-4V baseline comparison | Week 9 | API access |
| P1 | Ablation study (3 experiments) | Week 9-10 | Baseline results |
| P2 | Dataset release preparation | Week 10 | License review |
| P2 | Write paper draft | Week 10-11 | All results |
| P3 | Internal review + revision | Week 12 | Draft complete |
