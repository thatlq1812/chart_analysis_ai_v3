# THESIS INSTRUCTIONS - Geo-SLM Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-01 | That Le | Phase 1 initialization, context scan complete |

---

## 1. Purpose

This file governs the thesis writing process for the Geo-SLM Chart Analysis capstone project.
It is the reference for all thesis-related work across Phase 1-4 of the writing workflow.

### 1.1. Core Design Philosophy

The system is built as a **modular, extensible pipeline** -- each component (stage, adapter, provider)
is independently replaceable and testable. Key architectural traits:

- **Adapter Pattern**: AI providers (Local SLM, Gemini, OpenAI) share a common interface (`BaseAIAdapter`),
  allowing seamless addition of new providers without modifying pipeline logic.
- **Stage Abstraction**: Each pipeline stage inherits from `BaseStage[InputT, OutputT]`,
  enabling independent development, testing, and replacement of any stage.
- **Router Pattern**: `AIRouter` decouples task-level logic from provider selection,
  supporting confidence-based fallback chains configurable via YAML.
- **Config-Driven**: All thresholds, model paths, and stage toggles are externalized
  to YAML files, enabling experiment reproducibility without code changes.
- **Core-First Independence**: `core_engine/` has zero dependency on any web framework;
  serving (FastAPI), task queue (Celery), and interface (CLI/Streamlit) are separate layers.

This modularity is a **deliberate design contribution** of the thesis and should be
emphasized in both System Design and Methodology sections.

---

## 2. Instructor Requirements (from instruction_Mar01.md)

### Meeting 1 - Context and Motivation
- Project context (what, why, applications, existing work globally/in Vietnam)
- Visual aids: images, charts, diagrams

### Meeting 2 - Related Work and Approach
- At least 3 related methods (global/Vietnam)
- Detailed approach: modifications/additions/combinations from related work
- EDA overview of dataset

### Meeting 3+ - Training Results
- Self-collected data details (not from Kaggle/Roboflow)
- Training/fine-tuning results, analysis, improvement proposals

### Report Requirements
- LaTeX format using FPT University template (aip491.sty)
- Overleaf collaboration with instructor
- Progressive writing aligned with weekly meetings

---

## 3. LaTeX Template Structure (aip491.sty)

Template provides: A4, 25.4mm margins, Fourier font, natbib numbered references.

### Required Sections (from main.tex)
1. **Introduction** - Background, Objective, Problem Statement, Scope & Limitations
2. **Project Management Plan** - Team, Communication (optional for this course)
3. **Literature Review** - At least 3 existing methods
4. **Methodology** - Data, Approach/Models
5. **System Design and Implementation** - AI integration, data flow, interface, deployment
6. **Results and Discussion** - Metrics, analysis, comparison, recommendations
7. **Conclusion**
8. **References** (elsarticle-num-names.bst)
9. **Appendix**

---

## 4. Mapping: Project Content -> Thesis Sections

| Thesis Section | Source Content |
| --- | --- |
| Introduction | MASTER_CONTEXT.md Section 1, METHODOLOGY.md problem statement |
| Literature Review | METHODOLOGY.md related work, research/papers/ |
| Methodology | Pipeline stages architecture, training.yaml, module-*.instructions.md |
| System Design | Architecture docs, pipeline.yaml, models.yaml, src/core_engine/ code |
| Results | Weekly progress reports, data_pipeline_report_v1.md, model metrics |
| Conclusion | MASTER_CONTEXT.md phases, future work from reports |

---

## 5. Key Metrics for Thesis (Verified from Project)

| Metric | Value | Source |
| --- | --- | --- |
| Total raw PDFs | ~4,000 arXiv papers | data_pipeline_report_v1.md |
| Page images extracted | ~150,000 | data_pipeline_report_v1.md |
| Candidate charts | ~70,000 | data_pipeline_report_v1.md |
| Classified charts | 32,364 (8 types) | MASTER_CONTEXT v3.0.0 |
| OCR cache entries | 46,910 (~589 MB) | MASTER_CONTEXT v3.0.0 |
| YOLO detection mAP@0.5 | >0.85 (93.5% reported) | WEEKLY_PROGRESS_20260204 |
| ResNet-18 test accuracy | 94.14% | MASTER_CONTEXT v3.0.0 |
| ResNet-18 ONNX inference | 6.90ms mean (CPU) | MASTER_CONTEXT v3.0.0 |
| Stage 3 extraction | 32,364/32,364, 0% error | WEEKLY_PROGRESS_20260301 |
| Stage 3 overall confidence | 92.6% | WEEKLY_PROGRESS_20260129 |
| SLM training dataset (v3) | 268,799 samples | MASTER_CONTEXT v3.0.0 |
| Test suite | 176/177 passing (99.4%) | MASTER_CONTEXT v3.0.0 |
| Chart types supported | 8 (area, bar, box, heatmap, histogram, line, pie, scatter) | models.yaml |
| Hardware | RTX 3060 6GB VRAM | TRAINING.md Section 12 |

---

## 6. Writing Rules

- Write in English (academic, technical tone)
- No emojis or informal language
- All figures must be 300dpi, saved to thesis_capstone/figures/
- Every claim must reference a metric, log, or config file
- Use \cite{} for all external references
- Draft in Markdown first (thesis_capstone/drafts/), then convert to .tex
- Cross-reference figures and tables using \cref{}

---

## 7. Directory Convention

```
docs/thesis_capstone/
    main.tex                  # Master document
    aip491.sty                # FPT University style
    refs.bib                  # Bibliography
    fptlogo.png               # Logo
    contents/                 # Final .tex sections
        introduction.tex
        project_management.tex
        literature_review.tex
        methodology.tex
        system_desigin_and_implementation.tex
        results_discussion.tex
        conclusion.tex
    drafts/                   # Markdown drafts (Phase 2 output)
        project_structure_scan.json
        01_introduction.md
        02_literature_review.md
        03_methodology.md
        04_data_pipeline.md
        05_system_architecture.md
        06_stage1_ingestion.md
        07_stage2_detection.md
        08_stage3_extraction.md
        09_stage4_reasoning.md
        10_stage5_reporting.md
        11_results_discussion.md
        12_conclusion.md
        12_novelty_assessment.md
    figures/                  # Auto-generated figures (Phase 3 output)
        fig_chart_type_distribution.pdf
        fig_slm_dataset_distribution.pdf
        fig_feature_quality_by_type.pdf
        fig_processing_time_breakdown.pdf
        fig_data_pipeline_funnel.pdf
        fig_resnet_per_class_accuracy.pdf
        fig_dataset_v2_vs_v3.pdf
        tables/               # LaTeX table .tex snippets
            tab_technology_stack.tex
            tab_ai_providers.tex
            tab_chart_type_distribution.tex
            tab_dataset_splits.tex
            tab_yolo_results.tex
            tab_resnet_results.tex
            tab_stage3_feature_quality.tex
            tab_comparison_related_work.tex
            tab_pipeline_stages.tex
            tab_data_pipeline_scale.tex
            tab_slm_training_config.tex
            tab_test_suite.tex
        tikz/                 # TikZ architecture diagrams
            tikz_pipeline_architecture.tex
            tikz_layer_architecture.tex
            tikz_ai_router.tex
            tikz_stage3_submodules.tex
            tikz_data_pipeline.tex
            tikz_approach_comparison.tex
```

---

## 8. Phase 3 Asset Mapping (Figures -> Thesis Sections)

### Section 4: Methodology
| Asset | Type | File |
| --- | --- | --- |
| Pipeline architecture | TikZ | tikz_pipeline_architecture.tex |
| Data pipeline flow | TikZ | tikz_data_pipeline.tex |
| Chart type distribution | PDF figure | fig_chart_type_distribution.pdf |
| Data pipeline funnel | PDF figure | fig_data_pipeline_funnel.pdf |
| Dataset v2 vs v3 | PDF figure | fig_dataset_v2_vs_v3.pdf |
| Pipeline stages summary | LaTeX table | tab_pipeline_stages.tex |
| Chart type distribution | LaTeX table | tab_chart_type_distribution.tex |
| Dataset splits | LaTeX table | tab_dataset_splits.tex |
| Data pipeline scale | LaTeX table | tab_data_pipeline_scale.tex |
| SLM training config | LaTeX table | tab_slm_training_config.tex |

### Section 5: System Design and Implementation
| Asset | Type | File |
| --- | --- | --- |
| Layer architecture | TikZ | tikz_layer_architecture.tex |
| Stage 3 submodules | TikZ | tikz_stage3_submodules.tex |
| AI Router flow | TikZ | tikz_ai_router.tex |
| Approach comparison | TikZ | tikz_approach_comparison.tex |
| Technology stack | LaTeX table | tab_technology_stack.tex |
| AI providers | LaTeX table | tab_ai_providers.tex |

### Section 6: Results and Discussion
| Asset | Type | File |
| --- | --- | --- |
| Feature quality by type | PDF figure | fig_feature_quality_by_type.pdf |
| Processing time | PDF figure | fig_processing_time_breakdown.pdf |
| ResNet per-class accuracy | PDF figure | fig_resnet_per_class_accuracy.pdf |
| SLM dataset distribution | PDF figure | fig_slm_dataset_distribution.pdf |
| YOLO results | LaTeX table | tab_yolo_results.tex |
| ResNet results | LaTeX table | tab_resnet_results.tex |
| Stage 3 quality | LaTeX table | tab_stage3_feature_quality.tex |
| Related work comparison | LaTeX table | tab_comparison_related_work.tex |
| Test suite | LaTeX table | tab_test_suite.tex |

### Existing Assets (copy to figures/)
| Asset | Source | Notes |
| --- | --- | --- |
| Grad-CAM summary | models/explainability/gradcam_summary_all_classes.png | ResNet-18 attention |
| Per-class Grad-CAM | models/explainability/gradcam_*.png | 8 individual types |

---

## 9. Pending Items (to supplement after SLM training)

| Item | Dependency | Target Week |
| --- | --- | --- |
| SLM inference metrics (JSON valid rate, field accuracy) | QLoRA training complete | Week 7-8 |
| ChartQA/PlotQA benchmark comparison | SLM checkpoint | Week 9 |
| GPT-4V baseline comparison | API access | Week 9 |
| Ablation study results | Baseline results | Week 9-10 |
| Stage 4 Gemini vs SLM comparison | Both evaluated | Week 8 |

---

## 10. Large Data Directories (DO NOT READ)

The following directories contain large binary/data files and must NOT be read during thesis work:
- data/ (120.80 GB, 468K files)
- models/weights/ (60.2 MB)
- models/slm/ (6.52 GB)
- models/onnx/
- runs/ (1.4 MB)

Safe directories for content reading: .github/instructions/, config/, docs/, src/, tests/, scripts/, tools/
