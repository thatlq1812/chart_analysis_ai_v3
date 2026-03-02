# Documentation Index

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 3.0.0 | 2026-03-02 | That Le | Full pipeline implemented, thesis complete, docs refreshed |
| 2.0.0 | 2026-02-04 | That Le | Complete documentation refresh |

## Quick Links

| Document | Purpose | Audience |
|----------|---------|----------|
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | Project overview and status | Everyone |
| [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | System architecture | Developers |
| [guides/QUICK_START.md](guides/QUICK_START.md) | Getting started | New developers |

## Documentation Structure

```
docs/
+-- MASTER_CONTEXT.md          # Main project overview (v4.0.0)
+-- CHANGELOG.md               # Change log
+-- README.md                  # This file
+-- architecture/              # System design docs
|   +-- SYSTEM_OVERVIEW.md     # High-level architecture
|   +-- PIPELINE_FLOW.md       # Pipeline diagrams
|   +-- STAGE3_EXTRACTION.md   # Stage 3 details
|   +-- STAGE4_REASONING.md    # Stage 4 details
|   +-- STAGE5_REPORTING.md    # Stage 5 details
+-- guides/                    # How-to guides
|   +-- QUICK_START.md         # Getting started
|   +-- DEVELOPMENT.md         # Development workflow
|   +-- TRAINING.md            # Comprehensive training guide
|   +-- CHART_QA_GUIDE.md      # QA pair generation
|   +-- ARXIV_DOWNLOAD_GUIDE.md # PDF download instructions
+-- thesis_capstone/           # Academic thesis (39 pages)
|   +-- main.tex               # Master document (XeLaTeX)
|   +-- refs.bib               # Bibliography (21 entries)
|   +-- contents/              # 7 chapter .tex files
|   +-- figures/               # 7 PDFs + 12 tables + 6 TikZ
+-- progress/                  # Weekly progress reports
+-- reports/                   # Technical reports
+-- research/                  # Research documents
+-- archive/                   # Historical docs
```

## Project Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| **Data Collection** | Complete | 32,364 charts, 8 types |
| **Chart Detection** | Complete | YOLOv8m 93.5% mAP@50 |
| **Chart Classification** | Complete | ResNet-18 94.14% accuracy |
| **Stage 3 Extraction** | Complete | OCR + Element Detection + Geometric Mapping |
| **Stage 4 Reasoning** | Complete | AI Router + 4 adapters (SLM/Gemini/OpenAI) |
| **Stage 5 Reporting** | Complete | Insights, validation, JSON + text output |
| **AI Routing Layer** | Complete | `src/core_engine/ai/` (8 files, 55 tests) |
| **Pipeline Wiring** | Complete | All 5 stages live in pipeline.py |
| **SLM Training Data** | Complete | 268,799 samples (v3, all 8 types) |
| **SLM Fine-tuning** | In Progress | QLoRA on Qwen-2.5-1.5B |
| **Academic Thesis** | Complete | 39 pages, 0 LaTeX errors |
| **Test Suite** | Complete | 232 tests, 21 files, all passing |

## Key Documents by Role

### For New Team Members
1. Start with [QUICK_START.md](guides/QUICK_START.md)
2. Read [MASTER_CONTEXT.md](MASTER_CONTEXT.md)
3. Review [SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md)

### For Developers
1. [DEVELOPMENT.md](guides/DEVELOPMENT.md) - Development workflow
2. [PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) - Pipeline details
3. [STAGE4_REASONING.md](architecture/STAGE4_REASONING.md) - AI reasoning

### For Researchers
1. [TRAINING.md](guides/TRAINING.md) - Model training (SLM, ResNet, YOLO)
2. [STAGE4_REASONING.md](architecture/STAGE4_REASONING.md) - Reasoning design
3. Weekly reports in `progress/`
4. Thesis in `thesis_capstone/`

## Recent Updates

| Date | Document | Change |
|------|----------|--------|
| 2026-03-02 | All docs | Full documentation refresh (v4.0.0) |
| 2026-03-02 | thesis_capstone/ | Thesis complete: 39 pages, 0 LaTeX errors |
| 2026-03-01 | MASTER_CONTEXT.md | Data pipeline complete, v3 dataset |
| 2026-02-28 | Instructions | Production architecture upgrade |
| 2026-02-04 | All docs | Complete refresh after cleanup |

## Documentation Standards

### Formatting
- Use Markdown tables for structured data
- Include version and date in document headers
- Use mermaid diagrams for architecture

### Naming Conventions
- Architecture docs: `COMPONENT_NAME.md` (uppercase)
- Guides: `GUIDE_NAME.md` (uppercase)
- Reports: `WEEKLY_PROGRESS_YYYYMMDD.md`

### Language
- Technical docs: English
- Comments: English
- Conversation: Vietnamese

---

*Last updated: 2026-03-02*
