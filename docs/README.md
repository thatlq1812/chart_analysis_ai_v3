# Documentation Index

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-01-25 | That Le | Restructured documentation with full navigation |
| 1.1.0 | 2026-01-24 | That Le | Updated to reflect Phase 1 completion |
| 1.0.0 | 2026-01-19 | That Le | Documentation structure for V3 |

## Overview

This directory contains all technical documentation for the **Geo-SLM Chart Analysis** project - a hybrid AI system for extracting structured data from chart images.

---

## Quick Navigation

| Need to... | Go to |
| --- | --- |
| Understand the project | [MASTER_CONTEXT.md](MASTER_CONTEXT.md) |
| Get started quickly | [guides/QUICK_START.md](guides/QUICK_START.md) |
| Set up development | [guides/DEVELOPMENT.md](guides/DEVELOPMENT.md) |
| Understand pipeline flow | [architecture/PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) |
| See test results | [reports/ACADEMIC_DATASET_TEST_REPORT.md](reports/ACADEMIC_DATASET_TEST_REPORT.md) |

---

## Directory Structure

```
docs/
|
+-- MASTER_CONTEXT.md           # Project overview (START HERE)
+-- README.md                   # This file
|
+-- architecture/               # System design documentation
|   +-- SYSTEM_OVERVIEW.md      # High-level architecture
|   +-- PIPELINE_FLOW.md        # Stage-by-stage data flow
|   +-- STAGE3_EXTRACTION.md    # Stage 3: Extraction details
|   +-- STAGE4_REASONING.md     # Stage 4: SLM Reasoning (planned)
|   +-- STAGE5_REPORTING.md     # Stage 5: Reporting (planned)
|
+-- guides/                     # How-to guides
|   +-- QUICK_START.md          # Getting started guide
|   +-- DEVELOPMENT.md          # Development setup
|   +-- ARXIV_DOWNLOAD_GUIDE.md # PDF download from Arxiv
|   +-- CHART_QA_GUIDE.md       # Chart QA dataset guide
|
+-- research/                   # Research documentation
|   +-- METHODOLOGY.md          # Research methodology
|
+-- reports/                    # Generated reports & benchmarks
|   +-- ACADEMIC_DATASET_TEST_REPORT.md
|   +-- STAGE3_VISUALIZATION.md
|   +-- CLASSIFIER_IMPROVEMENTS.md
|   +-- *.json                  # Benchmark results
|
+-- images/                     # Generated visualizations
|   +-- stage3/                 # Stage 3 test outputs
|   +-- stage3_academic/        # Academic dataset outputs
|
+-- archive/                    # Historical/completed docs
    +-- CONFIRM_CHART_QA_PIPELINE.md
    +-- SESSION_LOG_*.md
    +-- instruction_p2_*.md
```

---

## Key Documents

### Core Documentation

| Document | Description | Status |
| --- | --- | --- |
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | Project overview, architecture, status | [CURRENT] |
| [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | System design philosophy | [CURRENT] |
| [architecture/PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) | Data flow diagrams | [CURRENT] |

### Stage Documentation

| Stage | Document | Status |
| --- | --- | --- |
| Stage 1-2 | [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | [DONE] |
| Stage 3 | [architecture/STAGE3_EXTRACTION.md](architecture/STAGE3_EXTRACTION.md) | [DONE] |
| Stage 4 | [architecture/STAGE4_REASONING.md](architecture/STAGE4_REASONING.md) | [PLANNED] |
| Stage 5 | [architecture/STAGE5_REPORTING.md](architecture/STAGE5_REPORTING.md) | [PLANNED] |

### Guides

| Guide | Purpose | Audience |
| --- | --- | --- |
| [QUICK_START.md](guides/QUICK_START.md) | Get up and running | All users |
| [DEVELOPMENT.md](guides/DEVELOPMENT.md) | Dev environment setup | Developers |
| [ARXIV_DOWNLOAD_GUIDE.md](guides/ARXIV_DOWNLOAD_GUIDE.md) | Download academic PDFs | Data collection |
| [CHART_QA_GUIDE.md](guides/CHART_QA_GUIDE.md) | Generate QA dataset | ML training |

### Research

| Document | Description |
| --- | --- |
| [research/METHODOLOGY.md](research/METHODOLOGY.md) | Research approach and contributions |

---

## Project Progress

| Phase | Status | Documentation |
| --- | --- | --- |
| Phase 1: Foundation | [COMPLETED] | Dataset: 2,852 charts |
| Phase 2: Core Engine | [IN PROGRESS] | Stage 3 done, 4-5 pending |
| Phase 3: Optimization | [PLANNED] | Benchmarking, fine-tuning |
| Phase 4: Presentation | [PLANNED] | Demo, thesis |

---

## Documentation Standards

All documentation follows standards in: `.github/instructions/docs.instructions.md`

### Key Rules

1. **Version Header**: Every file must start with a version table
2. **Mermaid Diagrams**: Use mermaid for flow/architecture diagrams
3. **No Emojis**: Use text indicators `[DONE]`, `[TODO]`, `[IN PROGRESS]`
4. **Code Blocks**: Always specify language
5. **Keep Updated**: Update docs when code changes

---

## Related Files

| File | Location | Purpose |
| --- | --- | --- |
| AI Instructions | `.github/instructions/` | Guidelines for AI agents |
| Configuration | `config/` | YAML config files |
| Source Code | `src/core_engine/` | Implementation |
| Tests | `tests/` | Test suite |
| Notebooks | `notebooks/` | Interactive exploration |

---

## Contributing to Documentation

1. Follow the standards in `.github/instructions/docs.instructions.md`
2. Update this README when adding new docs
3. Keep `MASTER_CONTEXT.md` current with project status
4. Archive completed/obsolete docs in `archive/`
