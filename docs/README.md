# Documentation Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.1.0 | 2026-01-24 | That Le | Updated to reflect Phase 1 completion |
| 1.0.0 | 2026-01-19 | That Le | Documentation structure for V3 |

## Overview

This directory contains all technical documentation for the Geo-SLM Chart Analysis project.

## Current Structure

```
docs/
|
+-- MASTER_CONTEXT.md           # Project overview for AI agents
+-- SESSION_LOG_2026_01_24.md   # Session log: Stage 3 testing
|
+-- architecture/               # System design documentation
|   +-- SYSTEM_OVERVIEW.md      # High-level architecture
|   +-- PIPELINE_FLOW.md        # Stage-by-stage data flow (with mermaid)
|   +-- STAGE3_EXTRACTION.md    # Stage 3 module documentation
|
+-- guides/                     # How-to guides
|   +-- ARXIV_DOWNLOAD_GUIDE.md # PDF download from Arxiv
|   +-- CHART_QA_GUIDE.md       # Chart QA generation guide
|
+-- reports/                    # Generated reports
|   +-- STAGE3_VISUALIZATION.md     # Stage 3 visualization report
|   +-- ACADEMIC_DATASET_TEST_REPORT.md  # Academic dataset test
|
+-- images/                     # Generated images
|   +-- stage3_report/          # Stage 3 unit test visualizations
|   +-- stage3_academic/        # Academic dataset test visualizations
|
+-- archive/                    # Completed/deprecated docs
    +-- CONFIRM_CHART_QA_PIPELINE.md  # [COMPLETED] QA pipeline proposal
```

## Key Documents

| Document | Purpose | Status |
| --- | --- | --- |
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | Project overview for AI | [CURRENT] v1.2.1 |
| [architecture/PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) | Pipeline stages detail | [CURRENT] |
| [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | System architecture | [CURRENT] |
| [architecture/STAGE3_EXTRACTION.md](architecture/STAGE3_EXTRACTION.md) | Stage 3 modules | [NEW] |
| [guides/CHART_QA_GUIDE.md](guides/CHART_QA_GUIDE.md) | Chart QA generation | [CURRENT] |
| [guides/ARXIV_DOWNLOAD_GUIDE.md](guides/ARXIV_DOWNLOAD_GUIDE.md) | Arxiv PDF download | [CURRENT] |
| [reports/ACADEMIC_DATASET_TEST_REPORT.md](reports/ACADEMIC_DATASET_TEST_REPORT.md) | Stage 3 test results | [NEW] |
| [SESSION_LOG_2026_01_24.md](SESSION_LOG_2026_01_24.md) | Session log | [NEW] |

## Project Progress

| Phase | Status | Key Deliverables |
| --- | --- | --- |
| Phase 1: Foundation | [COMPLETED] | Dataset: 2,852 charts, 13,297 QA pairs |
| Phase 2: Core Engine | [IN PROGRESS] | Stage 3 DONE, Stage 4-5 pending |
| Phase 3: Optimization | [PLANNED] | Model fine-tuning, benchmarking |
| Phase 4: Presentation | [PLANNED] | Demo UI, thesis document |

## Documentation Standards

All documentation follows the standards defined in:
`.github/instructions/docs.instructions.md`

Key rules:
1. Every file must have a version header table
2. Use mermaid for diagrams
3. No emojis - use text indicators: [DONE], [TODO], [IN PROGRESS]
4. Code blocks must specify language
5. Update docs when code changes
