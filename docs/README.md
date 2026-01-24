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
|
+-- architecture/               # System design documentation
|   +-- SYSTEM_OVERVIEW.md      # High-level architecture
|   +-- PIPELINE_FLOW.md        # Stage-by-stage data flow (with mermaid)
|
+-- guides/                     # How-to guides
|   +-- ARXIV_DOWNLOAD_GUIDE.md # PDF download from Arxiv
|   +-- CHART_QA_GUIDE.md       # Chart QA generation guide
|
+-- archive/                    # Completed/deprecated docs
    +-- CONFIRM_CHART_QA_PIPELINE.md  # [COMPLETED] QA pipeline proposal
```

## Key Documents

| Document | Purpose | Status |
| --- | --- | --- |
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | Project overview for AI | [CURRENT] |
| [architecture/PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) | Pipeline stages detail | [CURRENT] |
| [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | System architecture | [CURRENT] |
| [guides/CHART_QA_GUIDE.md](guides/CHART_QA_GUIDE.md) | Chart QA generation | [CURRENT] |
| [guides/ARXIV_DOWNLOAD_GUIDE.md](guides/ARXIV_DOWNLOAD_GUIDE.md) | Arxiv PDF download | [CURRENT] |

## Project Progress

| Phase | Status | Key Deliverables |
| --- | --- | --- |
| Phase 1: Foundation | [COMPLETED] | Dataset: 2,852 charts, 13,297 QA pairs |
| Phase 2: Core Engine | [IN PROGRESS] | Stage 3-5 implementation |
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
