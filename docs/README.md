# Documentation Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-19 | That Le | Documentation structure for V3 |

## Overview

This directory contains all technical documentation for the Geo-SLM Chart Analysis project.

## Structure

```
docs/
|
+-- MASTER_CONTEXT.md       # Project overview for AI agents
|
+-- architecture/           # System design documentation
|   +-- SYSTEM_OVERVIEW.md  # High-level architecture
|   +-- PIPELINE_FLOW.md    # Stage-by-stage data flow
|   +-- DATA_SCHEMAS.md     # Pydantic schema definitions
|   +-- DEPLOYMENT.md       # Deployment architecture
|
+-- research/               # Research documentation
|   +-- PAPER_NOTES.md      # Summary of referenced papers
|   +-- EXPERIMENTS.md      # Experiment tracking log
|   +-- METHODOLOGY.md      # Research methodology
|
+-- guides/                 # How-to guides
|   +-- QUICK_START.md      # Getting started
|   +-- DEVELOPMENT.md      # Development setup
|   +-- TRAINING.md         # Model training guide
|   +-- TROUBLESHOOTING.md  # Common issues
|
+-- reports/                # Progress reports
    +-- thesis/             # LaTeX thesis files
    +-- weekly/             # Weekly progress reports
```

## Key Documents

| Document | Purpose | Audience |
| --- | --- | --- |
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | AI agent context | AI Assistants |
| [architecture/PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) | Pipeline design | Developers |
| [guides/QUICK_START.md](guides/QUICK_START.md) | Getting started | New users |
| [research/METHODOLOGY.md](research/METHODOLOGY.md) | Research approach | Academic |

## Documentation Standards

All documentation follows the standards defined in:
`.github/instructions/docs.instructions.md`

Key rules:
1. Every file must have a version header table
2. Use mermaid for diagrams
3. No emojis - use text indicators: [DONE], [TODO], [IN PROGRESS]
4. Code blocks must specify language
5. Update docs when code changes
