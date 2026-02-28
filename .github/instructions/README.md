# Instructions Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-02-28 | That Le | v2.0 - Hierarchical system with module instructions |
| 1.1.0 | 2026-01-25 | That Le | Updated with stage documentation links |
| 1.0.0 | 2026-01-19 | That Le | Initial instructions for Geo-SLM Chart Analysis |

## Overview

This directory contains AI Agent instructions for the **Chart Analysis AI v3** project. These files guide AI assistants (Copilot, Cursor, etc.) in maintaining code quality, architecture consistency, and project conventions.

The instruction system follows a **3-tier hierarchy**: Global → Project → Module.

## Tier 1: Global Instructions (Always Loaded)

| File | Scope | Description |
| --- | --- | --- |
| `system.instructions.md` | `**` | AI agent persona, communication protocol, coding standards, design patterns |
| `project.instructions.md` | `**` | Project identity, tech stack, architecture overview, AI routing, pipeline stages |

## Tier 2: Workflow Instructions

| File | Scope | Description |
| --- | --- | --- |
| `workflow.instructions.md` | `.github/**`, `scripts/**`, `Makefile` | CI/CD, Git branching, release process, Makefile commands |

## Tier 3: Module Instructions (Auto-loaded by scope)

| File | Scope (`applyTo`) | Description |
| --- | --- | --- |
| `module-detection.instructions.md` | `src/**/s2_detection*`, `config/yolo*` | YOLO chart detection, Stage 2 |
| `module-extraction.instructions.md` | `src/**/s3_extraction/**`, `models/weights/resnet*` | OCR, classification, Stage 3 |
| `module-reasoning.instructions.md` | `src/**/s4_reasoning/**`, `src/**/ai/**` | AI adapter pattern, routing, Stage 4 |
| `module-training.instructions.md` | `scripts/train_*`, `data/slm_training/**` | SLM fine-tuning framework, LoRA |
| `module-serving.instructions.md` | `src/api/**`, `src/worker/**`, `docker-compose.yml` | FastAPI, Celery, Docker deployment |

## Tier 3: Domain Instructions (Existing)

| File | Scope (`applyTo`) | Description |
| --- | --- | --- |
| `pipeline.instructions.md` | `src/core_engine/**` | Pipeline stages, schemas, error handling |
| `coding-standards.instructions.md` | `src/**` | Python standards, type hints, docstrings |
| `research.instructions.md` | `research/**`, `notebooks/**` | Experiment workflow, dataset management |
| `docs.instructions.md` | `docs/**` | Documentation standards, mermaid diagrams |

## Reference Files

| File | Purpose |
| --- | --- |
| `references/shared_vocabulary.md` | Constants, enums, naming conventions |

## Loading Hierarchy

```
┌─────────────────────────────────────────────────┐
│ Tier 1: ALWAYS LOADED                           │
│   system.instructions.md                        │
│   project.instructions.md                       │
├─────────────────────────────────────────────────┤
│ Tier 2: WORKFLOW (CI/CD, Git, releases)         │
│   workflow.instructions.md                      │
├─────────────────────────────────────────────────┤
│ Tier 3: MODULE (auto-loaded by file path)       │
│   module-detection.instructions.md              │
│   module-extraction.instructions.md             │
│   module-reasoning.instructions.md              │
│   module-training.instructions.md               │
│   module-serving.instructions.md                │
├─────────────────────────────────────────────────┤
│ Tier 3: DOMAIN (auto-loaded by file path)       │
│   pipeline.instructions.md                      │
│   coding-standards.instructions.md              │
│   research.instructions.md                      │
│   docs.instructions.md                          │
└─────────────────────────────────────────────────┘
```

**Conflict resolution:** Module > Domain > Project > System

## Related Documentation

| Topic | Document |
| --- | --- |
| Project Overview | [docs/MASTER_CONTEXT.md](../../docs/MASTER_CONTEXT.md) |
| Upgrade Report | [docs/reports/UPGRADE_REPORT_PRODUCTION_READY.md](../../docs/reports/UPGRADE_REPORT_PRODUCTION_READY.md) |
| Pipeline Flow | [docs/architecture/PIPELINE_FLOW.md](../../docs/architecture/PIPELINE_FLOW.md) |
| Stage 3 Details | [docs/architecture/STAGE3_EXTRACTION.md](../../docs/architecture/STAGE3_EXTRACTION.md) |
| Stage 4 (Planned) | [docs/architecture/STAGE4_REASONING.md](../../docs/architecture/STAGE4_REASONING.md) |
| Stage 5 (Planned) | [docs/architecture/STAGE5_REPORTING.md](../../docs/architecture/STAGE5_REPORTING.md) |
| Research Method | [docs/research/METHODOLOGY.md](../../docs/research/METHODOLOGY.md) |

## Key Rules Summary

1. **Language**: Code/Docs in English, Conversation in Vietnamese
2. **No Emojis**: Use `[DONE]`, `[TODO]`, `[IN PROGRESS]` instead
3. **Type Hints**: Required for all Python functions
4. **Logging**: Always include context (session_id, chart_id, etc.)
5. **Schemas**: Use Pydantic v2 for all data structures
6. **Testing**: Minimum 70% coverage for core_engine
7. **AI Routing**: All LLM calls go through AIRouter with fallback chains
8. **Adapters**: Provider SDKs only imported inside adapter classes
