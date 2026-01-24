# Instructions Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.1.0 | 2026-01-25 | That Le | Updated with stage documentation links |
| 1.0.0 | 2026-01-19 | That Le | Initial instructions for Geo-SLM Chart Analysis |

## Overview

This directory contains AI Agent instructions for the **Geo-SLM Chart Analysis** project. These files guide AI assistants (Cursor, Copilot, etc.) in maintaining code quality and project consistency.

## Instruction Files

| File | Scope (`applyTo`) | Description |
| --- | --- | --- |
| `system.instructions.md` | `**` | Global rules: language, formatting, logging |
| `project.instructions.md` | `**` | Project architecture, 5-stage pipeline, tech stack |
| `pipeline.instructions.md` | `src/core_engine/**` | Pipeline stages, schemas, error handling |
| `coding-standards.instructions.md` | `src/**` | Python standards, type hints, docstrings |
| `research.instructions.md` | `research/**`, `notebooks/**` | Experiment workflow, dataset management |
| `docs.instructions.md` | `docs/**` | Documentation standards, mermaid diagrams |

## Reference Files

| File | Purpose |
| --- | --- |
| `references/shared_vocabulary.md` | Constants, enums, naming conventions |

## Usage

### Always Attached (Global)

These are automatically loaded for all files:
- `system.instructions.md` - Base operational rules
- `project.instructions.md` - Project-specific architecture

### Context-Specific (Auto-loaded by scope)

| Working On | Instruction Loaded |
| --- | --- |
| `src/core_engine/**` | `pipeline.instructions.md` |
| `src/**` | `coding-standards.instructions.md` |
| `docs/**` | `docs.instructions.md` |
| `research/**`, `notebooks/**` | `research.instructions.md` |

## Hierarchy

```
system.instructions.md (Base rules)
    |
    +-- Language: Code=English, Conversation=Vietnamese
    +-- No emojis, context-rich logging
    +-- Core-first architecture
    |
    v
project.instructions.md (Project architecture)
    |
    +-- 5-stage pipeline design
    +-- Technology stack (locked)
    +-- Development phases
    |
    v
[module].instructions.md (Feature-specific)
    |
    +-- pipeline.instructions.md -> Schemas, error handling
    +-- coding-standards.instructions.md -> Type hints, docstrings
    +-- research.instructions.md -> Experiments, training
    +-- docs.instructions.md -> Markdown, mermaid
```

## Related Documentation

| Topic | Document |
| --- | --- |
| Project Overview | [docs/MASTER_CONTEXT.md](../../docs/MASTER_CONTEXT.md) |
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
