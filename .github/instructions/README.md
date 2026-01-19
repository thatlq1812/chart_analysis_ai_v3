# Instructions Directory

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-19 | That Le | Initial instructions for Geo-SLM Chart Analysis |

## Overview

This directory contains AI Agent instructions for the **Geo-SLM Chart Analysis** project. These files guide AI assistants (Cursor, Copilot, etc.) in maintaining code quality and project consistency.

## Instruction Files

| File | Scope | Description |
| --- | --- | --- |
| `system.instructions.md` | `**` | Global rules: language, formatting, no emojis |
| `project.instructions.md` | `**` | Project architecture, principles, technology stack |
| `research.instructions.md` | `research/**`, `notebooks/**` | AI/ML research workflow, experiment tracking |
| `coding-standards.instructions.md` | `src/**` | Python coding standards, type hints, docstrings |
| `docs.instructions.md` | `docs/**` | Documentation standards, mermaid diagrams |
| `pipeline.instructions.md` | `src/core_engine/**` | Pipeline stages, data flow, schemas |

## Usage

### Always Attached (Global)

```yaml
applyTo: '**'
```
- `system.instructions.md` - Always loaded
- `project.instructions.md` - Always loaded

### On-Demand (Module-Specific)

Load when working on specific areas:
- Working on AI pipeline? Load `pipeline.instructions.md`
- Writing experiments? Load `research.instructions.md`
- Updating docs? Load `docs.instructions.md`

## Hierarchy

```
system.instructions.md (Base rules)
    ↓
project.instructions.md (Project-specific)
    ↓
[module].instructions.md (Feature-specific)
```

Lower-level instructions can override higher-level ones if explicitly stated.
