# Documentation Index

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 2.0.0 | 2026-02-04 | That Le | Complete documentation refresh |

## Quick Links

| Document | Purpose | Audience |
|----------|---------|----------|
| [MASTER_CONTEXT.md](MASTER_CONTEXT.md) | Project overview & status | Everyone |
| [architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | System architecture | Developers |
| [guides/QUICK_START.md](guides/QUICK_START.md) | Getting started | New developers |

## Documentation Structure

```
docs/
├── MASTER_CONTEXT.md          # Main project overview
├── README.md                  # This file
├── architecture/              # System design docs
│   ├── SYSTEM_OVERVIEW.md     # High-level architecture
│   ├── PIPELINE_FLOW.md       # Pipeline diagrams
│   ├── STAGE3_EXTRACTION.md   # Stage 3 details
│   ├── STAGE4_REASONING.md    # Stage 4 details
│   └── STAGE5_REPORTING.md    # Stage 5 details
├── guides/                    # How-to guides
│   ├── QUICK_START.md         # Getting started
│   ├── DEVELOPMENT.md         # Development workflow
│   ├── TESTING.md             # Testing guide
│   └── DATA_COLLECTION.md     # Data collection guide
├── research/                  # Research documents
│   └── SLM_FINE_TUNING_PLAN.md # SLM training plan
└── reports/                   # Progress reports
    └── WEEKLY_PROGRESS_*.md   # Weekly updates
```

## Project Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| **Data Collection** | ✅ Complete | 32,445 charts, 8 types |
| **Chart Detection** | ✅ Complete | YOLO 93.5% mAP@50 |
| **Chart Classification** | ✅ Complete | ResNet-18 94.14% accuracy |
| **Stage 3 Extraction** | ✅ Complete | OCR + Element Detection |
| **OCR Cache** | ✅ Complete | 46,910 entries |
| **Stage 4 Reasoning** | 🔄 In Progress | SLM integration |
| **Stage 5 Reporting** | ⏳ Planned | - |

## Key Documents by Role

### For New Team Members
1. Start with [QUICK_START.md](guides/QUICK_START.md)
2. Read [MASTER_CONTEXT.md](MASTER_CONTEXT.md)
3. Review [SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md)

### For Developers
1. [DEVELOPMENT.md](guides/DEVELOPMENT.md) - Development workflow
2. [TESTING.md](guides/TESTING.md) - Testing guide
3. [PIPELINE_FLOW.md](architecture/PIPELINE_FLOW.md) - Pipeline details

### For Researchers
1. [SLM_FINE_TUNING_PLAN.md](research/SLM_FINE_TUNING_PLAN.md) - SLM training
2. [STAGE4_REASONING.md](architecture/STAGE4_REASONING.md) - Reasoning design
3. Weekly reports in `reports/`

## Recent Updates

| Date | Document | Change |
|------|----------|--------|
| 2026-02-04 | All docs | Complete refresh after cleanup |
| 2026-01-30 | MASTER_CONTEXT.md | Stage 3 completion |
| 2026-01-29 | WEEKLY_PROGRESS | Progress report |

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

*Last updated: 2026-02-04*
