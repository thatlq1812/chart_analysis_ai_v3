# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Stage 3: Extraction (OCR + Geometric Analysis)
- Stage 4: Reasoning (SLM Integration)
- Stage 5: Reporting (Output Formatting)

---

## [0.2.0] - 2026-01-24

### Phase 1: Foundation [COMPLETED]

#### Added
- **Chart QA Dataset**: 2,852 classified charts with 13,297 QA pairs
  - Source: Arxiv academic papers (800+ PDFs)
  - Classification via Google Gemini API
  - 5 QA pairs per chart (structural, counting, comparison, reasoning, extraction)
  
- **Data Factory Tools** (`tools/data_factory/`)
  - PDF Miner: Extract images from PDF documents
  - Gemini Classifier: Chart detection and type classification
  - QA Generator: Automated QA pair generation
  
- **Core Engine Stages**
  - Stage 1: Ingestion (`src/core_engine/stages/s1_ingestion.py`)
  - Stage 2: Detection (`src/core_engine/stages/s2_detection.py`)
  
- **Documentation**
  - MASTER_CONTEXT.md: Project overview
  - PIPELINE_FLOW.md: 5-stage pipeline architecture
  - SYSTEM_OVERVIEW.md: System design
  - CHART_QA_GUIDE.md: Chart QA generation guide
  - ARXIV_DOWNLOAD_GUIDE.md: PDF download instructions

#### Data Statistics
| Metric | Value |
| --- | --- |
| Total Images Processed | 2,852 |
| Total QA Pairs Generated | 13,297 |
| Source PDFs | 800+ |
| Chart Types | bar, line, pie, scatter, area, other |

---

## [0.1.0] - 2026-01-19

### Project Initialization

#### Added
- Initial project structure (V3)
- Configuration files (`config/base.yaml`, `config/models.yaml`, `config/pipeline.yaml`)
- GitHub instructions for AI agents
- Basic schema definitions (`src/core_engine/schemas/`)
- Test fixtures and configuration

#### Structure
```
chart_analysis_ai_v3/
├── .github/instructions/    # AI agent guidelines
├── config/                  # YAML configurations
├── data/                    # Data directories
├── docs/                    # Documentation
├── src/core_engine/         # Main engine code
├── tests/                   # Test suite
└── tools/                   # Utility tools
```

---

## Versioning

- **Major version (X.0.0)**: Breaking changes, architecture redesign
- **Minor version (0.X.0)**: New features, phase completion
- **Patch version (0.0.X)**: Bug fixes, minor improvements
