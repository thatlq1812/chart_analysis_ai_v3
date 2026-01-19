# Phase 1: Data Factory Guide

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-01-20 | That Le | Data collection pipeline documentation |

## 1. Overview

Phase 1 focuses on collecting **1,000+ diverse chart images** for training the Geo-SLM model. The Data Factory provides automated tools to:

- Hunt data from multiple sources (Arxiv, HuggingFace, PMC, ACL)
- Mine charts from PDFs
- Sanitize and validate images
- Generate synthetic charts

## 2. Architecture

```
tools/data_factory/
├── __init__.py
├── main.py              # CLI entry point
├── config.py            # Configuration settings
├── schemas.py           # Pydantic models
└── services/
    ├── hunter.py        # ArxivHunter, RoboflowHunter
    ├── hf_hunter.py     # HuggingFaceHunter
    ├── pmc_hunter.py    # PMCHunter (PubMed Central)
    ├── acl_hunter.py    # ACLHunter (ACL Anthology)
    ├── miner.py         # PDFMiner
    ├── sanitizer.py     # ImageSanitizer, ChartDetector
    └── generator.py     # SyntheticChartGenerator
```

## 3. Data Sources

| Source | Type | Quality | Speed | Diversity |
|--------|------|---------|-------|-----------|
| **HuggingFace** | Pre-labeled datasets | High | Fast | High |
| **Arxiv** | Academic PDFs | Medium | Slow | Medium (academic style) |
| **PMC** | Biomedical papers | Medium | Medium | Medium (scientific) |
| **ACL** | NLP papers | Medium | Medium | Medium (benchmarks) |
| **Synthetic** | Generated | Low | Fast | Customizable |

### 3.1. HuggingFace Datasets (Recommended First)

Pre-labeled chart datasets - fastest path to 1,000+ samples:

| Dataset | Size | Description |
|---------|------|-------------|
| `chartqa` | 9.6K | Chart images with QA pairs |
| `plotqa` | 224K | Scientific plots |
| `dvqa` | 3.5M | Synthetic bar charts |
| `chart2text` | 44K | Charts with summaries |
| `unichart` | 20K | Multi-task dataset |

### 3.2. Arxiv Papers

Academic papers with performance charts, benchmark results, etc.

**Pros:**
- High-quality figures
- Diverse chart types
- Free and legal

**Cons:**
- Requires PDF mining
- Rate limited (~3s/request)
- Many non-chart figures

### 3.3. PubMed Central (PMC)

Biomedical/clinical research papers.

**Best for:**
- Clinical trial results
- Epidemiological trends
- Statistical charts

### 3.4. ACL Anthology

NLP/AI research papers with benchmark charts.

**Best for:**
- Model comparison tables
- Performance metrics
- Ablation studies

## 4. Quick Start

### 4.1. Environment Setup

```bash
cd D:\elix\chart_analysis_ai_v3

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Verify installation
python -c "from tools.data_factory.services import HuggingFaceHunter; print('OK')"
```

### 4.2. Download from HuggingFace (Fastest)

```bash
# Quick start: sample from multiple datasets
python -m tools.data_factory.main hunt --sources huggingface --quick-start --limit 500

# Download specific dataset (e.g., ChartQA)
python -m tools.data_factory.main hunt --sources hf --dataset chartqa --limit 1000
```

### 4.3. Download from Arxiv

```bash
# Small test
python -m tools.data_factory.main hunt --sources arxiv --limit 50

# Large batch (use dedicated script for stability)
python scripts/download_arxiv_batch.py --limit 500

# Resume if interrupted
python scripts/download_arxiv_batch.py --limit 500 --resume
```

### 4.4. Download from Multiple Sources

```bash
# Combine sources
python -m tools.data_factory.main hunt --sources huggingface,arxiv,pmc --limit 100
```

## 5. CLI Reference

### 5.1. Hunt Command

```bash
python -m tools.data_factory.main hunt [OPTIONS]

Options:
  --sources TEXT      Comma-separated: arxiv,huggingface,pmc,acl,roboflow
  --limit INT         Maximum items to download (default: 50)
  --dataset TEXT      Specific dataset name (for HuggingFace)
  --quick-start       Sample from multiple HF datasets
  --api-key TEXT      API key (for Roboflow)
```

### 5.2. Mine Command

```bash
python -m tools.data_factory.main mine [OPTIONS]

Options:
  --input-dir PATH    Directory containing PDFs
  --limit INT         Maximum PDFs to process
```

### 5.3. Sanitize Command

```bash
python -m tools.data_factory.main sanitize [OPTIONS]

Options:
  --input-dir PATH    Directory containing images
  --move-failed       Move failed images to subfolder
  --detect-charts     Use heuristic chart detection
```

### 5.4. Generate Command

```bash
python -m tools.data_factory.main generate [OPTIONS]

Options:
  --count INT         Number of charts to generate (default: 100)
  --types TEXT        Comma-separated: bar,line,pie,scatter,area
```

### 5.5. Stats Command

```bash
python -m tools.data_factory.main stats
```

## 6. Output Structure

```
data/
├── raw_pdfs/                    # Downloaded PDFs from Arxiv
│   └── arxiv_*.pdf
├── academic_dataset/
│   ├── images/
│   │   ├── huggingface/
│   │   │   ├── chartqa/         # HuggingFace images
│   │   │   └── plotqa/
│   │   ├── arxiv/               # Extracted from PDFs
│   │   └── synthetic/           # Generated charts
│   ├── metadata/                # JSON metadata per image
│   └── manifests/               # Dataset manifests
└── training/
    └── yolo_format/             # YOLO training format
```

## 7. Recommended Workflow

### Step 1: Quick Diverse Dataset (Day 1)

```bash
# Get 500 samples from HuggingFace (fast, high quality)
python -m tools.data_factory.main hunt --sources hf --quick-start --limit 500
```

**Expected:** ~500 images in 10-15 minutes

### Step 2: Add Academic Charts (Day 1-2)

```bash
# Download 200 Arxiv papers
python scripts/download_arxiv_batch.py --limit 200

# Extract charts from PDFs
python -m tools.data_factory.main mine --limit 200
```

**Expected:** ~500-1000 chart images (2-3 per paper)

### Step 3: Add Domain-Specific Charts (Day 2-3)

```bash
# Biomedical charts
python -m tools.data_factory.main hunt --sources pmc --limit 100

# NLP benchmark charts
python -m tools.data_factory.main hunt --sources acl --limit 50
```

### Step 4: Validate and Clean

```bash
# Filter low-quality images
python -m tools.data_factory.main sanitize --move-failed --detect-charts

# Check statistics
python -m tools.data_factory.main stats
```

### Step 5: Generate Synthetic (Optional)

```bash
# Fill gaps with synthetic data
python -m tools.data_factory.main generate --count 200 --types bar,line,pie
```

## 8. Quality Guidelines

### 8.1. Minimum Requirements

| Metric | Threshold |
|--------|-----------|
| Image width | ≥ 300px |
| Image height | ≥ 300px |
| File size | ≥ 5KB |
| Aspect ratio | 0.3 - 3.0 |
| Unique colors | ≥ 50 |

### 8.2. Diversity Targets

| Chart Type | Target % | Min Count |
|------------|----------|-----------|
| Bar | 30% | 300 |
| Line | 25% | 250 |
| Pie | 15% | 150 |
| Scatter | 10% | 100 |
| Other | 20% | 200 |

## 9. Troubleshooting

### Issue: Arxiv rate limiting

```
Error: HTTP 429 Too Many Requests
```

**Solution:** The hunter has built-in rate limiting (3s between requests). If still failing, increase delay in `config.py`:

```python
arxiv_rate_limit_seconds: float = Field(default=5.0)  # Increase from 3.0
```

### Issue: HuggingFace authentication

```
Warning: You are sending unauthenticated requests
```

**Solution:** Set HuggingFace token:

```bash
export HF_TOKEN=your_token_here
# Or in Windows:
set HF_TOKEN=your_token_here
```

### Issue: PDF extraction fails

```
Error: Failed to process PDF
```

**Solution:** Ensure PyMuPDF is installed:

```bash
pip install pymupdf
```

## 10. Next Steps

After collecting 1,000+ images:

1. **Annotate:** Use Label Studio or CVAT to annotate chart types
2. **Split:** Create train/val/test splits (70/15/15)
3. **Train:** Move to Phase 2 - YOLO training

See [Phase 2: Detection Training](./PHASE2_DETECTION.md) for next steps.
