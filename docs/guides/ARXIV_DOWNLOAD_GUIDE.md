# Arxiv PDF Download Guide

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-01-20 | That Le | Step-by-step guide for downloading Arxiv papers |

## 1. Overview

This guide explains how to download academic papers from Arxiv for chart extraction. Arxiv is a great source of high-quality figures including:

- Performance benchmark charts
- Ablation study results
- Training curves
- Comparison tables

## 2. Prerequisites

```bash
# Activate environment
cd D:\elix\chart_analysis_ai_v3
.venv\Scripts\activate

# Verify arxiv library
python -c "import arxiv; print('arxiv version:', arxiv.__version__)"
```

## 3. Quick Download (Small Batch)

For testing or small datasets:

```bash
# Download 10 papers (test)
python -m tools.data_factory.main hunt --sources arxiv --limit 10

# Download 50 papers
python -m tools.data_factory.main hunt --sources arxiv --limit 50
```

**Output location:** `data/raw_pdfs/arxiv_*.pdf`

## 4. Large Batch Download (500+ papers)

For large downloads, use the dedicated batch script which includes:
- Progress tracking
- Resume capability
- Better error handling

### 4.1. Basic Usage

```bash
# Download 500 papers
python scripts/download_arxiv_batch.py --limit 500
```

### 4.2. Resume After Interruption

If the download is interrupted (Ctrl+C, network error, etc.):

```bash
# Resume from where it stopped
python scripts/download_arxiv_batch.py --limit 500 --resume
```

Progress is saved in `data/arxiv_progress.json`.

### 4.3. Download 5000 Papers

For the full dataset target:

```bash
# This will take 4-5 hours
python scripts/download_arxiv_batch.py --limit 5000
```

**Important:** For long downloads, use a **separate terminal** (CMD/PowerShell) instead of VS Code's integrated terminal to avoid interruptions.

## 5. Time Estimates

| Papers | Search Time | Download Time | Total |
|--------|-------------|---------------|-------|
| 50 | ~2 min | ~3 min | ~5 min |
| 100 | ~3 min | ~6 min | ~10 min |
| 500 | ~10 min | ~30 min | ~40 min |
| 1000 | ~15 min | ~1 hour | ~1.5 hours |
| 5000 | ~30 min | ~4 hours | ~4.5 hours |

**Note:** Arxiv has rate limiting (~3 seconds between requests).

## 6. Search Queries

The hunter searches Arxiv using these queries (configured in `config.py`):

```python
ARXIV_QUERIES = [
    # Computer Vision
    'cat:cs.CV AND (chart OR visualization OR diagram)',
    'cat:cs.CV AND "bar chart"',
    'cat:cs.CV AND "line chart"',
    
    # Machine Learning
    'cat:cs.LG AND (benchmark OR comparison OR performance)',
    'cat:cs.LG AND ablation',
    
    # NLP
    'cat:cs.CL AND benchmark',
    'cat:cs.CL AND evaluation',
    
    # Chart-specific papers
    '"chart understanding"',
    '"chart question answering"',
    ...
]
```

### 6.1. Add Custom Queries

Edit `tools/data_factory/config.py`:

```python
ARXIV_QUERIES: List[str] = [
    # Add your custom queries
    'cat:cs.CV AND "your search term"',
    ...
]
```

## 7. Verified Papers

The hunter prioritizes these known-good papers with charts:

| Arxiv ID | Paper | Why |
|----------|-------|-----|
| 1906.02337 | Chart Understanding | Chart-specific |
| 2203.10244 | ChartQA | Chart QA benchmark |
| 1512.03385 | ResNet | Classic performance charts |
| 1706.03762 | Transformer | Attention is All You Need |
| 2010.11929 | ViT | Vision Transformer |

## 8. Output Files

### 8.1. Downloaded PDFs

```
data/raw_pdfs/
├── arxiv_1512_03385v1.pdf
├── arxiv_1706_03762v7.pdf
├── arxiv_1810_04805v2.pdf
└── ...
```

### 8.2. Progress File

```json
// data/arxiv_progress.json
{
  "downloaded": ["1512.03385v1", "1706.03762v7", ...],
  "failed": ["some_id"],
  "last_query_idx": 5
}
```

### 8.3. Logs

```
logs/
├── arxiv_batch_2026-01-20.log
└── data_factory_2026-01-20.log
```

## 9. Next Steps: Extract Charts from PDFs

After downloading PDFs, extract chart images:

```bash
# Extract charts from all PDFs
python -m tools.data_factory.main mine

# Extract from specific directory
python -m tools.data_factory.main mine --input-dir data/raw_pdfs --limit 100
```

This will:
1. Convert PDF pages to images
2. Detect chart regions
3. Crop and save individual charts
4. Generate metadata JSON

**Output:** `data/academic_dataset/images/arxiv/`

## 10. Troubleshooting

### Error: arxiv library not installed

```bash
pip install arxiv
```

### Error: Rate limited (429)

Wait a few minutes and retry. The script has built-in retry logic.

### Error: Connection timeout

Check your internet connection. The script will auto-retry 3 times.

### Process keeps getting interrupted

Use a standalone terminal instead of VS Code:

```cmd
# Windows CMD
cd D:\elix\chart_analysis_ai_v3
.venv\Scripts\python scripts\download_arxiv_batch.py --limit 5000
```

### Resume not working

Check if `data/arxiv_progress.json` exists and is valid JSON.

## 11. Monitoring Progress

### Check downloaded count

```bash
ls data/raw_pdfs/*.pdf | wc -l
```

### Check progress file

```bash
cat data/arxiv_progress.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Downloaded: {len(d[\"downloaded\"])}, Failed: {len(d[\"failed\"])}')"
```

### View live logs

```bash
tail -f logs/arxiv_batch_*.log
```

## 12. Sample Commands Summary

```bash
# Quick test (10 papers)
python -m tools.data_factory.main hunt --sources arxiv --limit 10

# Medium batch (100 papers)
python -m tools.data_factory.main hunt --sources arxiv --limit 100

# Large batch with progress (500 papers)
python scripts/download_arxiv_batch.py --limit 500

# Full dataset (5000 papers) - run in separate terminal
python scripts/download_arxiv_batch.py --limit 5000

# Resume after interruption
python scripts/download_arxiv_batch.py --limit 5000 --resume

# Extract charts from PDFs
python -m tools.data_factory.main mine
```
