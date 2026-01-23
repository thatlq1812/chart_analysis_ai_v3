# Chart QA Data Generation Guide

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-23 | That Le | Complete guide for Chart QA pipeline |

## 1. Overview

Module **Chart QA Data Generator** su dung Google Gemini API de:
- Phan loai hinh anh la chart hay khong
- Xac dinh loai chart (bar, line, pie, scatter, etc.)
- Tao 5 cap cau hoi-tra loi (QA pairs) cho moi chart

### 1.1. Use Cases

| Use Case | Description |
| --- | --- |
| Training Data | Tao dataset de train Chart QA models |
| Fine-tuning | Data cho fine-tune multimodal LLMs |
| Benchmarking | Tao test set de danh gia models |

### 1.2. Pipeline Flow

```
PDF Files (800+)
      |
      v
[1] PDF Miner -----> Extracted Images (~12,000)
      |
      v
[2] Gemini Classifier -----> Is Chart? + Chart Type
      |
      v
[3] QA Generator -----> 5 QA pairs per chart
      |
      v
[4] Dataset Export -----> dataset.json
```

## 2. Installation

### 2.1. Dependencies

```bash
# Install required packages
pip install google-generativeai tqdm

# Or install full data-factory dependencies
pip install -e ".[data-factory]"
```

### 2.2. API Key Setup

Tao file `.env` o project root (neu chua co):

```bash
# .env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_TEMPERATURE=0.1
```

**Lay API key tai:** https://aistudio.google.com/apikey

### 2.3. Verify Setup

```bash
# Check if API key is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('GEMINI_API_KEY') else 'MISSING')"
```

## 3. Quick Start

### 3.1. Full Pipeline (Recommended)

```bash
# Step 1: Extract images from PDFs (skip if already done)
python -m tools.data_factory.main mine

# Step 2: Generate QA pairs (with collection)
python -m tools.data_factory.main generate-qa --limit 100 --workers 10 --collect

# Step 3: Check status
python -m tools.data_factory.main qa-status
```

### 3.2. Step-by-Step

```bash
# Only classify images (no QA generation)
python -m tools.data_factory.main classify --limit 100

# Only generate QA for classified charts
python -m tools.data_factory.main generate-qa --limit 100

# Collect existing QA pairs into dataset.json
python -m tools.data_factory.main generate-qa --collect-only
```

## 4. CLI Commands Reference

### 4.1. classify

Phan loai hinh anh thanh chart/non-chart.

```bash
python -m tools.data_factory.main classify [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--input-dir` | `data/academic_dataset/images` | Thu muc chua hinh anh |
| `--limit` | None | So luong hinh toi da xu ly |
| `--workers` | 10 | So luong API workers (concurrent) |
| `--checkpoint-freq` | 100 | Luu checkpoint moi N hinh |
| `--session` | Auto | Session ID de resume |
| `--quiet` | False | Tat progress bar |

**Examples:**

```bash
# Classify first 500 images with 8 workers
python -m tools.data_factory.main classify --limit 500 --workers 8

# Resume previous session
python -m tools.data_factory.main classify --session 20260123_143022
```

### 4.2. generate-qa

Tao QA pairs cho charts.

```bash
python -m tools.data_factory.main generate-qa [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--input-dir` | `data/academic_dataset/images` | Thu muc chua hinh anh |
| `--limit` | None | So luong hinh toi da xu ly |
| `--workers` | 10 | So luong API workers |
| `--checkpoint-freq` | 100 | Luu checkpoint moi N hinh |
| `--session` | Auto | Session ID |
| `--collect` | False | Collect QA sau khi xu ly |
| `--collect-only` | False | Chi collect, khong xu ly |
| `--quiet` | False | Tat progress bar |

**Examples:**

```bash
# Process 1000 images and collect dataset
python -m tools.data_factory.main generate-qa --limit 1000 --collect

# Only collect existing QA pairs
python -m tools.data_factory.main generate-qa --collect-only

# High-throughput processing
python -m tools.data_factory.main generate-qa --workers 15 --checkpoint-freq 50
```

### 4.3. qa-status

Hien thi trang thai pipeline.

```bash
python -m tools.data_factory.main qa-status
```

**Output example:**

```
==================================================
CHART QA PIPELINE STATUS
==================================================

Category                      Count
------------------------------------
Charts classified               847
Non-charts                     1153
QA pairs generated              847

Recent Checkpoints
------------------------------------
  20260123_143022: 2000 processed, 847 charts

Output directory: data/academic_dataset/chart_qa
==================================================
```

## 5. Output Structure

### 5.1. Directory Layout

```
data/academic_dataset/chart_qa/
|
+-- classified/
|   +-- charts/              # Classification results for charts
|   |   +-- arxiv_2601_08743_p3_img2.json
|   |   +-- ...
|   +-- non_charts/          # Classification results for non-charts
|       +-- arxiv_2601_08743_p1_img1.json
|       +-- ...
|
+-- qa_pairs/                # QA pairs for each chart
|   +-- arxiv_2601_08743_p3_img2_qa.json
|   +-- ...
|
+-- checkpoints/             # Pipeline checkpoints
|   +-- checkpoint_20260123_143022.json
|
+-- stats/                   # Processing statistics
|   +-- stats_20260123_143022.json
|
+-- dataset.json             # Combined dataset file
```

### 5.2. Classification JSON Format

```json
{
    "image_id": "arxiv_2601_08743_p3_img2",
    "image_path": "data/academic_dataset/images/arxiv_2601_08743_p3_img2.png",
    "is_chart": true,
    "chart_type": "bar",
    "confidence": 0.95,
    "classification": {
        "is_chart": true,
        "chart_type": "bar",
        "confidence": 0.95,
        "elements": {
            "has_title": true,
            "has_legend": true,
            "has_x_axis": true,
            "has_y_axis": true,
            "has_grid": false,
            "has_data_labels": true
        },
        "brief_description": "Bar chart comparing model performance across 5 categories"
    },
    "processing_time_seconds": 2.34
}
```

### 5.3. QA Pairs JSON Format

```json
{
    "image_id": "arxiv_2601_08743_p3_img2",
    "image_path": "data/academic_dataset/images/arxiv_2601_08743_p3_img2.png",
    "chart_type": "bar",
    "qa_pairs": [
        {
            "question": "What is the title of this chart?",
            "answer": "Model Performance Comparison on GLUE Benchmark",
            "question_type": "structural"
        },
        {
            "question": "How many bars/categories are shown in the chart?",
            "answer": "There are 5 categories representing different models: BERT, RoBERTa, ALBERT, XLNet, and DeBERTa",
            "question_type": "counting"
        },
        {
            "question": "Which model achieves the highest score?",
            "answer": "DeBERTa achieves the highest score at approximately 91.3%",
            "question_type": "comparison"
        },
        {
            "question": "What trend can you observe across the models?",
            "answer": "There is a general improvement trend from older models (BERT) to newer ones (DeBERTa), with scores increasing from about 84% to 91%",
            "question_type": "reasoning"
        },
        {
            "question": "What is the approximate score of RoBERTa?",
            "answer": "RoBERTa achieves approximately 88.5% on the GLUE benchmark",
            "question_type": "extraction"
        }
    ],
    "generated_at": "2026-01-23T14:35:22"
}
```

### 5.4. Dataset JSON Format

```json
{
    "version": "1.0.0",
    "created_at": "2026-01-23T15:00:00",
    "total_images": 847,
    "total_qa_pairs": 4235,
    "samples": [
        {
            "image_id": "...",
            "image_path": "...",
            "chart_type": "bar",
            "qa_pairs": [...]
        }
    ]
}
```

## 6. Question Types

Moi chart se co 5 cau hoi thuoc 5 loai khac nhau:

| Type | Description | Example |
| --- | --- | --- |
| `structural` | Thong tin co cau (title, labels, legend) | "What is the title of this chart?" |
| `counting` | Dem so luong elements | "How many bars are shown?" |
| `comparison` | So sanh gia tri (max, min, differences) | "Which category has the highest value?" |
| `reasoning` | Xu huong, patterns, insights | "What trend can you observe?" |
| `extraction` | Trich xuat gia tri cu the | "What is the value of category A?" |

## 7. Jupyter Notebook Usage

### 7.1. Open Notebook

```bash
jupyter lab notebooks/02_chart_qa_generation.ipynb
```

### 7.2. Key Sections

| Section | Purpose |
| --- | --- |
| 1. Setup | Import modules, load API key |
| 2. Status | Check current pipeline status |
| 3. Extract | Extract images from PDFs |
| 4. Test Single | Test classification on one image |
| 5. Run Pipeline | Execute full pipeline |
| 6. Export | Collect QA into dataset |
| 7. Inspect | View random samples |
| 8. Statistics | Compute dataset stats |

### 7.3. Test Single Image

```python
from tools.data_factory.services.gemini_classifier import GeminiChartClassifier

classifier = GeminiChartClassifier()
result = classifier.process_image_full("path/to/image.png")

print(f"Is Chart: {result['classification']['is_chart']}")
print(f"Type: {result['classification']['chart_type']}")
for qa in result['qa_pairs']:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
```

## 8. Configuration

### 8.1. Pipeline Config

```python
from tools.data_factory.services.qa_generator import QAPipelineConfig

config = QAPipelineConfig(
    max_api_workers=10,          # Concurrent API calls
    max_pdf_workers=4,           # PDF processing workers
    checkpoint_frequency=100,    # Checkpoint every N images
    requests_per_minute=60,      # Gemini RPM limit
    tokens_per_minute=32000,     # Gemini TPM limit
    max_retries=3,               # Retry failed calls
    retry_delay=1.0,             # Delay between retries
    min_image_size=10000,        # Skip tiny images (bytes)
    skip_processed=True,         # Resume from checkpoint
)
```

### 8.2. Rate Limiting

Gemini API co rate limits. Config mac dinh:

| Limit | Value | Description |
| --- | --- | --- |
| RPM | 60 | Requests per minute |
| TPM | 32,000 | Tokens per minute |

**Tang throughput voi paid tier:**

```python
config = QAPipelineConfig(
    requests_per_minute=1000,    # Paid tier
    tokens_per_minute=4000000,   # Paid tier
    max_api_workers=20,          # More workers
)
```

## 9. Troubleshooting

### 9.1. Common Errors

| Error | Cause | Solution |
| --- | --- | --- |
| `GEMINI_API_KEY not found` | Missing .env file | Create .env with API key |
| `Rate limit exceeded` | Too many requests | Reduce workers, wait |
| `Image not found` | Invalid path | Check image paths |
| `JSON parse error` | Gemini response issue | Retry, check image quality |

### 9.2. Resume Failed Session

```bash
# Find session ID
python -m tools.data_factory.main qa-status

# Resume with session ID
python -m tools.data_factory.main generate-qa --session 20260123_143022
```

### 9.3. Clean Restart

```bash
# Remove all checkpoints and start fresh
rm -rf data/academic_dataset/chart_qa/checkpoints/*
python -m tools.data_factory.main generate-qa --limit 100
```

### 9.4. Debug Single Image

```python
from tools.data_factory.services.gemini_classifier import GeminiChartClassifier

classifier = GeminiChartClassifier()

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test
result = classifier.classify_image("path/to/image.png")
print(result)
```

## 10. Best Practices

### 10.1. Processing Strategy

1. **Start small**: Test voi 100 images truoc
2. **Monitor quality**: Inspect random samples
3. **Checkpoint often**: Dat checkpoint_frequency thap (50-100)
4. **Use appropriate workers**: 8-10 cho free tier, 15-20 cho paid

### 10.2. Quality Control

```bash
# After processing, inspect samples
python -c "
from tools.data_factory.services.qa_generator import get_pipeline_status
import json
from pathlib import Path

status = get_pipeline_status()
print(f'Charts: {status[\"directories\"][\"charts\"]}')
print(f'QA pairs: {status[\"directories\"][\"qa_pairs\"]}')

# Check a sample
qa_files = list(Path('data/academic_dataset/chart_qa/qa_pairs').glob('*.json'))
if qa_files:
    with open(qa_files[0]) as f:
        sample = json.load(f)
    print(f'Sample QA pairs: {len(sample[\"qa_pairs\"])}')
"
```

### 10.3. Cost Estimation

| Items | Count | Cost (gemini-3-flash) |
| --- | --- | --- |
| 12,000 images | ~24,000 API calls | ~$2-5 |
| ~3,600 charts | 5 QA each | ~$3-8 |
| **Total** | | **~$5-15** |

## 11. API Reference

### 11.1. GeminiChartClassifier

```python
class GeminiChartClassifier:
    def __init__(
        self,
        api_key: str = None,           # From GEMINI_API_KEY env
        model: str = None,             # From GEMINI_MODEL env
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ): ...
    
    def classify_image(self, image_path: Path) -> Dict:
        """Classify single image as chart/non-chart."""
        
    def generate_qa_pairs(self, image_path: Path, chart_type: str) -> List[Dict]:
        """Generate 5 QA pairs for a chart."""
        
    def process_image_full(self, image_path: Path) -> Dict:
        """Full processing: classify + generate QA."""
```

### 11.2. ChartQAPipeline

```python
class ChartQAPipeline:
    def __init__(
        self,
        config: QAPipelineConfig = None,
        classifier: GeminiChartClassifier = None,
    ): ...
    
    def run(
        self,
        source_dir: Path = None,
        session_id: str = None,
        limit: int = None,
        show_progress: bool = True,
    ) -> PipelineProgress:
        """Run the full pipeline."""
```

### 11.3. Utility Functions

```python
from tools.data_factory.services.qa_generator import (
    get_pipeline_status,    # Get current status
    collect_qa_dataset,     # Collect QA into dataset.json
)
```

## 12. Integration with Training

### 12.1. Load Dataset for Training

```python
import json
from pathlib import Path

# Load dataset
with open("data/academic_dataset/chart_qa/dataset.json") as f:
    dataset = json.load(f)

print(f"Total samples: {dataset['total_images']}")
print(f"Total QA pairs: {dataset['total_qa_pairs']}")

# Iterate samples
for sample in dataset['samples']:
    image_path = sample['image_path']
    chart_type = sample['chart_type']
    
    for qa in sample['qa_pairs']:
        question = qa['question']
        answer = qa['answer']
        q_type = qa['question_type']
        
        # Use for training...
```

### 12.2. Convert to HuggingFace Format

```python
from datasets import Dataset

# Convert to HF format
def convert_to_hf_format(dataset):
    records = []
    for sample in dataset['samples']:
        for qa in sample['qa_pairs']:
            records.append({
                'image': sample['image_path'],
                'question': qa['question'],
                'answer': qa['answer'],
                'question_type': qa['question_type'],
                'chart_type': sample['chart_type'],
            })
    return Dataset.from_list(records)

hf_dataset = convert_to_hf_format(dataset)
hf_dataset.push_to_hub("your-username/chart-qa-dataset")
```
