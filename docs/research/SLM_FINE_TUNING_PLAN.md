# SLM Fine-tuning Plan: Qwen 2.5-1.5B-Instruct

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-02-XX | That Le | Initial fine-tuning plan |

## 1. Model Selection

**Chosen Model:** `Qwen/Qwen2.5-1.5B-Instruct`

| Property | Value |
|----------|-------|
| Parameters | 1.5B |
| VRAM (FP16) | ~3GB |
| VRAM (4-bit) | ~1.5GB |
| License | Apache 2.0 |
| Context Length | 32K tokens |

**Why Qwen 2.5?**
- Strong multilingual support (EN + VI)
- Excellent instruction following
- Apache 2.0 license (commercial friendly)
- Better math/reasoning for value extraction
- Efficient for edge deployment

## 2. Task Definition

### 2.1 Primary Tasks

| Task | Input | Output | Priority |
|------|-------|--------|----------|
| **Chart QA** | Question + Chart metadata | Answer | High |
| **Value Extraction** | Chart metadata | Structured JSON | High |
| **Description Generation** | Chart data | Academic text | Medium |
| **OCR Correction** | Raw OCR text | Corrected text | Medium |

### 2.2 Input/Output Format

```
[Input Format]
<chart_type>: {type}
<title>: {title or "Unknown"}
<x_axis>: {label or "Unknown"}
<y_axis>: {label or "Unknown"}
<data_points>: {list of (x, y) or (label, value)}
<question>: {user question}

[Output Format]
<answer>: {direct answer}
<reasoning>: {optional step-by-step reasoning}
<confidence>: {high/medium/low}
```

## 3. Dataset Preparation

### 3.1 Source Data

| Source | Count | Description |
|--------|-------|-------------|
| Individual JSON files | 32,364 | QA pairs per chart |
| QA pairs total | ~270,000 | 8-10 QA per chart |

### 3.2 Data Split Strategy

| Split | Ratio | Estimated Count |
|-------|-------|-----------------|
| Train | 80% | ~216,000 QA |
| Validation | 10% | ~27,000 QA |
| Test | 10% | ~27,000 QA |

### 3.3 Conversation Format (ChatML)

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a chart analysis expert..."
    },
    {
      "role": "user", 
      "content": "Chart metadata + Question"
    },
    {
      "role": "assistant",
      "content": "Answer with reasoning"
    }
  ]
}
```

## 4. Prompt Templates

### 4.1 Chart QA Template (Primary)

**System Prompt:**
```
You are a chart analysis expert. Answer questions about charts accurately and concisely based on the provided metadata. Always provide your confidence level.
```

**User Prompt:**
```
Chart Type: {chart_type}
Title: {title}
X-Axis: {x_label}
Y-Axis: {y_label}
Data: {data_points}

Question: {question}
```

**Assistant Response:**
```
Answer: {answer}
Confidence: {high|medium|low}
```

### 4.2 Value Extraction Template

**System Prompt:**
```
You are a data extraction specialist. Extract structured information from chart metadata and return as valid JSON.
```

**User Prompt:**
```
Extract the following from this chart:
- All data points with their labels and values
- Units if present
- Trends or patterns

Chart Type: {chart_type}
Title: {title}
Axis Labels: X={x_label}, Y={y_label}
Raw Data: {raw_extracted_text}
```

**Assistant Response:**
```json
{
  "data_points": [
    {"label": "Q1", "value": 45, "unit": "million"}
  ],
  "trend": "increasing",
  "summary": "..."
}
```

### 4.3 Description Generation Template

**System Prompt:**
```
You are an academic writing assistant. Generate concise, professional descriptions of charts suitable for scientific papers.
```

**User Prompt:**
```
Generate a 2-3 sentence description for this chart:

Type: {chart_type}
Title: {title}
Data Summary: {data_summary}
Key Observations: {observations}
```

**Assistant Response:**
```
{academic_description}
```

### 4.4 OCR Correction Template

**System Prompt:**
```
You are an OCR error correction specialist. Fix common OCR mistakes in text extracted from charts. Common errors include: 'l' -> '1', 'O' -> '0', 'S' -> '5', missing spaces, etc.
```

**User Prompt:**
```
Fix OCR errors in these chart labels:
{ocr_text_list}

Context: This is from a {chart_type} chart.
```

**Assistant Response:**
```
Corrected:
{corrected_text_list}
```

## 5. Training Configuration

### 5.1 LoRA Configuration

```yaml
# LoRA parameters for efficient fine-tuning
lora:
  r: 16                    # Rank
  alpha: 32                # Scaling factor
  dropout: 0.05            # Dropout rate
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### 5.2 Training Hyperparameters

```yaml
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.03
  lr_scheduler: cosine
  weight_decay: 0.01
  max_seq_length: 2048
  
  # Memory optimization
  fp16: true
  gradient_checkpointing: true
```

### 5.3 Hardware Requirements

| Config | VRAM | Training Time (est.) |
|--------|------|---------------------|
| Full FP16 | ~6GB | N/A (too large) |
| LoRA FP16 | ~4GB | ~8 hours |
| LoRA 4-bit | ~3GB | ~10 hours |

## 6. Evaluation Metrics

### 6.1 Chart QA Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Exact Match** | Answer matches exactly | >70% |
| **F1 Score** | Token-level overlap | >85% |
| **Numerical Accuracy** | Within 5% of correct | >80% |
| **Confidence Calibration** | High conf = correct | >75% |

### 6.2 Value Extraction Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **JSON Valid** | Output is valid JSON | >95% |
| **Field Accuracy** | Correct fields extracted | >85% |
| **Value Accuracy** | Correct numeric values | >80% |

### 6.3 Description Quality

| Metric | Description | Target |
|--------|-------------|--------|
| **BLEU Score** | N-gram overlap | >40 |
| **Factual Accuracy** | No hallucinated data | >90% |
| **Academic Style** | Human evaluation | >4.0/5 |

## 7. Implementation Plan

### Phase 1: Data Preparation (Week 1)

- [ ] Merge all individual JSON files
- [ ] Convert to ChatML format
- [ ] Create train/val/test splits
- [ ] Validate data quality

### Phase 2: Initial Training (Week 1-2)

- [ ] Setup training environment
- [ ] Train on small subset first
- [ ] Validate prompt templates
- [ ] Iterate on data format

### Phase 3: Full Training (Week 2-3)

- [ ] Train on full dataset
- [ ] Monitor loss curves
- [ ] Save checkpoints
- [ ] Evaluate on validation set

### Phase 4: Evaluation and Optimization (Week 3-4)

- [ ] Run comprehensive evaluation
- [ ] Analyze failure cases
- [ ] Fine-tune hyperparameters
- [ ] Final model selection

## 8. Directory Structure

```
models/
  slm/
    qwen2.5-1.5b-chart-qa/
      adapter_config.json      # LoRA config
      adapter_model.safetensors # LoRA weights
      tokenizer/               # Tokenizer files
      
data/
  slm_training/
    train.json               # Training data (ChatML)
    val.json                 # Validation data
    test.json                # Test data
    
scripts/
  prepare_slm_data.py        # Convert QA pairs to ChatML
  train_qwen_lora.py         # LoRA fine-tuning script
  evaluate_slm.py            # Evaluation script
```

## 9. Next Steps

1. **Create data preparation script** - Convert 32K JSON files to training format
2. **Setup training script** - LoRA fine-tuning with Hugging Face
3. **Run initial training** - Small subset validation
4. **Full training** - Complete dataset
5. **Evaluation** - Comprehensive testing
6. **Integration** - Add to Stage 4 pipeline
