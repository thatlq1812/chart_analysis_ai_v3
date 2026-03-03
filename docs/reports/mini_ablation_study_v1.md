# Mini Ablation Study Report - Llama-3.2-1B QLoRA

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-03 | That Le | Hyperparameter sensitivity analysis on micro dataset |

---

## 1. Objective

Conduct a controlled hyperparameter sensitivity study on a small, balanced dataset to:

1. Verify the training pipeline correctness after bug fixes (see `SLM_TRAINING_POSTMORTEM_V1.md`)
2. Identify optimal hyperparameter ranges before committing to full-scale training (228K samples, ~36h/epoch)
3. Measure the impact of learning rate, LoRA rank, and gradient accumulation on convergence

---

## 2. Experimental Setup

### 2.1. Base Model

| Property | Value |
| --- | --- |
| Model | Llama-3.2-1B-Instruct |
| Parameters | ~1.24B total |
| Quantization | NF4 4-bit (BitsAndBytes) |
| Precision | bf16 (mixed precision) |
| Hardware | NVIDIA RTX 3060 Laptop GPU (6GB VRAM) |

### 2.2. Micro Dataset

Extracted from SLM Training Dataset v3 (268,799 total samples) using stratified per-type sampling.

| Split | Samples | Strategy |
| --- | --- | --- |
| Train | 800 | 100 samples per chart type x 8 types |
| Validation | 160 | 20 samples per chart type x 8 types |
| Test | 13,417 | Full v3 test set (unchanged) |

**Chart types (balanced):** area, bar, box, heatmap, histogram, line, pie, scatter

**Question types (11 categories):** comparison (24.6%), structural (17.0%), extraction (13.0%), interpolation (11.2%), why_reasoning (7.5%), multi_hop (7.4%), percentage_change (6.4%), range (4.9%), trend (4.1%), threshold (2.4%), counting (1.5%)

### 2.3. Token Length Analysis

Before training, the full v3 dataset was tokenized with the Llama-3.2-1B tokenizer to validate `max_seq_length`:

| Statistic | Tokens |
| --- | --- |
| Min | 85 |
| Max | 1,576 |
| Mean | 224 |
| Median | 223 |
| P90 | 309 |
| P95 | 342 |
| P99 | 412 |

**Token distribution:**

| Bucket | Count | Percentage |
| --- | --- | --- |
| <= 128 | 26,913 | 11.8% |
| <= 256 | 137,442 | 60.2% |
| <= 512 | 63,606 | 27.8% |
| <= 1,024 | 321 | 0.1% |
| <= 2,048 | 212 | 0.1% |
| > 4,096 | 0 | 0.0% |

**Conclusion:** 99.8% of samples fit within 512 tokens. The longest sample (1,576 tokens) is a heatmap with dense OCR data. `max_seq_length=1024` covers 99.98% of data while reducing padding waste by 75% compared to the previous 4096 setting.

**Conversation structure (3 roles):**

| Role | Min | Max | Mean | Content |
| --- | --- | --- | --- | --- |
| System | 26 | 28 | 28 | Fixed prompt: chart analysis/reasoning expert |
| User | 23 | 1,482 | 148 | Chart metadata (type, title, legend, ticks, data labels, axis info, elements) + question |
| Assistant | 2 | 493 | 15 | Answer (usually short, exact values) |

### 2.4. Fixed Parameters (Shared Across All Runs)

| Parameter | Value |
| --- | --- |
| Base Model | Llama-3.2-1B-Instruct |
| Quantization | NF4 4-bit |
| max_seq_length | 4096 (*) |
| Batch Size | 2 |
| bf16 | true |
| Optimizer | paged_adamw_32bit |
| LR Scheduler | cosine |
| Warmup Ratio | 0.05 |
| Weight Decay | 0.01 |
| Epochs | 10 |
| Eval Steps | 10 |
| Gradient Checkpointing | true |
| LoRA Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA Dropout | 0.05 |

(*) Note: `max_seq_length` was 4096 for this experiment run. Subsequent runs will use 1024 based on the token analysis above.

---

## 3. Ablation Configurations

| Run | Variable Changed | LoRA Rank | LoRA Alpha | Learning Rate | Grad Accum | Eff. Batch Size |
| --- | --- | --- | --- | --- | --- | --- |
| **baseline** | (reference) | 16 | 32 | 2e-4 | 8 | 16 |
| **low_lr** | Learning rate | 16 | 32 | **1e-5** | 8 | 16 |
| **rank8** | LoRA capacity | **8** | **16** | 2e-4 | 8 | 16 |
| **no_accum** | Batch dynamics | 16 | 32 | 2e-4 | **1** | **2** |

---

## 4. Results

### 4.1. Training Metrics Summary

| Run | Final Train Loss | Final Eval Loss | Best Eval Loss | Final Train Acc | Final Eval Acc |
| --- | --- | --- | --- | --- | --- |
| **baseline** | 0.2687 | 1.3069 | **0.9464** | 93.2% | 78.3% |
| **low_lr** | 1.2279 | **1.2026** | 1.2024 | 77.9% | 76.9% |
| **rank8** | 0.4799 | **1.1321** | 0.9491 | 88.7% | **78.9%** |
| **no_accum** | **0.0736** | 1.5399 | **0.9405** | **97.6%** | **78.9%** |

### 4.2. Analysis by Hyperparameter

#### Learning Rate (baseline vs low_lr)

| Metric | baseline (lr=2e-4) | low_lr (lr=1e-5) | Delta |
| --- | --- | --- | --- |
| Final Train Loss | 0.2687 | 1.2279 | +0.9592 |
| Final Eval Loss | 1.3069 | 1.2026 | -0.1043 |
| Best Eval Loss | 0.9464 | 1.2024 | +0.2560 |
| Train Acc | 93.2% | 77.9% | -15.3pp |
| Eval Acc | 78.3% | 76.9% | -1.4pp |

**Interpretation:** lr=1e-5 is too conservative. After 10 epochs, the model barely moved from its initial state (train_loss still >1.2). The final eval loss appears lower than baseline only because baseline has already entered the overfitting regime. The best eval loss (early stopping metric) clearly favors baseline (0.9464 vs 1.2024). For the full dataset, lr=2e-4 with cosine decay is appropriate; lr=1e-5 would require 50+ epochs to converge.

#### LoRA Rank (baseline vs rank8)

| Metric | baseline (rank=16) | rank8 (rank=8) | Delta |
| --- | --- | --- | --- |
| Final Train Loss | 0.2687 | 0.4799 | +0.2112 |
| Final Eval Loss | 1.3069 | 1.1321 | -0.1748 |
| Best Eval Loss | 0.9464 | 0.9491 | +0.0027 |
| Train Acc | 93.2% | 88.7% | -4.5pp |
| Eval Acc | 78.3% | 78.9% | +0.6pp |

**Interpretation:** rank=8 provides a strong regularization effect. It achieves slightly better final eval loss (1.13 vs 1.31) and eval accuracy (78.9% vs 78.3%) with fewer parameters. The best eval loss is nearly identical (~0.95). For small datasets, rank=8 generalizes better; for the full 228K dataset, rank=16 may be justified since overfitting risk is lower.

#### Gradient Accumulation (baseline vs no_accum)

| Metric | baseline (accum=8, eff_bs=16) | no_accum (accum=1, eff_bs=2) | Delta |
| --- | --- | --- | --- |
| Final Train Loss | 0.2687 | 0.0736 | -0.1951 |
| Final Eval Loss | 1.3069 | 1.5399 | +0.2330 |
| Best Eval Loss | 0.9464 | 0.9405 | -0.0059 |
| Train Acc | 93.2% | 97.6% | +4.4pp |
| Eval Acc | 78.3% | 78.9% | +0.6pp |

**Interpretation:** Without gradient accumulation (eff_bs=2), the model overfits aggressively (train_loss=0.07, nearly memorizing). The noisy small-batch gradients enable fast initial descent (best_eval_loss=0.94, slightly better than baseline) but lead to severe generalization degradation by epoch 10 (eval_loss=1.54). For stable training on the full dataset, gradient accumulation of 8 (effective batch size 16) is strongly recommended.

---

## 5. Key Findings

### 5.1. Pipeline Validation

The training pipeline is **confirmed working correctly** after the bug fixes documented in `SLM_TRAINING_POSTMORTEM_V1.md`:
- All 4 runs completed successfully (exit code 0)
- Train loss consistently decreased across epochs
- Model achieved >93% train accuracy on micro dataset, confirming it can learn the chart QA task format
- Eval accuracy reached ~78-79% across configurations, indicating meaningful generalization even on 800 samples

### 5.2. Overfitting Characteristics

On 800 samples with 10 epochs, the overfitting gap provides useful information:

| Run | Train-Eval Loss Gap | Overfitting Severity |
| --- | --- | --- |
| baseline | 1.04 | Moderate |
| low_lr | 0.03 | None (underfitting) |
| rank8 | 0.65 | Mild |
| no_accum | 1.47 | Severe |

This confirms that with the full 228K dataset, overfitting risk is minimal for rank=8 or rank=16 with proper regularization.

### 5.3. Recommended Configuration for Full Training

Based on ablation results, the recommended configuration for full-scale training:

| Parameter | Recommended | Rationale |
| --- | --- | --- |
| Learning Rate | 2e-4 | Strong convergence, cosine decay handles later phases |
| LoRA Rank | 16 | Full dataset has enough samples to utilize; rank=8 as fallback |
| LoRA Alpha | 32 | Standard 2x rank ratio |
| Grad Accumulation | 8 | Stable gradients, prevents overfitting |
| Effective Batch Size | 16 | Good balance for 6GB VRAM |
| max_seq_length | **1024** | Covers 99.98% of data, reduces padding 75% |
| Epochs | 3-5 | Extrapolated from convergence rate on mini data |

### 5.4. Expected Training Time (Full Dataset)

| Scenario | Samples | Steps/Epoch | Est. Time/Epoch | Total |
| --- | --- | --- | --- | --- |
| RTX 3060 (6GB) | 228,494 | ~14,281 | ~35-40h | 105-200h (3-5 epochs) |
| A100 (80GB) | 228,494 | ~14,281 | ~3-4h | 9-20h (3-5 epochs) |

---

## 6. Artifacts

| Artifact | Path |
| --- | --- |
| Ablation output directory | `models/slm/llama-3.2-1b-instruct-ablation/` |
| Baseline adapter (best) | `models/slm/llama-3.2-1b-instruct-ablation/ablation_baseline/final/` |
| Rank8 adapter | `models/slm/llama-3.2-1b-instruct-ablation/ablation_rank8/final/` |
| Ablation summary JSON | `models/slm/llama-3.2-1b-instruct-ablation/ablation_summary.json` |
| Ablation report | `models/slm/llama-3.2-1b-instruct-ablation/ablation_report.txt` |
| Micro dataset | `data/slm_training_mini/` (800 train, 160 val) |
| Ablation runner script | `scripts/training/run_mini_ablation.py` |

---

## 7. Next Steps

1. **Multi-model comparison**: Run baseline config on Qwen-2.5-1.5B and Qwen-2.5-0.5B for cross-architecture comparison
2. **Evaluation on test set**: Run `evaluate_slm.py` on ablation adapters against 13K test samples
3. **Full-scale training**: Train best configuration on full 228K dataset (cloud GPU recommended)
4. **Integrate into thesis**: Transfer results tables and analysis to Chapter 5 (Results and Discussion)
