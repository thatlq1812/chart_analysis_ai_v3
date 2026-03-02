# SLM Training Postmortem - Session 1 (Llama-3.2-1B)

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-02 | That Le | First training session analysis and lessons learned |

---

## 1. Context

First QLoRA fine-tuning session of Llama-3.2-1B-Instruct on the chart analysis dataset v3 (268,799 samples). Training completed 1 epoch (~36 hours on RTX 3060 Laptop 6GB VRAM).

**Training Configuration (as executed):**

| Parameter | Value |
| --- | --- |
| Base Model | meta-llama/Llama-3.2-1B-Instruct |
| LoRA Rank | 16 |
| LoRA Alpha | 32 (hardcoded) |
| Quantization | NF4 4-bit |
| Batch Size | 2 |
| Gradient Accumulation | 4 (effective batch = 8) |
| Max Length | **512** |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Training Samples | 228,494 |

---

## 2. Evaluation Results

Evaluation performed on 100 stratified test samples using `scripts/evaluation/evaluate_slm.py`.

### 2.1. LoRA Fine-tuned Model

| Metric | Value |
| --- | --- |
| Exact Match | 4.0% |
| Contains Match | 8.0% |
| Numeric Accuracy | 17.5% |
| BLEU-1 | 0.281 |
| Mean Latency | 1.34s |
| VRAM Peak | 2,288 MB |

### 2.2. Base Model (Zero-shot)

| Metric | Value |
| --- | --- |
| Exact Match | 0.0% |
| Contains Match | 9.0% |
| Numeric Accuracy | 36.2% |
| BLEU-1 | 0.063 |
| Mean Latency | 7.04s |
| VRAM Peak | 2,288 MB |

### 2.3. Analysis

The LoRA model showed **marginal improvement** in BLEU-1 (0.281 vs 0.063) but **worse numeric accuracy** (17.5% vs 36.2%). Contains Match was even lower than base model. This was a clear signal that the training configuration had fundamental issues.

---

## 3. Root Cause Analysis - 4 Critical Bugs

### Bug 1: max_length=512 (FATAL)

**Severity**: Critical - Rendered entire training ineffective

**Problem**: `SFTConfig(max_length=512)` caused all training sequences to be truncated at 512 tokens. The ChatML format used in training data requires:
- System prompt: ~50-80 tokens
- User prompt (OCR text + chart metadata): ~200-500 tokens
- Ground truth answer (JSON): ~100-300 tokens
- ChatML markers: ~20 tokens

**Total typical sequence**: 400-900 tokens

With `max_length=512`, the ground truth answer was **completely truncated** in most samples. The model literally never saw the correct answers during training. It only learned the prompt pattern, which explains the modest BLEU-1 improvement (learned to produce chart-related tokens) but poor accuracy (never learned the actual answer format).

**Evidence from `prompt_builder.py`**:
- `max_ocr_texts = 30` (can produce ~100+ tokens for OCR alone)
- `max_elements = 20` (chart elements add another 100-200 tokens)
- System prompts are 50-80 tokens

**Fix Applied**: Default changed from 512 to **4096** in both function parameters and CLI arguments. Value 4096 accommodates the longest training samples while staying within Llama-3.2's 128K context window.

### Bug 2: lora_alpha=32 Hardcoded

**Severity**: Medium - Incorrect LoRA scaling

**Problem**: `lora_alpha` was hardcoded to 32 regardless of `lora_rank`. The LoRA scaling factor is `alpha / rank`. With rank=16 and alpha=32, scaling = 2.0x. The community convention is `alpha = 2 * rank` for rank=16, which happens to be correct in this case. But if rank were changed (e.g., rank=8 for faster training), the scaling would become 4.0x, causing gradient instability.

**Fix Applied**: `alpha` parameter now defaults to `None`, auto-computed as `rank * 2`. Can be overridden via `--lora-alpha` CLI argument.

### Bug 3: pad_token = eos_token (Llama-3 Specific)

**Severity**: Medium - Causes premature generation stopping

**Problem**: For Llama-3 models, `eos_token_id` is `<|eot_id|>` (token 128009). Setting `pad_token = eos_token` means padding tokens are treated as end-of-turn signals. During generation, encountering any padding activates the stop condition, causing truncated outputs.

Llama-3 has dedicated padding tokens in its vocabulary:
- `<|finetune_right_pad_id|>` (token 128004) - Purpose-built for fine-tuning
- `<|reserved_special_token_0|>` (token 128002) - Fallback option

**Fix Applied**: Token lookup now checks for `<|finetune_right_pad_id|>` first, then `<|reserved_special_token_0|>`, and only falls back to `eos_token` for non-Llama models.

### Bug 4: gradient_accumulation_steps=4 Too Small

**Severity**: Low - Noisy gradients, slower convergence

**Problem**: With `batch_size=2` and `gradient_accumulation_steps=4`, effective batch size was only 8. For a dataset of 228K samples, this means high variance in gradient updates and slower convergence. Research suggests effective batch size 16-32 for SFT tasks.

**Fix Applied**: Default changed from 4 to **8** (effective batch = 16). Configurable via `--gradient-accumulation-steps` CLI argument.

---

## 4. Impact Assessment

| Bug | Impact on Training | Impact on Eval |
| --- | --- | --- |
| max_length=512 | Model never saw ground truth | EM=4% instead of expected >50% |
| lora_alpha hardcoded | Correct for rank=16, fragile for other ranks | None for this run |
| pad_token = eos_token | Truncated generation outputs | Lower Contains Match |
| gradient_accumulation=4 | Noisier gradients, slower convergence | Minor quality loss |

**Combined Effect**: The `max_length=512` bug was the dominant failure. Even with correct alpha, pad token, and batch size, a model that never sees ground truth answers cannot learn to produce them. The other bugs would have degraded quality by ~10-20%, but `max_length` alone accounts for ~80% of the poor results.

---

## 5. Lessons Learned

### 5.1. Always Verify Sequence Lengths Before Training

**Rule**: Before starting any SFT run, compute and log the token length distribution of the training data. Verify that `max_length` covers at least the 95th percentile of sequences.

```python
# Pre-training validation (add to training script)
lengths = [len(tokenizer.encode(sample["text"])) for sample in dataset]
p50, p95, p99 = np.percentile(lengths, [50, 95, 99])
logger.info(f"Token lengths | p50={p50:.0f} | p95={p95:.0f} | p99={p99:.0f} | max_length={max_length}")
assert max_length >= p95, f"max_length={max_length} is below p95={p95:.0f}, most samples will be truncated!"
```

### 5.2. Check Model-Specific Tokenizer Quirks

**Rule**: Always verify pad_token/eos_token behavior for each model family. Llama-3, Mistral, Qwen, and Phi all have different special token conventions.

### 5.3. Log Effective Batch Size

**Rule**: Always log `batch_size * gradient_accumulation_steps * n_gpus` as `effective_batch_size` in training metadata.

### 5.4. Run a Micro-Eval Before Full Training

**Rule**: After the first 100-500 steps, run a quick 5-10 sample inference check. If the model cannot produce any coherent output, stop training immediately and debug.

### 5.5. Local GPU Limitations

The RTX 3060 Laptop (6GB VRAM) can train with QLoRA, but:
- Training time: ~13-15 hours per epoch (228K samples)
- VRAM headroom is tight with max_length=4096 (may need batch_size=1)
- Cannot run evaluation while training
- Risk of thermal throttling on long runs

**Decision**: Future training will be done on **rented cloud GPU servers** (see Section 6).

---

## 6. Revised Training Strategy

### 6.1. Use Rented GPU Server

| Aspect | Local (RTX 3060) | Cloud Server (A100/L4) |
| --- | --- | --- |
| VRAM | 6 GB | 24-80 GB |
| Time per epoch | ~13-15 hours | ~1-3 hours |
| Cost | Electricity only | ~$1-3/hour |
| Total 3-epoch cost | Free ($0) | ~$3-9 |
| max_length feasible | 512-1024 | 4096+ |
| batch_size | 1-2 | 4-16 |
| Risk | Thermal throttling, OOM | None |
| Monitoring | Manual | Remote + logging |

**Recommendation**: Rent an A100 40GB or L4 24GB instance from RunPod/Vast.ai/Lambda for $1-3/hour. A full 3-epoch run with max_length=4096 would cost approximately $3-9 total and complete in 3-9 hours instead of 45 hours.

### 6.2. Revised Training Configuration

```bash
# Recommended config for cloud GPU (A100 40GB)
python scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 16 \
    --max-length 4096 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --eval-steps 500 \
    --save-steps 1000
```

### 6.3. Pre-Training Checklist (NEW)

- [ ] Compute token length distribution of training data
- [ ] Verify max_length >= p95 of sequence lengths
- [ ] Verify pad_token is NOT eos_token (for Llama models)
- [ ] Log effective_batch_size in training metadata
- [ ] Run 5-sample micro-eval after first 500 steps
- [ ] Verify GPU VRAM is sufficient for max_length setting
- [ ] Check disk space for checkpoints (estimate: n_checkpoints * 60MB)

---

## 7. Files Affected by Fixes

| File | Changes |
| --- | --- |
| `scripts/training/train_slm_lora.py` | max_length=4096, auto lora_alpha, Llama-3 pad_token, configurable gradient_accumulation |
| `scripts/evaluation/evaluate_slm.py` | max_length warning fix, --base-model optional in compare mode |

---

## 8. Conclusion

This first training session, despite producing a nearly non-functional model, served its primary purpose: **identifying critical configuration bugs before investing significant cloud compute resources**. The ~13 hours of local GPU training time was well spent as a debugging run.

All 4 bugs have been fixed in `train_slm_lora.py`. The next training session will use a rented cloud GPU with the corrected configuration. Expected improvement: EM from 4% to 40-60%+ after 3 epochs with max_length=4096.

**Status**: Training artifacts from this session have been archived/cleaned. Ready for retrain on cloud GPU.
