# SLM Training Guide — Llama 3.2 1B Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-03-02 | That Le | Updated for cloud GPU training, fixed config bugs |
| 1.0.0 | 2026-03-01 | That Le | Initial guide for incremental training workflow |

---

## 0. Bai hoc tu phien train dau tien

> Xem chi tiet tai `docs/reports/SLM_TRAINING_POSTMORTEM_V1.md`

Phien train dau tien (Session 1, local RTX 3060) da phat hien **4 loi cau hinh nghiem trong**:

| Bug | Muc do | Van de |
| --- | --- | --- |
| `max_length=512` | **FATAL** | Model khong bao gio thay ground truth (bi cat) |
| `lora_alpha=32` hardcoded | Medium | LoRA scaling bi sai khi thay doi rank |
| `pad_token = eos_token` | Medium | Llama-3 dung sai padding, generation bi cat |
| `gradient_accumulation=4` | Low | Effective batch=8 qua nho, gradient noisy |

**Ket qua**: EM=4%, Contains=8% - model gan nhu khong hoc duoc gi.

**Tat ca 4 bug da duoc sua** trong `scripts/training/train_slm_lora.py`. Phien tiep theo se dung cloud GPU.

---

## 1. Tong quan chien luoc

### 1.1. Tai sao dung Cloud GPU thay vi Local

| So sanh | Local (RTX 3060 6GB) | Cloud (A100 40GB) |
| --- | --- | --- |
| VRAM | 6 GB | 40-80 GB |
| Thoi gian/epoch | ~13-15 gio | ~1-3 gio |
| max_length kha thi | 512-1024 (OOM risk) | 4096+ |
| batch_size | 1-2 | 4-16 |
| Chi phi 3 epochs | $0 (dien + ~45h) | ~$3-9 (~3-9h) |
| Rui ro | Thermal throttling, OOM | Khong dang ke |

**Quyet dinh**: Su dung cloud GPU (RunPod / Vast.ai / Lambda) de dam bao `max_length=4096` va hoan thanh trong 1 ngay.

### 1.2. Chien luoc train

Training chay **3 epoch lien tuc** tren cloud GPU (khong can chia session nhu truoc).

```
Cloud GPU Session (~3-9h total)
    |
  Epoch 1 → Epoch 2 → Epoch 3
  ~1-3h      ~1-3h      ~1-3h
    |          |          |
  checkpoint  checkpoint  final/
    |          |          |
  [Auto eval] [Auto eval] [Full eval]
```

Neu can train tung epoch (vi gioi han thoi gian thue), van co the dung `--resume`.

---

## 2. Cau hinh he thong

### 2.1. Cloud GPU (RECOMMENDED)

| Thanh phan | Gia tri |
| --- | --- |
| GPU | NVIDIA A100 40GB (hoac L4 24GB) |
| Provider | RunPod / Vast.ai / Lambda |
| Chi phi uoc tinh | $1-3/hour |
| CUDA | 12.x |
| BF16 | Supported |
| Model | Llama-3.2-1B-Instruct |
| Dataset | `data/slm_training_v3/` (228,494 train / 26,888 val) |
| Quantization | NF4 4-bit (BitsAndBytes) |
| LoRA rank | 16 - 11.27M trainable params (0.9%) |

### 2.2. Local GPU (chi de debug/test)

| Thanh phan | Gia tri |
| --- | --- |
| GPU | NVIDIA RTX 3060 Laptop |
| VRAM | 6.0 GB |
| Luu y | Chi dung voi max-samples nho (<1000) de kiem tra config |

---

## 3. Lenh train

### Pre-training Checklist (BAT BUOC)

- [ ] Verify max_length >= p95 cua sequence lengths (4096 la an toan)
- [ ] Verify pad_token KHONG PHAI eos_token (da fix trong script)
- [ ] Check disk space cho checkpoints (moi checkpoint ~60MB)
- [ ] Upload data len cloud server (data/slm_training_v3/ + models/slm/llama-3.2-1b-instruct/)

### Option A: Full 3-epoch run (Cloud GPU - Recommended)

```bash
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

**Ket qua mong doi (A100 40GB):**
- `models/slm/llama-3.2-1b-chart-lora-v4/final/` (~60MB LoRA adapter)
- Thoi gian: 3-9 gio tong cong
- train_loss: bat dau ~3.2, ket thuc ~1.0-1.5

### Option B: Per-epoch sessions (Cloud GPU co gioi han gio)

#### Session 1 - Epoch 1

```bash
python scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \
    --epochs 1 \
    --batch-size 4 \
    --lora-rank 16 \
    --max-length 4096 \
    --gradient-accumulation-steps 4 \
    --eval-steps 500 \
    --save-steps 1000
```

#### Session 2 - Epoch 2 (Resume)

```bash
python scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \
    --epochs 2 \
    --batch-size 4 \
    --lora-rank 16 \
    --max-length 4096 \
    --gradient-accumulation-steps 4 \
    --eval-steps 500 \
    --save-steps 1000 \
    --resume
```

#### Session 3 - Epoch 3 (Resume)

```bash
python scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \
    --epochs 3 \
    --batch-size 4 \
    --lora-rank 16 \
    --max-length 4096 \
    --gradient-accumulation-steps 4 \
    --eval-steps 500 \
    --save-steps 1000 \
    --resume
```

### Option C: Quick local smoke test (RTX 3060 - Debug only)

```bash
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-debug \
    --epochs 1 \
    --batch-size 1 \
    --lora-rank 16 \
    --max-length 2048 \
    --gradient-accumulation-steps 16 \
    --max-samples 500 \
    --eval-steps 50 \
    --save-steps 100
```

> Dung de verify script chay dung truoc khi deploy len cloud. Khong dung ket qua de danh gia.

---

## 4. Cách hoạt động của --resume

`--resume` sử dụng `find_latest_checkpoint()` để scan thư mục output:

```
models/slm/llama-3.2-1b-chart-lora-v3/
    checkpoint-14250/    <- giữa chừng epoch 1
    checkpoint-28500/    <- cuối epoch 1  <-- CHỌN CÁI NÀY
    checkpoint-2/        <- smoke test (bị bỏ qua)
```

Sau đó truyền vào `trainer.train(resume_from_checkpoint=...)`. HuggingFace Trainer:
1. Load lại trọng số LoRA từ checkpoint
2. Load lại optimizer state (không cần warmup lại)
3. Tính toán step đã chạy → tự bỏ qua các epoch đã hoàn thành
4. Tiếp tục từ epoch tiếp theo

**Nếu muốn chỉ định checkpoint cụ thể:**

```bash
.venv/Scripts/python.exe scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --epochs 2 \
    --resume-from-checkpoint models/slm/llama-3.2-1b-chart-lora-v3/checkpoint-28500
```

---

## 5. Kiểm tra loss sau mỗi epoch

### 5.1. Đọc loss từ trainer_state.json

```bash
# Sau khi session kết thúc, đọc loss history
.venv/Scripts/python.exe -c "
import json
from pathlib import Path

state_path = Path('models/slm/llama-3.2-1b-chart-lora-v3/trainer_state.json')
if state_path.exists():
    state = json.loads(state_path.read_text())
    print('Best metric:', state.get('best_metric'))
    print('Best model checkpoint:', state.get('best_model_checkpoint'))
    print()
    print('Eval history (last 10):')
    evals = [e for e in state.get('log_history', []) if 'eval_loss' in e]
    for e in evals[-10:]:
        print(f\"  step={e['step']:>6}  epoch={e.get('epoch', '?'):.2f}  eval_loss={e['eval_loss']:.4f}\")
"
```

### 5.2. Biểu đồ loss bằng TensorBoard (nếu cần)

Script hiện tại đã tắt external logging (`report_to=[]`). Nếu muốn bật TensorBoard:

```bash
# Sửa SFTConfig: report_to=["tensorboard"]
# Sau đó xem biểu đồ:
.venv/Scripts/python.exe -m tensorboard.main --logdir models/slm/llama-3.2-1b-chart-lora-v3/runs
```

### 5.3. Dấu hiệu hội tụ tốt

| Trường hợp | train_loss | eval_loss | Hành động |
| --- | --- | --- | --- |
| Bình thường | Giảm đều | Giảm theo | Tiếp tục epoch tiếp |
| Overfitting | Tiếp tục giảm | Bắt đầu tăng | Dừng, dùng checkpoint tốt nhất |
| Underfitting | Giảm chậm | Giảm chậm | Thêm epoch hoặc tăng learning rate |
| Divergence | Tăng hoặc NaN | NaN | Giảm learning rate, kiểm tra data |

**Threshold mục tiêu:**

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| train_loss | ~2.0 | ~1.5 | ~1.2 |
| eval_loss | ~2.2 | ~1.7 | ~1.4 |

---

## 6. Kiểm tra định tính sau mỗi epoch

Sau khi session hoàn thành, test nhanh với `test_qwen_slm.py` (hoặc tự viết):

```bash
.venv/Scripts/python.exe scripts/pipeline/test_qwen_slm.py \
    --model-path models/slm/llama-3.2-1b-chart-lora-v3/final
```

Hoặc test inline:

```python
# Quick inference check (chạy sau mỗi epoch)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE = "models/slm/llama-3.2-1b-instruct"
LORA = "models/slm/llama-3.2-1b-chart-lora-v3/final"  # hoặc checkpoint dir

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, LORA)

messages = [
    {"role": "system", "content": "You are a chart analysis expert."},
    {"role": "user", "content": "Chart Type: bar\nOCR Texts: ['Reverue', '2021', '2022', '2023']\nDetected Elements: 3 bars\nAxis Info: x=['2021','2022','2023'], y_range=[0,25]\n\nExtract as JSON."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
print(tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

**Kỳ vọng output sau epoch 1:**
```json
{"chart_type": "bar", "x_axis": "Year", "series": [...]}
```
Có thể chưa hoàn chỉnh — chủ yếu kiểm tra model đã học được format JSON chưa.

---

## 7. Cau truc output sau 3 epochs

```
models/slm/llama-3.2-1b-chart-lora-v4/
    checkpoint-1000/         <- step 1000
    checkpoint-2000/         <- step 2000
    ...
    final/                   <- adapter cuoi cung (sau epoch 3)
        adapter_config.json
        adapter_model.safetensors  (~60MB)
        tokenizer.json
        ...
    training_info.json       <- metadata + sessions log
    trainer_state.json       <- HF Trainer state (loss history, best checkpoint)
```

`save_total_limit=2` nghia la chi giu 2 checkpoint gan nhat de tiet kiem disk.

---

## 8. Xu ly su co

### May tat giua chung / OOM crash

Khong mat cong. Checkpoint gan nhat (trong vong `save_steps` steps cuoi) van con:

```bash
# Kiem tra checkpoint nao ton tai
ls models/slm/llama-3.2-1b-chart-lora-v4/

# Resume tu do (dung --resume hoac --resume-from-checkpoint)
python scripts/training/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \
    --epochs 3 \
    --max-length 4096 \
    --resume
```

### CUDA OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

Giai phap theo thu tu:
1. Giam `--batch-size 2` (hoac 1)
2. Tang `--gradient-accumulation-steps 8` (giu effective batch=16)
3. Giam `--max-length 2048` (neu van OOM)
4. Doi GPU lon hon

> max_length KHONG DUOC giam duoi 1024. Xem postmortem: `docs/reports/SLM_TRAINING_POSTMORTEM_V1.md`

### eval_loss khong giam sau epoch 1

- Learning rate co the qua cao. Thu `--learning-rate 1e-4`
- Hoac max_length qua nho (kiem tra p95 cua sequence lengths)
- Hoac dataset co van de - kiem tra lai voi `_full_audit.py`

---

## 9. Cloud GPU Setup Guide

### 9.1. Chuan bi data de upload

```bash
# Pack du lieu can thiet (~2.5GB base model + ~150MB dataset)
tar -czf slm_training_bundle.tar.gz \
    scripts/training/train_slm_lora.py \
    scripts/evaluation/evaluate_slm.py \
    data/slm_training_v3/ \
    config/training.yaml \
    pyproject.toml

# Base model can upload rieng (2.4GB)
# hoac dung HuggingFace truc tiep: --model meta-llama/Llama-3.2-1B-Instruct
```

### 9.2. Setup tren cloud server

```bash
# 1. Install dependencies
pip install torch transformers peft bitsandbytes trl datasets accelerate

# 2. Upload data + scripts
# (dung scp, rsync, hoac cloud provider file manager)

# 3. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"

# 4. Run training (xem Section 3 cho lenh cu the)
```

### 9.3. Download ket qua ve local

```bash
# Chi can download LoRA adapter (~60MB), khong can full model
scp -r user@server:models/slm/llama-3.2-1b-chart-lora-v4/final/ \
    models/slm/llama-3.2-1b-chart-lora-v4/final/

# Download training info
scp user@server:models/slm/llama-3.2-1b-chart-lora-v4/training_info.json \
    models/slm/llama-3.2-1b-chart-lora-v4/

# Download trainer state (loss history)
scp user@server:models/slm/llama-3.2-1b-chart-lora-v4/trainer_state.json \
    models/slm/llama-3.2-1b-chart-lora-v4/
```

---

## 9. Sau khi train xong 3 epochs

### 9.1. So sanh checkpoint tot nhat

```bash
# Xem checkpoint nao co eval_loss tot nhat
python -c "
import json
from pathlib import Path
state = json.loads(Path('models/slm/llama-3.2-1b-chart-lora-v4/trainer_state.json').read_text())
print('Best checkpoint:', state['best_model_checkpoint'])
print('Best eval_loss:', state['best_metric'])
"
```

### 9.2. Danh gia chinh thuc

```bash
# Evaluate LoRA fine-tuned model
python scripts/evaluation/evaluate_slm.py \
    --base-model models/slm/llama-3.2-1b-instruct \
    --lora-path models/slm/llama-3.2-1b-chart-lora-v4/final \
    --test-data data/slm_training_v3/test.json \
    --output models/evaluation/llama-1b-lora-v4.json \
    --max-samples 500 --stratified
```

### 9.3. So sanh voi Base va Qwen (thesis contribution)

```bash
# Generate comparison table
python scripts/evaluation/evaluate_slm.py \
    --compare \
        models/evaluation/llama-1b-lora-v4.json \
        models/evaluation/llama-1b-base-quick.json \
    --compare-output models/evaluation/comparison_table_v4.md
```

| Model | EM% | Contains% | Numeric% | BLEU-1 | Latency |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B (base) | 0.0% | 9.0% | 36.2% | 0.063 | 7.04s |
| Llama-3.2-1B (LoRA v3, broken) | 4.0% | 8.0% | 17.5% | 0.281 | 1.34s |
| Llama-3.2-1B (LoRA v4, fixed) | ? | ? | ? | ? | ? |
| Qwen2.5-1.5B (LoRA) | ? | ? | ? | ? | ? |
| Gemini-2.0-flash | - | - | ~99%* | - | API |

*Bảng này là đóng góp học thuật chính của luận văn.*

---

## 10. Tom tat nhanh - checklist

### Truoc khi train (Cloud)
- [ ] Upload data + scripts len server
- [ ] Verify GPU VRAM >= 24GB
- [ ] Install dependencies (torch, transformers, peft, trl, bitsandbytes)
- [ ] Run smoke test voi --max-samples 10

### Truoc khi train (Local - debug only)
- [ ] Dong Chrome, cac app nang khac
- [ ] Kiem tra VRAM free: `nvidia-smi`
- [ ] Chi dung --max-samples nho (<1000) va --max-length 2048

### Khi train dang chay
- [ ] Monitor loss qua terminal output (logging_steps=10)
- [ ] Kiem tra disk space (checkpoint ~60MB moi cai)

### Sau khi train xong
- [ ] Doc `train_loss` va `eval_loss` cuoi epoch
- [ ] Chay evaluation: `evaluate_slm.py`
- [ ] Download LoRA adapter ve local (~60MB)
- [ ] Generate comparison table
- [ ] Cap nhat MASTER_CONTEXT.md va CHANGELOG.md
