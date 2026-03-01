# SLM Training Guide — Llama 3.2 1B Chart Analysis

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-01 | That Le | Initial guide for incremental training workflow |

---

## 1. Tổng quan chiến lược

Training được chia thành **3 session riêng biệt**, mỗi session 1 epoch. Sau mỗi session, checkpoint tự động được lưu và session tiếp theo resume từ đó.

```
Session 1 (Ngày 1)         Session 2 (Ngày 2)         Session 3 (Ngày 3)
    |                           |                           |
  Epoch 1                     Epoch 2                     Epoch 3
  ~13-15h                     ~13-15h                     ~13-15h
    |                           |                           |
  checkpoint-28500            checkpoint-57000            final/
    |                           |                           |
  [Kiểm tra loss]             [Kiểm tra loss]             [Đánh giá]
  [Quyết định tiếp?]          [Quyết định tiếp?]          [So sánh models]
```

**Lý do train từng epoch:**
- RTX 3060 Laptop: ~13-15 giờ/epoch, không thể bỏ máy 45h liên tục
- Có cơ hội kiểm tra loss curve sau mỗi epoch trước khi tiếp tục
- Nếu overfitting xuất hiện ở epoch 2, dừng được ngay
- Checkpoint sau mỗi epoch là "điểm an toàn" không bị mất công

---

## 2. Cấu hình hệ thống

| Thành phần | Giá trị |
| --- | --- |
| GPU | NVIDIA RTX 3060 Laptop |
| VRAM | 6.0 GB |
| CUDA | 11.8 |
| BF16 | Supported (native) |
| Model | Llama-3.2-1B-Instruct (local: `models/slm/llama-3.2-1b-instruct/`) |
| Dataset | `data/slm_training_v3/` (228,494 train / 26,888 val) |
| Quantization | NF4 4-bit (BitsAndBytes) — VRAM ~3.5GB |
| LoRA rank | 16 — 11.27M trainable params (0.9%) |

---

## 3. Lệnh train từng session

### Session 1 — Epoch 1 (Fresh start)

```bash
.venv/Scripts/python.exe scripts/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v3 \
    --epochs 1 \
    --batch-size 2 \
    --lora-rank 16 \
    --max-length 512 \
    --eval-steps 1000 \
    --save-steps 2000
```

**Kết quả mong đợi:**
- `models/slm/llama-3.2-1b-chart-lora-v3/checkpoint-28500/` (approx)
- `train_loss` bắt đầu ~3.2, kết thúc ~1.5-2.0
- Thời gian: 13-15 giờ

---

### Session 2 — Epoch 2 (Resume)

```bash
.venv/Scripts/python.exe scripts/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v3 \
    --epochs 2 \
    --batch-size 2 \
    --lora-rank 16 \
    --max-length 512 \
    --eval-steps 1000 \
    --save-steps 2000 \
    --resume
```

> `--epochs 2` = tổng epochs mục tiêu là 2. Trainer tự biết epoch 1 đã xong, chỉ train epoch 2.
> `--resume` = tự động tìm checkpoint mới nhất trong `output-dir`.

---

### Session 3 — Epoch 3 (Resume)

```bash
.venv/Scripts/python.exe scripts/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --output-dir models/slm/llama-3.2-1b-chart-lora-v3 \
    --epochs 3 \
    --batch-size 2 \
    --lora-rank 16 \
    --max-length 512 \
    --eval-steps 1000 \
    --save-steps 2000 \
    --resume
```

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
.venv/Scripts/python.exe scripts/train_slm_lora.py \
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
.venv/Scripts/python.exe scripts/test_qwen_slm.py \
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

## 7. Cấu trúc output sau 3 sessions

```
models/slm/llama-3.2-1b-chart-lora-v3/
    checkpoint-2000/         <- cuối step 2000 (epoch 1)
    checkpoint-4000/         <- cuối step 4000 (epoch 1)
    ...
    checkpoint-28500/        <- cuối epoch 1  [CHECKPOINT EPOCH 1]
    checkpoint-30500/        <- đầu epoch 2
    ...
    checkpoint-57000/        <- cuối epoch 2  [CHECKPOINT EPOCH 2]
    ...
    checkpoint-85500/        <- cuối epoch 3  [CHECKPOINT EPOCH 3]
    final/                   <- adapter cuối cùng (sau epoch 3)
        adapter_config.json
        adapter_model.safetensors  (~50MB)
        tokenizer.json
        ...
    training_info.json       <- metadata + sessions log
    trainer_state.json       <- HF Trainer state (loss history, best checkpoint)
```

`save_total_limit=2` nghĩa là chỉ giữ 2 checkpoint gần nhất để tiết kiệm disk.

---

## 8. Xử lý sự cố

### Máy tắt giữa chừng / OOM crash

Không mất công. Checkpoint gần nhất (trong vòng `save_steps=2000` steps cuối) vẫn còn:

```bash
# Kiểm tra checkpoint nào tồn tại
ls models/slm/llama-3.2-1b-chart-lora-v3/

# Resume từ đó (dùng --resume hoặc --resume-from-checkpoint)
.venv/Scripts/python.exe scripts/train_slm_lora.py \
    --model llama-1b \
    --data-dir data/slm_training_v3 \
    --epochs 1 \
    --resume
```

### CUDA OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

Giải pháp theo thứ tự:
1. Đóng các ứng dụng khác dùng GPU (Chrome, game, v.v.)
2. Kiểm tra VRAM thực tế: `nvidia-smi`
3. Giảm `--max-length 384` (từ 512 xuống)
4. Giảm `--batch-size 1` (từ 2 xuống, tăng `gradient_accumulation_steps` lên 8 trong code)

### eval_loss không giảm sau epoch 1

- Learning rate có thể quá cao. Thử `--learning-rate 1e-4`
- Hoặc dataset có vấn đề — kiểm tra lại với `_full_audit.py`

---

## 9. Sau khi train xong 3 epochs

### 9.1. So sánh checkpoint tốt nhất

```bash
# Xem checkpoint nào có eval_loss tốt nhất
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
state = json.loads(Path('models/slm/llama-3.2-1b-chart-lora-v3/trainer_state.json').read_text())
print('Best checkpoint:', state['best_model_checkpoint'])
print('Best eval_loss:', state['best_metric'])
"
```

### 9.2. Đánh giá chính thức

```bash
# Khi scripts/evaluate_slm.py được tạo:
.venv/Scripts/python.exe scripts/evaluate_slm.py \
    --model-path models/slm/llama-3.2-1b-chart-lora-v3/final \
    --test-data data/slm_training_v3/test.json \
    --output models/evaluation/llama1b_v3_eval.json
```

### 9.3. So sánh với Qwen (thesis contribution)

| Model | train_loss | eval_loss | JSON Valid % | Latency |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B (base) | - | - | ? | ? |
| Llama-3.2-1B (LoRA v3) | ? | ? | ? | ? |
| Qwen2.5-1.5B (LoRA v3) | ? | ? | ? | ? |
| Gemini-2.0-flash | - | - | ~99%\* | API |

*Bảng này là đóng góp học thuật chính của luận văn.*

---

## 10. Tóm tắt nhanh — checklist hàng ngày

### Trước khi bật train
- [ ] Đóng Chrome, các app nặng khác
- [ ] Kiểm tra VRAM free: `nvidia-smi`
- [ ] Kiểm tra disk còn trống: checkpoint ~500MB mỗi cái, cần ít nhất 5GB free

### Khi train đang chạy
- [ ] Không cần theo dõi real-time, để máy chạy
- [ ] Nếu muốn xem log: kiểm tra terminal output (logging_steps=10)

### Sau khi session kết thúc
- [ ] Đọc `train_loss` và `eval_loss` cuối epoch
- [ ] Chạy 1-2 inference test để xem output quality
- [ ] Ghi lại kết quả vào experiment log
- [ ] Quyết định tiếp epoch tiếp hay dừng
