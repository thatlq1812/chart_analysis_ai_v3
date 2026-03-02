#!/bin/bash
# =============================================================================
# Cloud GPU Training Setup Script
# =============================================================================
# Automated environment setup for SLM fine-tuning on cloud GPU instances.
# Tested on: RunPod, Vast.ai, Google Compute Engine (Deep Learning Image)
#
# Prerequisites:
#   - NVIDIA GPU with >= 24GB VRAM (RTX 3090, RTX 4090, A100)
#   - Ubuntu 20.04+ with CUDA 12.x pre-installed (use Deep Learning template)
#   - Internet access for pip packages
#
# Usage:
#   # 1. SSH into cloud instance
#   # 2. Clone repo or upload code
#   # 3. Run this script:
#   bash scripts/training/setup_cloud_training.sh
#
#   # 4. Start training in tmux (see end of script output for commands)
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "  Cloud GPU Training Setup"
echo "  Chart Analysis AI v3 - SLM Fine-tuning"
echo "=============================================="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. System Check
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/6] Checking system requirements..."

# GPU check
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU: $GPU_NAME ($GPU_VRAM MiB VRAM)"

if [ "$GPU_VRAM" -lt 20000 ]; then
    echo "  WARNING: GPU VRAM < 20GB. May need batch_size=1 and max_length=2048"
fi

# Python check
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found"
    exit 1
fi
echo "  Python: $($PYTHON_CMD --version)"

# Disk check
DISK_FREE=$(df -BG --output=avail . | tail -1 | tr -d ' G')
echo "  Disk free: ${DISK_FREE}GB"
if [ "$DISK_FREE" -lt 20 ]; then
    echo "  WARNING: Less than 20GB free disk space. Checkpoints may fill disk."
fi

# tmux check (install if missing)
if ! command -v tmux &> /dev/null; then
    echo "  Installing tmux..."
    sudo apt-get update -qq && sudo apt-get install -y -qq tmux
fi
echo "  tmux: $(tmux -V)"

echo "  System check PASSED"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Virtual Environment
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/6] Setting up Python environment..."

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv $VENV_DIR
    echo "  Created virtual environment: $VENV_DIR"
else
    echo "  Virtual environment already exists: $VENV_DIR"
fi

source $VENV_DIR/bin/activate
echo "  Activated: $(which python)"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install Dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/6] Installing dependencies..."

pip install --quiet --upgrade pip

# Core training dependencies
pip install --quiet \
    torch \
    transformers \
    peft \
    trl \
    datasets \
    accelerate \
    bitsandbytes

# Evaluation dependencies
pip install --quiet \
    numpy \
    scipy

echo "  Installed packages:"
python -c "
import torch, transformers, peft, trl, datasets, accelerate
print(f'    torch={torch.__version__} (CUDA={torch.cuda.is_available()})')
print(f'    transformers={transformers.__version__}')
print(f'    peft={peft.__version__}')
print(f'    trl={trl.__version__}')
print(f'    datasets={datasets.__version__}')
print(f'    accelerate={accelerate.__version__}')
"

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify Data
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/6] Verifying data and models..."

# Check for training data
DATA_FOUND=0
if [ -d "data/slm_training_v3" ] && [ -f "data/slm_training_v3/train.json" ]; then
    TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('data/slm_training_v3/train.json'))))")
    echo "  Full dataset: data/slm_training_v3/ ($TRAIN_COUNT train samples)"
    DATA_FOUND=1
fi

if [ -d "data/slm_training_mini" ] && [ -f "data/slm_training_mini/train.json" ]; then
    MINI_COUNT=$(python -c "import json; print(len(json.load(open('data/slm_training_mini/train.json'))))")
    echo "  Mini dataset: data/slm_training_mini/ ($MINI_COUNT train samples)"
    DATA_FOUND=1
fi

if [ "$DATA_FOUND" -eq 0 ]; then
    echo "  WARNING: No training data found!"
    echo "  Upload data/slm_training_v3/ or data/slm_training_mini/ to this machine."
    echo ""
    echo "  From local machine:"
    echo "    scp -r data/slm_training_mini/ user@server:/workspace/chart_analysis_ai_v3/data/"
    echo "    # or for full dataset:"
    echo "    scp -r data/slm_training_v3/ user@server:/workspace/chart_analysis_ai_v3/data/"
fi

# Check for base models (can also download from HuggingFace Hub)
if [ -d "models/slm/llama-3.2-1b-instruct" ]; then
    echo "  Base model: models/slm/llama-3.2-1b-instruct/ (local)"
else
    echo "  Base model: Will download from HuggingFace Hub"
    echo "  (Llama-3.2 requires access token: huggingface-cli login)"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 5. GPU Compatibility Test
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/6] Testing GPU compatibility..."

python -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem/1e9:.1f} GB)')
print(f'  BF16 supported: {torch.cuda.is_bf16_supported()}')
"

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 6. Print Training Commands
# ─────────────────────────────────────────────────────────────────────────────
echo "[6/6] Setup complete!"
echo ""
echo "=============================================="
echo "  READY TO TRAIN"
echo "=============================================="
echo ""
echo "IMPORTANT: Always use tmux to prevent session loss!"
echo ""
echo "  tmux new -s training"
echo ""
echo "--- Smoke Test (verify config, ~2 minutes) ---"
echo ""
echo "  python scripts/training/train_slm_lora.py \\"
echo "      --model llama-1b \\"
echo "      --data-dir data/slm_training_mini \\"
echo "      --smoke-test"
echo ""
echo "--- Micro-Training (model selection, ~1-2h per model) ---"
echo ""
echo "  python scripts/training/run_model_selection.py \\"
echo "      --data-dir data/slm_training_mini \\"
echo "      --models llama-1b qwen-1.5b \\"
echo "      --epochs 3"
echo ""
echo "--- Full Training (after model selection, ~3-9h) ---"
echo ""
echo "  python scripts/training/train_slm_lora.py \\"
echo "      --model llama-1b \\"
echo "      --data-dir data/slm_training_v3 \\"
echo "      --output-dir models/slm/llama-3.2-1b-chart-lora-v4 \\"
echo "      --epochs 3 \\"
echo "      --batch-size 4 \\"
echo "      --max-length 4096 \\"
echo "      --gradient-accumulation-steps 4 \\"
echo "      --eval-steps 500 \\"
echo "      --save-steps 1000"
echo ""
echo "--- Monitor GPU (in another tmux pane: Ctrl+B then %) ---"
echo ""
echo "  watch -n 2 nvidia-smi"
echo ""
echo "--- Detach from tmux (keep training running) ---"
echo ""
echo "  Ctrl+B then D"
echo ""
echo "--- Reattach ---"
echo ""
echo "  tmux attach -t training"
echo ""
echo "=============================================="
