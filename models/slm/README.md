# SLM Model Registry

Model weights for fine-tuned Small Language Models.

## Directory Structure

```
models/slm/
├── README.md                           # This file
├── qwen2.5-1.5b-chart-lora/           # LoRA adapter (after training)
│   ├── final/                          # Best checkpoint
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── training_info.json              # Hyperparams, dataset stats
│   ├── eval_results.json               # Evaluation metrics
│   └── tensorboard/                    # Training logs
└── qwen2.5-1.5b-chart-merged/         # Merged model (for deployment)
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## Current Models

| Model | Base | Method | Status | Accuracy |
| --- | --- | --- | --- | --- |
| (none yet) | Qwen2.5-1.5B-Instruct | QLoRA r=16 | PLANNED | - |

## Training

See `config/training.yaml` for hyperparameters.
See `.github/instructions/module-training.instructions.md` for full framework.

```bash
# Train
.venv/Scripts/python.exe scripts/train_slm_lora.py --config config/training.yaml

# Evaluate
.venv/Scripts/python.exe scripts/evaluate_slm.py --model-path models/slm/qwen2.5-1.5b-chart-lora/final

# Merge LoRA into base model
.venv/Scripts/python.exe scripts/merge_slm_lora.py --lora-path models/slm/qwen2.5-1.5b-chart-lora/final
```

## Rules

- Only LoRA adapters (~50MB) are stored here, NOT full base models (~3GB)
- Full model weights are cached in `~/.cache/huggingface/hub/`
- Merged models are created on-demand for deployment
- All model files are in `.gitignore` (except this README)
