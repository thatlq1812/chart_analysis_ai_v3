#!/usr/bin/env python3
"""
Fine-tune Chart Analysis SLM with LoRA (trl 0.29 compatible).

Supports:
- Qwen2.5-0.5B / 1.5B-Instruct
- Llama-3.2-1B / 3B-Instruct

Usage:
    # Full training using YAML config (RECOMMENDED):
    python scripts/training/train_slm_lora.py --config config/training.yaml

    # Config + dynamic override (ablation study):
    python scripts/training/train_slm_lora.py --config config/training.yaml \
        --override slm_training.training.learning_rate=1e-5 \
        --override slm_training.lora.rank=32

    # Choose model:
    python scripts/training/train_slm_lora.py --model qwen-1.5b
    python scripts/training/train_slm_lora.py --model llama-3b

    # Smoke test (2 steps, verifies pipeline end-to-end):
    python scripts/training/train_slm_lora.py --smoke-test
    python scripts/training/train_slm_lora.py --smoke-test --model qwen-0.5b

    # Custom hyperparameters:
    python scripts/training/train_slm_lora.py --epochs 5 --batch-size 2 --lora-rank 32

    # --- INCREMENTAL / RESUME TRAINING ---
    # Session 1: train epoch 1
    python scripts/training/train_slm_lora.py --model llama-1b --data-dir data/slm_training_v3 --epochs 1

    # Session 2: resume and train up to epoch 2 (auto-detects latest checkpoint)
    python scripts/training/train_slm_lora.py --model llama-1b --data-dir data/slm_training_v3 --epochs 2 --resume

    # Session 3: resume up to epoch 3
    python scripts/training/train_slm_lora.py --model llama-1b --data-dir data/slm_training_v3 --epochs 3 --resume

    # Resume from explicit checkpoint path:
    python scripts/training/train_slm_lora.py --epochs 2 --resume-from-checkpoint models/slm/llama-3.2-1b-instruct-chart-lora/checkpoint-28500

    # Experiment tracking with WandB:
    python scripts/training/train_slm_lora.py --config config/training.yaml \
        --tracker wandb

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes

Output:
    runs/<model>_<timestamp>/  (isolated run directory with frozen config)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.run_manager import RunManager
from src.training.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry (mirrors download_models.py)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models" / "slm"
# v3 dataset: 268k samples (all 8 chart types, Stage3 geometric features + axis info)
# v2 dataset: 32k balanced samples (legacy baseline)
DATA_PATH = PROJECT_ROOT / "data" / "slm_training_v3"
DATA_PATH_LEGACY = PROJECT_ROOT / "data" / "slm_training_v2"
DEFAULT_MODEL = "llama-1b"

MODEL_REGISTRY: Dict[str, Dict] = {
    "qwen-0.5b": {
        "local_name": "qwen2.5-0.5b-instruct",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "chat_format": "qwen",
        "trust_remote_code": True,
    },
    "qwen-1.5b": {
        "local_name": "qwen2.5-1.5b-instruct",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "chat_format": "qwen",
        "trust_remote_code": True,
    },
    "llama-1b": {
        "local_name": "llama-3.2-1b-instruct",
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_format": "llama",
        "trust_remote_code": False,
    },
    "llama-3b": {
        "local_name": "llama-3.2-3b-instruct",
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "chat_format": "llama",
        "trust_remote_code": False,
    },
}


def check_dependencies() -> None:
    """Check required packages are installed."""
    required = ["transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"]
    missing = [pkg for pkg in required if not __import__("importlib").util.find_spec(pkg)]
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        sys.exit(1)
    logger.info("All dependencies installed")


def resolve_model_path(model_key: str) -> str:
    """
    Return local path if model is downloaded, else HuggingFace repo_id.

    Args:
        model_key: Key from MODEL_REGISTRY (e.g. 'qwen-1.5b')

    Returns:
        Path string suitable for AutoModel.from_pretrained
    """
    entry = MODEL_REGISTRY[model_key]
    local_path = MODELS_DIR / entry["local_name"]
    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Using local model: {local_path}")
        return str(local_path)
    logger.warning(
        f"Local model not found at {local_path}. "
        f"Falling back to HuggingFace Hub: {entry['repo_id']}\n"
        f"Run: python scripts/utils/download_models.py --model {model_key}"
    )
    return entry["repo_id"]


def format_conversation(messages: list, tokenizer, chat_format: str) -> str:
    """
    Format a conversation into a single training string.

    Tries tokenizer.apply_chat_template first; falls back to hand-crafted
    ChatML (Qwen) or Llama-3 format if the tokenizer does not support it.

    Args:
        messages: List of {'role': ..., 'content': ...} dicts
        tokenizer: HuggingFace tokenizer
        chat_format: 'qwen' or 'llama'

    Returns:
        Formatted string ready for SFTTrainer
    """
    # Prefer native chat template when available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass

    # Fallback: manual formatting
    if chat_format == "llama":
        # Llama-3 BOS / header format
        text = "<|begin_of_text|>"
        for msg in messages:
            text += (
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>\n"
            )
        return text
    else:
        # ChatML (Qwen default)
        text = ""
        for msg in messages:
            text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        return text


def load_dataset_split(data_path: Path, split: str, tokenizer, chat_format: str):
    """
    Load a JSON split and format conversations into training text.

    Args:
        data_path: Directory containing train.json / val.json
        split: 'train' or 'val'
        tokenizer: Tokenizer for apply_chat_template
        chat_format: 'qwen' or 'llama'

    Returns:
        HuggingFace Dataset with 'text' column
    """
    from datasets import Dataset

    file_path = data_path / f"{split}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        messages = item.get("conversations", item.get("messages", []))
        text = format_conversation(messages, tokenizer, chat_format)
        records.append({"text": text})

    logger.info(f"{split}: {len(records)} examples loaded")
    return Dataset.from_list(records)


def detect_lora_target_modules(model_path: str) -> List[str]:
    """
    Auto-detect LoRA target modules from model config.json.

    Reads the model's config.json (without loading weights) and maps the
    model_type / architectures to the correct linear layer names for LoRA.
    Falls back to a safe universal set if the architecture is unknown.

    Args:
        model_path: Local path or HuggingFace repo_id

    Returns:
        List of module name patterns to inject LoRA adapters into
    """
    # Architecture -> known target_modules mapping
    # Covers most decoder-only transformer variants used for SLM fine-tuning
    ARCH_TO_TARGETS: Dict[str, List[str]] = {
        # Qwen2 / Qwen2.5 family
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen2moe": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Llama / Llama-2 / Llama-3 family
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Mistral / Mixtral
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Phi / Phi-3
        "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        # Gemma / Gemma2
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # GPT-2
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        # OPT
        "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        # Falcon
        "RefinedWeb": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    }

    # Safest universal fallback (works for most LlamaForCausalLM-derived models)
    UNIVERSAL_FALLBACK = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    import json
    from pathlib import Path

    config_candidates = [
        Path(model_path) / "config.json",   # local path
    ]

    model_config: Optional[Dict] = None

    # Try loading local config.json first (fast, no download)
    for config_path in config_candidates:
        if config_path.exists():
            try:
                model_config = json.loads(config_path.read_text(encoding="utf-8"))
                logger.info(f"Loaded model config from {config_path}")
                break
            except Exception as exc:
                logger.warning(f"Failed to read {config_path}: {exc}")

    # If not local, fetch from HuggingFace Hub (config only, no weights)
    if model_config is None:
        try:
            from huggingface_hub import hf_hub_download
            tmp = hf_hub_download(repo_id=model_path, filename="config.json")
            model_config = json.loads(Path(tmp).read_text(encoding="utf-8"))
            logger.info(f"Fetched config.json from HuggingFace: {model_path}")
        except Exception as exc:
            logger.warning(f"Could not fetch config.json from {model_path}: {exc}")

    if model_config is None:
        logger.warning(
            f"Could not load config.json for {model_path}. "
            f"Using universal LoRA target_modules: {UNIVERSAL_FALLBACK}"
        )
        return UNIVERSAL_FALLBACK

    # Extract architecture identifier
    model_type = model_config.get("model_type", "").lower()
    architectures = [a.lower() for a in model_config.get("architectures", [])]

    # Try model_type first, then architecture class names
    candidates = [model_type] + architectures
    for cand in candidates:
        for arch_key, targets in ARCH_TO_TARGETS.items():
            if arch_key.lower() in cand:
                logger.info(
                    f"Detected architecture '{arch_key}' for '{model_type}' "
                    f"-> target_modules: {targets}"
                )
                return targets

    logger.warning(
        f"Unknown architecture: model_type='{model_type}', architectures={architectures}. "
        f"Using universal fallback: {UNIVERSAL_FALLBACK}. "
        "To get exact modules, inspect model.named_modules() after loading."
    )
    return UNIVERSAL_FALLBACK


def setup_lora_config(
    rank: int = 16,
    alpha: Optional[int] = None,
    target_modules: Optional[List[str]] = None,
):
    """Create LoRA configuration targeting attention + MLP projections.

    Args:
        rank: LoRA rank (r). Higher = more capacity, more VRAM.
        alpha: LoRA alpha scaling factor. If None, defaults to rank * 2
               (standard best practice for stable scaling).
        target_modules: Module names to inject LoRA into.
                        If None, uses default Qwen/Llama set.
                        Use detect_lora_target_modules() for automatic detection.
    """
    from peft import LoraConfig, TaskType

    # Auto-scale alpha: alpha = 2 * rank is the widely-accepted default
    # that keeps the LoRA scaling factor stable across different rank values.
    lora_alpha = alpha if alpha is not None else (rank * 2)

    # Default target_modules for Qwen2/Llama3 (works for all models in MODEL_REGISTRY)
    default_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    modules = target_modules if target_modules is not None else default_targets

    logger.info(
        f"LoRA config | rank={rank} | alpha={lora_alpha} | "
        f"scaling_factor={lora_alpha/rank:.2f} | targets={len(modules)} modules"
    )

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=modules,
        bias="none",
    )


def load_model_and_tokenizer(model_path: str, trust_remote_code: bool, use_4bit: bool = True):
    """
    Load base model with optional 4-bit quantization.

    Args:
        model_path: Local path or HuggingFace repo_id
        trust_remote_code: Forward to from_pretrained
        use_4bit: Enable BitsAndBytes NF4 quantization

    Returns:
        (model, tokenizer) tuple, model prepared for kbit training if 4-bit
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    logger.info(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # Set pad_token: Llama-3 uses special tokens, avoid eos_token collision
    if tokenizer.pad_token is None:
        model_path_lower = str(model_path).lower()
        if "llama-3" in model_path_lower or "llama3" in model_path_lower:
            # Llama-3 has <|finetune_right_pad_id|> or reserved special tokens.
            # Using eos_token causes early stopping issues during generation.
            # Try reserved token first, fall back to eos_token if unavailable.
            special_candidates = [
                "<|finetune_right_pad_id|>",
                "<|reserved_special_token_0|>",
            ]
            pad_set = False
            for candidate in special_candidates:
                if candidate in tokenizer.get_vocab():
                    tokenizer.pad_token = candidate
                    logger.info(f"Llama-3 pad_token set to '{candidate}'")
                    pad_set = True
                    break
            if not pad_set:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning(
                    "Llama-3: No dedicated pad token found in vocab. "
                    "Falling back to eos_token (may cause generation issues)."
                )
        else:
            tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    bnb_config: Optional[BitsAndBytesConfig] = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info(f"Loading model ({'4-bit' if use_4bit else 'fp16'}): {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False  # Required for gradient checkpointing
    return model, tokenizer


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """
    Scan output_dir for checkpoint-* subdirectories and return the path
    with the highest step number.

    Args:
        output_dir: Training output directory (e.g. models/slm/llama-3.2-1b-instruct-chart-lora)

    Returns:
        Absolute path string of the latest checkpoint, or None if not found.
    """
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        [
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
            and d.name.split("-")[-1].isdigit()
        ],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    logger.info(f"Auto-detected latest checkpoint: {latest}")
    return str(latest)


def train(
    model_key: str = DEFAULT_MODEL,
    data_path: Path = DATA_PATH,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: Optional[int] = None,
    gradient_accumulation_steps: int = 8,
    use_4bit: bool = True,
    eval_steps: int = 50,
    save_steps: int = 100,
    smoke_test: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    lr_scheduler_type: str = "cosine",
    run_manager: Optional[RunManager] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> None:
    """
    Run LoRA fine-tuning with trl 0.29 SFTTrainer + SFTConfig API.

    Supports incremental training: call with --epochs N --resume to continue
    from a previous session's checkpoint. HuggingFace Trainer automatically
    skips already-completed epochs and trains only the remaining ones.

    Args:
        model_key: Key from MODEL_REGISTRY
        data_path: Directory with train.json / val.json
        output_dir: Where to save adapter and training artifacts
        epochs: Total epochs target (not epochs *this run*). If resuming from
                epoch 1, set epochs=2 to train only epoch 2.
        batch_size: Per-device batch size
        learning_rate: Peak LR (cosine schedule)
        max_length: Maximum token sequence length (must fit full prompt + answer)
        lora_rank: LoRA rank (r parameter)
        lora_alpha: LoRA alpha scaling. None = auto (rank * 2)
        gradient_accumulation_steps: Steps to accumulate gradients (effective_batch = batch_size * this)
        use_4bit: NF4 quantization flag
        eval_steps: Run eval every N steps
        save_steps: Checkpoint every N steps
        smoke_test: If True, train for 2 steps only and exit
        resume_from_checkpoint: Path to checkpoint dir, or None for fresh start.
                                 Use find_latest_checkpoint() to auto-detect.
        run_manager: Optional RunManager for isolated run directories
        tracker: Optional ExperimentTracker for metrics logging
    """
    from trl import SFTConfig, SFTTrainer

    entry = MODEL_REGISTRY[model_key]

    # Determine output directory: RunManager > explicit > default
    if run_manager:
        output_dir = run_manager.checkpoints_dir
    elif output_dir is None:
        output_dir = MODELS_DIR / f"{entry['local_name']}-chart-lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve local vs remote model path
    model_path = resolve_model_path(model_key)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        trust_remote_code=entry["trust_remote_code"],
        use_4bit=use_4bit,
    )

    # Load datasets — must pass tokenizer for chat formatting
    chat_fmt = entry["chat_format"]
    train_dataset = load_dataset_split(data_path, "train", tokenizer, chat_fmt)
    val_dataset = load_dataset_split(data_path, "val", tokenizer, chat_fmt)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    if smoke_test:
        logger.info("SMOKE TEST: truncating to 4 examples, max_steps=2")
        train_dataset = train_dataset.select(range(min(4, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(2, len(val_dataset))))

    # LoRA config — auto-detect target_modules from model architecture
    # This ensures correct module names whether using Qwen, Llama, Phi, Gemma, etc.
    detected_targets = detect_lora_target_modules(model_path)
    logger.info(f"Auto-detected LoRA target_modules: {detected_targets}")
    lora_config = setup_lora_config(rank=lora_rank, alpha=lora_alpha, target_modules=detected_targets)

    # SFTConfig replaces TrainingArguments for trl 0.29+
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs if not smoke_test else 1,
        max_steps=2 if smoke_test else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=1 if smoke_test else 10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=tracker.get_report_to() if tracker else [],
        logging_dir=str(run_manager.logs_dir) if run_manager else None,
        fp16=False,
        bf16=True,  # Qwen2.5 / Llama-3.2 are natively BFloat16
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        # trl 0.29: max_length and dataset_text_field live here (not in SFTTrainer)
        max_length=max_length,
        dataset_text_field="text",
    )

    # SFTTrainer — trl 0.29 breaking changes:
    #   processing_class= replaces tokenizer=
    #   peft_config= replaces get_peft_model() call
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    logger.info("Trainable parameters:")
    trainer.model.print_trainable_parameters()

    # Log resume status clearly
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        logger.info(f"Target total epochs: {epochs} (trainer will skip completed epochs)")
    else:
        logger.info(f"Fresh training run. Target epochs: {epochs}")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if smoke_test:
        logger.info("Smoke test passed -- pipeline is functional")
        if run_manager:
            run_manager.finalize(metrics={"status": "smoke_test_passed"}, status="completed")
        if tracker:
            tracker.finish()
        return

    # Save final adapter
    final_dir = run_manager.final_model_dir if run_manager else (output_dir / "final")
    logger.info(f"Saving adapter to {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Build training info
    info_path = (run_manager.run_dir if run_manager else output_dir) / "training_info.json"
    existing_info: Dict[str, Any] = {}
    if info_path.exists():
        try:
            with open(info_path) as f:
                existing_info = json.load(f)
        except Exception:
            pass

    sessions = existing_info.get("sessions", [])
    sessions.append({
        "epochs_target": epochs,
        "resumed_from": resume_from_checkpoint,
        "completed_at": datetime.now().isoformat(),
    })

    # Extract final metrics from trainer state
    final_train_loss = None
    final_eval_loss = None
    best_eval_loss = None
    final_train_acc = None
    final_eval_acc = None
    if hasattr(trainer, "state") and trainer.state.log_history:
        log_history = trainer.state.log_history
        train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
        eval_logs = [l for l in log_history if "eval_loss" in l]
        if train_logs:
            final_train_loss = train_logs[-1].get("loss")
            final_train_acc = train_logs[-1].get("mean_token_accuracy")
        if eval_logs:
            final_eval_loss = eval_logs[-1].get("eval_loss")
            final_eval_acc = eval_logs[-1].get("eval_mean_token_accuracy")
        if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
            best_eval_loss = trainer.state.best_metric

    # Extract trainable parameters count
    trainable_params = None
    total_params = None
    try:
        tp, ap = 0, 0
        for p in trainer.model.parameters():
            ap += p.numel()
            if p.requires_grad:
                tp += p.numel()
        trainable_params = tp
        total_params = ap
    except Exception:
        pass

    # Extract peak VRAM
    vram_peak_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            vram_peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except Exception:
        pass

    info: Dict[str, Any] = {
        "model_key": model_key,
        "base_model": entry["repo_id"],
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha if lora_alpha is not None else (lora_rank * 2),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "total_epochs_target": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "max_length": max_length,
        "use_4bit": use_4bit,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "final_train_accuracy": final_train_acc,
        "final_eval_accuracy": final_eval_acc,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_params / total_params * 100, 2) if trainable_params and total_params else None,
        "vram_peak_mb": vram_peak_mb,
        "sessions": sessions,
        "last_updated": datetime.now().isoformat(),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Finalize run and tracker
    final_metrics = {
        "model_key": model_key,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": epochs,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
    }

    if tracker:
        tracker.log_summary(final_metrics)
        tracker.log_artifact(str(final_dir), name=f"{model_key}-lora-adapter", artifact_type="model")
        tracker.finish()

    if run_manager:
        run_manager.save_artifact(info, "training_info.json")
        run_manager.finalize(metrics=final_metrics, status="completed")

    logger.info(f"Training complete. Adapter saved to {final_dir}")
    logger.info(f"Sessions logged: {len(sessions)} total")
    if run_manager:
        logger.info(f"Run directory: {run_manager.run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Chart Analysis SLM with LoRA (trl 0.29 compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Run Management (NEW) ---
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file to load (e.g. config/training.yaml). "
             "When provided, creates an isolated run directory under runs/",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dynamic config overrides in OmegaConf dot-notation. "
             "Can be specified multiple times. "
             "Example: --override slm_training.training.learning_rate=1e-5",
    )
    parser.add_argument(
        "--tracker",
        choices=["wandb", "tensorboard", "json", "none"],
        default=None,
        help="Experiment tracking backend. Overrides config/training.yaml setting. "
             "Default: from config or 'json'",
    )

    # --- Model Selection ---
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to train. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_PATH,
        help="Directory with train.json / val.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: managed by RunManager or models/slm/<model>-chart-lora/)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--max-length", type=int, default=4096,
        help="Max token sequence length. Must fit full prompt + answer (default: 4096)",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--lora-alpha", type=int, default=None,
        help="LoRA alpha scaling factor. Default: auto (rank * 2)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=8,
        help="Gradient accumulation steps. Effective batch = batch_size * this (default: 8)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "constant_with_warmup",
                 "cosine_with_restarts", "polynomial", "inverse_sqrt"],
        help="Learning rate scheduler type (default: cosine)",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 training steps to verify pipeline without downloading large models",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Auto-detect and resume from the latest checkpoint in output-dir",
    )
    resume_group.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Resume from an explicit checkpoint directory path",
    )
    args = parser.parse_args()

    check_dependencies()

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA not available -- training will be very slow")

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Run: python scripts/training/prepare_slm_training_v3.py --output-dir data/slm_training_v3")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────
    # Initialize RunManager + ExperimentTracker (if --config provided)
    # ─────────────────────────────────────────────────────────────────────
    run_manager: Optional[RunManager] = None
    tracker: Optional[ExperimentTracker] = None

    if args.config:
        # Create isolated run with frozen config
        run_prefix = f"slm_lora_{args.model}"
        run_manager = RunManager(
            config_path=args.config,
            cli_overrides=args.override,
            run_prefix=run_prefix,
        )

        # Extract hyperparams from resolved config (CLI args override config values)
        cfg = run_manager.config
        slm_cfg = cfg.get("slm_training", {})
        training_cfg = slm_cfg.get("training", {}) if slm_cfg else {}
        lora_cfg = slm_cfg.get("lora", {}) if slm_cfg else {}
        data_cfg = slm_cfg.get("data", {}) if slm_cfg else {}
        run_mgmt_cfg = cfg.get("run_management", {})

        # Config values serve as defaults; CLI args override them
        # (argparse defaults are used only if neither config nor CLI provides a value)
        if training_cfg:
            if args.epochs == 3:  # argparse default
                args.epochs = training_cfg.get("num_train_epochs", args.epochs)
            if args.batch_size == 2:
                args.batch_size = training_cfg.get("per_device_train_batch_size", args.batch_size)
            if args.learning_rate == 2e-4:
                args.learning_rate = training_cfg.get("learning_rate", args.learning_rate)
            if args.max_length == 4096:
                args.max_length = training_cfg.get("max_seq_length", args.max_length)
            if args.gradient_accumulation_steps == 8:
                args.gradient_accumulation_steps = training_cfg.get(
                    "gradient_accumulation_steps", args.gradient_accumulation_steps
                )
            if args.eval_steps == 50:
                args.eval_steps = training_cfg.get("eval_steps", args.eval_steps)
            if args.save_steps == 100:
                args.save_steps = training_cfg.get("save_steps", args.save_steps)
        if lora_cfg:
            if args.lora_rank == 16:
                args.lora_rank = lora_cfg.get("rank", args.lora_rank)
            if args.lora_alpha is None:
                args.lora_alpha = lora_cfg.get("alpha", None)
        if data_cfg:
            if args.data_dir == DATA_PATH:
                data_train = data_cfg.get("train_file", "")
                if data_train:
                    args.data_dir = Path(data_train).parent

        # Initialize experiment tracker
        tracker_backend = args.tracker or run_mgmt_cfg.get("tracking_backend", "json")
        wandb_cfg = run_mgmt_cfg.get("wandb", {})

        tracker = ExperimentTracker(
            backend=tracker_backend,
            project=wandb_cfg.get("project", "chart_analysis_ai_v3") if wandb_cfg else "chart_analysis_ai_v3",
            run_name=run_manager.run_name,
            config={
                "model_key": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "max_length": args.max_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "use_4bit": not args.no_4bit,
            },
            log_dir=run_manager.logs_dir,
            tags=wandb_cfg.get("tags", []) if wandb_cfg else [],
        )

        logger.info(
            f"Run management active | run={run_manager.run_name} | "
            f"tracker={tracker.backend} | config_hash={run_manager.config_hash[:12]}"
        )

    # Resolve checkpoint path for resume
    checkpoint_path: Optional[str] = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
    elif args.resume:
        # Determine output_dir the same way train() does, to find checkpoints
        if run_manager:
            effective_output_dir = run_manager.checkpoints_dir
        else:
            effective_output_dir = args.output_dir
            if effective_output_dir is None:
                entry = MODEL_REGISTRY[args.model]
                effective_output_dir = MODELS_DIR / f"{entry['local_name']}-chart-lora"
        checkpoint_path = find_latest_checkpoint(effective_output_dir)
        if checkpoint_path is None:
            logger.error(
                f"--resume specified but no checkpoints found in {effective_output_dir}. "
                "Run without --resume for a fresh start."
            )
            sys.exit(1)

    try:
        train(
            model_key=args.model,
            data_path=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_4bit=not args.no_4bit,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            smoke_test=args.smoke_test,
            resume_from_checkpoint=checkpoint_path,
            lr_scheduler_type=args.lr_scheduler,
            run_manager=run_manager,
            tracker=tracker,
        )
    except Exception as exc:
        logger.error(f"Training failed | error={exc}")
        if run_manager:
            run_manager.finalize(metrics={"error": str(exc)}, status="failed")
        if tracker:
            tracker.finish()
        raise


if __name__ == "__main__":
    main()
