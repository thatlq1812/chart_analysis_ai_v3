#!/usr/bin/env python3
"""
Fine-tune Chart Analysis SLM with LoRA (trl 0.29 compatible).

Supports:
- Qwen2.5-0.5B / 1.5B-Instruct
- Llama-3.2-1B / 3B-Instruct

Usage:
    # Full training on default model (Qwen2.5-1.5B):
    python scripts/train_slm_lora.py

    # Choose model:
    python scripts/train_slm_lora.py --model qwen-1.5b
    python scripts/train_slm_lora.py --model llama-3b

    # Smoke test (2 steps, verifies pipeline end-to-end):
    python scripts/train_slm_lora.py --smoke-test
    python scripts/train_slm_lora.py --smoke-test --model qwen-0.5b

    # Custom hyperparameters:
    python scripts/train_slm_lora.py --epochs 5 --batch-size 2 --lora-rank 32

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes

Output:
    models/slm/<model_name>-chart-lora/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
# v2 dataset: 32k balanced samples (all 8 chart types, Gemini-2.0-flash annotations)
# v1 dataset:  9.7k line-heavy samples (legacy, to be removed after v2 training completes)
DATA_PATH = PROJECT_ROOT / "data" / "slm_training_v2"
DATA_PATH_LEGACY = PROJECT_ROOT / "data" / "slm_training"
DEFAULT_MODEL = "qwen-1.5b"

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
        f"Run: python scripts/download_models.py --model {model_key}"
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
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
):
    """Create LoRA configuration targeting attention + MLP projections.

    Args:
        rank: LoRA rank (r). Higher = more capacity, more VRAM.
        alpha: LoRA alpha scaling factor (typically 2x rank).
        target_modules: Module names to inject LoRA into.
                        If None, uses default Qwen/Llama set.
                        Use detect_lora_target_modules() for automatic detection.
    """
    from peft import LoraConfig, TaskType

    # Default target_modules for Qwen2/Llama3 (works for all models in MODEL_REGISTRY)
    default_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    modules = target_modules if target_modules is not None else default_targets

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
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
    if tokenizer.pad_token is None:
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


def train(
    model_key: str = DEFAULT_MODEL,
    data_path: Path = DATA_PATH,
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    lora_rank: int = 16,
    use_4bit: bool = True,
    eval_steps: int = 50,
    save_steps: int = 100,
    smoke_test: bool = False,
) -> None:
    """
    Run LoRA fine-tuning with trl 0.29 SFTTrainer + SFTConfig API.

    Args:
        model_key: Key from MODEL_REGISTRY
        data_path: Directory with train.json / val.json
        output_dir: Where to save adapter and training artifacts
        epochs: Training epochs
        batch_size: Per-device batch size
        learning_rate: Peak LR (cosine schedule)
        max_length: Maximum token sequence length
        lora_rank: LoRA rank (r parameter)
        use_4bit: NF4 quantization flag
        eval_steps: Run eval every N steps
        save_steps: Checkpoint every N steps
        smoke_test: If True, train for 2 steps only and exit
    """
    from trl import SFTConfig, SFTTrainer

    entry = MODEL_REGISTRY[model_key]
    if output_dir is None:
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
    lora_config = setup_lora_config(rank=lora_rank, target_modules=detected_targets)

    # SFTConfig replaces TrainingArguments for trl 0.29+
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs if not smoke_test else 1,
        max_steps=2 if smoke_test else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=1 if smoke_test else 10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=[],  # disable external logging (no wandb/tensorboard required)
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

    logger.info("Starting training...")
    trainer.train()

    if smoke_test:
        logger.info("Smoke test passed — pipeline is functional")
        return

    # Save final adapter
    final_dir = output_dir / "final"
    logger.info(f"Saving adapter to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Persist training metadata
    info = {
        "model_key": model_key,
        "base_model": entry["repo_id"],
        "lora_rank": lora_rank,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "use_4bit": use_4bit,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "completed_at": datetime.now().isoformat(),
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Training complete. Adapter saved to {final_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Chart Analysis SLM with LoRA (trl 0.29 compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help="Output directory (default: models/slm/<model>-chart-lora/)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 training steps to verify pipeline without downloading large models",
    )
    args = parser.parse_args()

    check_dependencies()

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu.name} | VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA not available — training will be very slow")

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Run: python scripts/prepare_slm_training_data.py")
        sys.exit(1)

    train(
        model_key=args.model,
        data_path=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        use_4bit=not args.no_4bit,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
