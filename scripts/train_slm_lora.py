#!/usr/bin/env python3
"""
Fine-tune Qwen 2.5 1.5B with LoRA for Chart Analysis.

This script implements:
- LoRA fine-tuning with PEFT
- Gradient checkpointing for memory efficiency
- ChatML format training
- Evaluation during training

Usage:
    python scripts/train_slm_lora.py
    python scripts/train_slm_lora.py --epochs 3 --batch-size 4

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes

Output:
    models/slm/qwen2.5-1.5b-chart-lora/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model constants
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = PROJECT_ROOT / "models" / "slm" / "qwen2.5-1.5b-chart-lora"


def check_dependencies():
    """Check if required packages are installed."""
    required = ["transformers", "peft", "trl", "datasets", "accelerate"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        sys.exit(1)
    
    logger.info("All dependencies installed")


def load_dataset(data_path: Path, split: str = "train"):
    """Load training data from JSON file."""
    from datasets import Dataset
    
    file_path = data_path / f"{split}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert ChatML format to text
    texts = []
    for item in data:
        # Format as Qwen conversation
        messages = item["conversations"]
        text = format_qwen_messages(messages)
        texts.append({"text": text, **item.get("metadata", {})})
    
    return Dataset.from_list(texts)


def format_qwen_messages(messages: list) -> str:
    """Format messages into Qwen's ChatML format."""
    # Qwen uses special tokens for chat
    formatted = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return formatted


def setup_lora_config(rank: int = 16, alpha: int = 32):
    """Create LoRA configuration."""
    from peft import LoraConfig, TaskType
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )


def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load base model with quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    
    logger.info(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantization config for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False  # Required for gradient checkpointing
    
    return model, tokenizer


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    lora_rank: int = 16,
    use_4bit: bool = True,
    eval_steps: int = 100,
    save_steps: int = 200,
):
    """Run LoRA fine-tuning."""
    from transformers import TrainingArguments
    from peft import get_peft_model
    from trl import SFTTrainer
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(data_path, "train")
    val_dataset = load_dataset(data_path, "val")
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, use_4bit=use_4bit)
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    lora_config = setup_lora_config(rank=lora_rank)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        fp16=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )
    
    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_length,
        args=training_args,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    # Save training info
    info = {
        "base_model": BASE_MODEL,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "completed_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Qwen 2.5 with LoRA for chart QA")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/slm_training"),
        help="Directory with train.json, val.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--eval-steps", type=int, default=100, help="Eval every N steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save every N steps")
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA not available, training will be slow!")
    
    # Check data exists
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Run prepare_slm_data.py first!")
        sys.exit(1)
    
    # Run training
    train(
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
    )


if __name__ == "__main__":
    main()
