#!/usr/bin/env python3
"""
Download SLM models for Chart Analysis.

Downloads supported Small Language Models to models/slm/ directory.
Supports: Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B, Llama-3.2-3B.

Usage:
    # Interactive menu:
    python scripts/utils/download_models.py

    # Download specific model:
    python scripts/utils/download_models.py --model qwen-1.5b
    python scripts/utils/download_models.py --model qwen-0.5b
    python scripts/utils/download_models.py --model llama-1b
    python scripts/utils/download_models.py --model llama-3b

    # List available models:
    python scripts/utils/download_models.py --list

    # Download all:
    python scripts/utils/download_models.py --all

Requirements:
    pip install huggingface_hub transformers

Notes:
    - Qwen models are public (no token required).
    - Llama-3.2 models require:
        1. A HuggingFace account with HF_TOKEN set or `huggingface-cli login` run
        2. License accepted at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
        Requests are processed hourly by Meta.
    - Llama 4 (Scout 17B / Maverick 17B) is too large for 6GB VRAM training.
      Use Llama-3.2-1B or 3B for local SLM fine-tuning on RTX 3060/3070/4060.
    - VRAM requirement with 4-bit quantization:
        qwen-0.5b:   ~0.5 GB  (comfortable on 4 GB)
        qwen-1.5b:   ~1.5 GB  (comfortable on 6 GB)
        llama-1b:    ~0.8 GB  (comfortable on 4 GB)
        llama-3b:    ~2.5 GB  (comfortable on 6 GB)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Supported models catalog
# ─────────────────────────────────────────────────────────────────────────────
MODELS: Dict[str, Dict] = {
    "qwen-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "qwen2.5-0.5b-instruct",
        "vram_gb_int4": 0.5,
        "description": "Qwen2.5 0.5B - Fastest, minimal VRAM",
        "recommended_for": "Smoke-test, CPU inference",
        "trust_remote_code": True,
        "requires_token": False,
    },
    "qwen-1.5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "local_name": "qwen2.5-1.5b-instruct",
        "vram_gb_int4": 1.5,
        "description": "Qwen2.5 1.5B - Primary training target",
        "recommended_for": "Production SLM for chart QA",
        "trust_remote_code": True,
        "requires_token": False,
    },
    "llama-1b": {
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "local_name": "llama-3.2-1b-instruct",
        "vram_gb_int4": 0.8,
        "description": "Llama 3.2 1B - Meta's compact model",
        "recommended_for": "Comparison with Qwen, competitive accuracy",
        "trust_remote_code": False,
        "requires_token": True,
        "license_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
    },
    "llama-3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "local_name": "llama-3.2-3b-instruct",
        "vram_gb_int4": 2.5,
        "description": "Llama 3.2 3B - Higher capacity",
        "recommended_for": "Better accuracy on complex chart reasoning",
        "trust_remote_code": False,
        "requires_token": True,
        "license_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
    },
}

MODELS_DIR = PROJECT_ROOT / "models" / "slm"


# ─────────────────────────────────────────────────────────────────────────────
# Core download logic
# ─────────────────────────────────────────────────────────────────────────────

def check_hf_token() -> Optional[str]:
    """
    Check for HuggingFace token in environment or .env file.

    Returns:
        Token string if found, None otherwise.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Check HuggingFace cached login (set by huggingface_hub.login())
    hf_cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache_token.exists():
        stored = hf_cache_token.read_text().strip()
        if stored:
            return stored

    # Check .env file
    env_file = PROJECT_ROOT / "config" / "secrets" / ".env"
    if not env_file.exists():
        env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def download_model(model_key: str, force: bool = False) -> Path:
    """
    Download a model from HuggingFace Hub to models/slm/.

    Args:
        model_key: Key from MODELS dict (e.g. "qwen-1.5b")
        force: Re-download even if already present

    Returns:
        Path to the downloaded model directory

    Raises:
        KeyError: If model_key is not in MODELS
        RuntimeError: If download fails
    """
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Use --list to see available models.")

    info = MODELS[model_key]
    local_dir = MODELS_DIR / info["local_name"]

    # Check if already downloaded
    config_file = local_dir / "config.json"
    if config_file.exists() and not force:
        logger.info(f"Model already downloaded | key={model_key} | path={local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)

    # Check token requirements
    token = check_hf_token()

    logger.info(f"Downloading model | repo={info['repo_id']} | dest={local_dir}")
    logger.info(f"Description: {info['description']}")
    logger.info(f"VRAM (4-bit): ~{info['vram_gb_int4']} GB")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(local_dir),
            token=token,
            ignore_patterns=["*.safetensors.index.json", "original/**"],
        )

        logger.info(f"Download complete | path={local_dir}")
        return local_dir

    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "GatedRepo" in error_msg or "Access to model" in error_msg:
            license_url = info.get("license_url", f"https://huggingface.co/{info['repo_id']}")
            logger.error(
                f"License not accepted for {info['repo_id']}.\n"
                f"  1. Visit: {license_url}\n"
                "  2. Click 'Expand to review and access' and accept the Meta license.\n"
                "  3. Requests are processed hourly. Re-run this script after approval.\n"
                "  (Your HF token is valid — only the license acceptance is missing.)"
            )
        elif "401" in error_msg or "gated" in error_msg.lower() or "token" in error_msg.lower():
            logger.error(
                f"Authentication required for {info['repo_id']}.\n"
                "Set HF_TOKEN in config/secrets/.env or environment variable:\n"
                "  export HF_TOKEN=hf_your_token_here\n"
                "Or run: python -c \"from huggingface_hub import login; login()\"\n"
                "Get token at: https://huggingface.co/settings/tokens"
            )
        else:
            logger.error(f"Download failed | repo={info['repo_id']} | error={e}")
        raise RuntimeError(f"Download failed: {e}") from e


def verify_model(model_key: str) -> bool:
    """
    Verify a downloaded model can be loaded.

    Args:
        model_key: Key from MODELS dict

    Returns:
        True if model loads successfully
    """
    info = MODELS[model_key]
    local_dir = MODELS_DIR / info["local_name"]

    if not (local_dir / "config.json").exists():
        logger.warning(f"Model not downloaded | key={model_key} | run download first")
        return False

    logger.info(f"Verifying model | key={model_key}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(local_dir),
            trust_remote_code=info["trust_remote_code"],
        )
        tokens = tokenizer.encode("Hello, chart analysis!")
        logger.info(
            f"Tokenizer OK | vocab_size={tokenizer.vocab_size} | "
            f"sample_tokens={len(tokens)}"
        )
        return True

    except Exception as e:
        logger.error(f"Verification failed | key={model_key} | error={e}")
        return False


def list_models() -> None:
    """Print catalog of downloadable models with status."""
    print("\nAvailable models for SLM training:")
    print("=" * 72)
    print(f"{'Key':<14} {'Model':<30} {'VRAM 4-bit':<12} {'Downloaded'}")
    print("-" * 72)

    for key, info in MODELS.items():
        local_dir = MODELS_DIR / info["local_name"]
        downloaded = "YES" if (local_dir / "config.json").exists() else "no"
        vram_str = f"~{info['vram_gb_int4']} GB"
        model_short = info["repo_id"].split("/")[-1]
        print(f"{key:<14} {model_short:<30} {vram_str:<12} {downloaded}")

    print("=" * 72)
    print(f"\nDownload destination: {MODELS_DIR}")
    print("\nRecommended for 6 GB VRAM (RTX 3060):")
    print("  Training:  qwen-1.5b  (primary),  llama-3b  (comparison)")
    print("  Smoke-test: qwen-0.5b or llama-1b (fast, uses ~1 GB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SLM models for Geo-SLM Chart Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and download status",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded model(s) can be loaded",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already present",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.all:
        targets = list(MODELS.keys())
    elif args.model:
        targets = [args.model]
    else:
        # Interactive menu
        list_models()
        print("\nWhich model to download? (e.g. qwen-1.5b): ", end="")
        choice = input().strip()
        if choice not in MODELS:
            print(f"Unknown model '{choice}'. Exiting.")
            sys.exit(1)
        targets = [choice]

    # Download
    failed = []
    for key in targets:
        try:
            local_dir = download_model(key, force=args.force)
            if args.verify:
                verify_model(key)
        except RuntimeError:
            failed.append(key)

    if failed:
        logger.error(f"Failed downloads: {failed}")
        sys.exit(1)
    else:
        logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
