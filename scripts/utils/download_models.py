#!/usr/bin/env python3
"""
Model Downloader - Geo-SLM Chart Analysis

Downloads all models required for Stage 3 (VLM Extraction), Stage 4 (SLM
Reasoning/Training), and Stage 5 (Reporting) to the local models/ directory.

Uses huggingface_hub.snapshot_download() with resume support.
Models already on disk are detected automatically and skipped.

=============================================================================
MODEL INVENTORY
=============================================================================

GROUP: vlm  -- Stage 3 chart-to-table extraction (Pix2Struct family)
  google/deplot                     ~1.5 GB   Primary chart derendering
  google/matcha-base                ~1.1 GB   Math+chart reasoning base
  google/matcha-chartqa             ~1.1 GB   MatCha fine-tuned on ChartQA
  google/pix2struct-base            ~1.1 GB   Baseline (no chart fine-tuning)
  google/pix2struct-chartqa-large   ~3.1 GB   Pix2Struct fine-tuned on ChartQA
  Qwen/Qwen2-VL-2B-Instruct        ~4.5 GB   SVLM zero-shot backend

GROUP: slm  -- Stage 4 reasoning + fine-tuning base models
  Qwen/Qwen2.5-7B-Instruct         ~15 GB    PRIMARY QLoRA training target (v4)
  meta-llama/Llama-3.2-3B-Instruct  ~6 GB    Ablation comparison (3B class)

ALREADY ON DISK (auto-skipped):
  models/slm/llama-3.2-1b-instruct/     Llama-3.2-1B-Instruct
  models/slm/qwen2.5-0.5b-instruct/     Qwen2.5-0.5B-Instruct
  models/slm/qwen2.5-1.5b-instruct/     Qwen2.5-1.5B-Instruct
  models/paddleocr_vl/                   PaddleOCR-VL
  models/slm/vintern_finetuned/          Vintern-1B-v3.5

=============================================================================
AUTH REQUIREMENTS
=============================================================================
Llama models require accepting Meta's license:
  1. Visit https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
  2. Accept the license (processed hourly by Meta)
  3. Set HF_TOKEN env variable or run: huggingface-cli login

=============================================================================
USAGE
=============================================================================
  # Preview what would be downloaded (no actual download)
  .venv/Scripts/python.exe scripts/utils/download_models.py --dry-run

  # Download VLM models only (Stage 3) ~12 GB
  .venv/Scripts/python.exe scripts/utils/download_models.py --group vlm

  # Download SLM models only (Stage 4) ~21 GB
  .venv/Scripts/python.exe scripts/utils/download_models.py --group slm

  # Download everything
  .venv/Scripts/python.exe scripts/utils/download_models.py --group all

  # Skip models >7 GB (quick Stage 3 setup without 7B)
  .venv/Scripts/python.exe scripts/utils/download_models.py --group vlm --skip-large

  # Download a specific HuggingFace model by ID
  .venv/Scripts/python.exe scripts/utils/download_models.py --hf-id google/deplot

  # List all models with status
  .venv/Scripts/python.exe scripts/utils/download_models.py --list

LEGACY: old SLM-only shortcuts still work
  --model qwen-0.5b | qwen-1.5b | llama-1b | llama-3b | qwen-7b
"""

# NOTE: This file replaces the old SLM-only downloader (v1).
# v2 adds: VLM group (Stage 3), Qwen2.5-7B, Qwen2-VL-2B, dry-run, skip-large.

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelSpec:
    """Specification for a single model download."""

    hf_id: str
    local_path: str               # Relative to PROJECT_ROOT
    group: str                    # "vlm" | "slm"
    size_gb: float                # Disk space estimate
    description: str
    short_key: str = ""           # Legacy short key (e.g. "qwen-1.5b")
    requires_auth: bool = False   # True for gated models (Llama)
    large: bool = False           # True if >7 GB
    priority: int = 1             # Lower = higher priority (downloaded first)
    ignore_patterns: List[str] = field(default_factory=list)


MODEL_REGISTRY: List[ModelSpec] = [
    # -------------------------------------------------------------------------
    # GROUP: vlm -- Stage 3 chart-to-table VLM extraction
    # -------------------------------------------------------------------------
    ModelSpec(
        hf_id="google/deplot",
        local_path="models/vlm/deplot",
        group="vlm",
        size_gb=1.5,
        description="DePlot: chart derendering -> linearized table (primary Stage 3 backend)",
        priority=1,
    ),
    ModelSpec(
        hf_id="google/matcha-base",
        local_path="models/vlm/matcha-base",
        group="vlm",
        size_gb=1.1,
        description="MatCha-base: math+chart reasoning pre-trained",
        priority=2,
    ),
    ModelSpec(
        hf_id="google/matcha-chartqa",
        local_path="models/vlm/matcha-chartqa",
        group="vlm",
        size_gb=1.1,
        description="MatCha fine-tuned on ChartQA benchmark",
        priority=3,
    ),
    ModelSpec(
        hf_id="google/pix2struct-base",
        local_path="models/vlm/pix2struct-base",
        group="vlm",
        size_gb=1.1,
        description="Pix2Struct-base: screenshot understanding (ablation baseline)",
        priority=4,
    ),
    ModelSpec(
        hf_id="google/pix2struct-large",
        local_path="models/vlm/pix2struct-large",
        group="vlm",
        size_gb=3.1,
        description="Pix2Struct-large: large base model (stronger than pix2struct-base)",
        priority=5,
        ignore_patterns=["pytorch_model.bin"],
    ),
    ModelSpec(
        hf_id="Qwen/Qwen2-VL-2B-Instruct",
        local_path="models/vlm/qwen2-vl-2b",
        group="vlm",
        size_gb=4.5,
        description="Qwen2-VL-2B: zero-shot SVLM backend for Stage 3",
        priority=6,
    ),
    # -------------------------------------------------------------------------
    # GROUP: slm -- Stage 4 reasoning + QLoRA fine-tuning candidates
    # -------------------------------------------------------------------------
    ModelSpec(
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_path="models/slm/qwen2.5-0.5b-instruct",
        group="slm",
        size_gb=1.0,
        description="Qwen2.5-0.5B: smoke-test / CPU inference (already on disk)",
        short_key="qwen-0.5b",
        priority=1,
    ),
    ModelSpec(
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_path="models/slm/qwen2.5-1.5b-instruct",
        group="slm",
        size_gb=3.1,
        description="Qwen2.5-1.5B: previous training target (already on disk)",
        short_key="qwen-1.5b",
        priority=2,
    ),
    ModelSpec(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        local_path="models/slm/qwen2.5-7b-instruct",
        group="slm",
        size_gb=15.0,
        description="Qwen2.5-7B: PRIMARY QLoRA training target for dataset v4",
        short_key="qwen-7b",
        large=True,
        priority=3,
    ),
    ModelSpec(
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        local_path="models/slm/llama-3.2-1b-instruct",
        group="slm",
        size_gb=2.5,
        description="Llama-3.2-1B: ablation 1B model (already on disk)",
        short_key="llama-1b",
        requires_auth=True,
        priority=4,
    ),
    ModelSpec(
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        local_path="models/slm/llama-3.2-3b-instruct",
        group="slm",
        size_gb=6.0,
        description="Llama-3.2-3B: 3B ablation comparison (requires HF token)",
        short_key="llama-3b",
        requires_auth=True,
        priority=5,
    ),
]

# Build lookup maps
_BY_HF_ID: Dict[str, ModelSpec] = {m.hf_id: m for m in MODEL_REGISTRY}
_BY_SHORT_KEY: Dict[str, ModelSpec] = {
    m.short_key: m for m in MODEL_REGISTRY if m.short_key
}


# =============================================================================
# Download helpers
# =============================================================================

_SENTINEL_FILES = ["config.json", "model.safetensors", "pytorch_model.bin"]


def _is_downloaded(local_path: Path) -> bool:
    """Return True if model directory has sentinel weight/config files."""
    if not local_path.exists():
        return False
    for name in _SENTINEL_FILES:
        if (local_path / name).exists():
            return True
    return len(list(local_path.glob("model-*-of-*.safetensors"))) > 0


def _get_token() -> Optional[str]:
    """Return HF token from env or cached huggingface-cli login."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    hf_cache = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache.exists():
        stored = hf_cache.read_text().strip()
        if stored:
            return stored
    env_file = PROJECT_ROOT / "config" / "secrets" / ".env"
    if not env_file.exists():
        env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def download_model(spec: ModelSpec, dry_run: bool = False, force: bool = False) -> bool:
    """
    Download a single model via huggingface_hub.snapshot_download().

    Args:
        spec: ModelSpec to download
        dry_run: Print action without downloading
        force: Re-download even if model already exists on disk

    Returns:
        True on success / already-present, False on error
    """
    local_path = PROJECT_ROOT / spec.local_path
    tag = "[DRY-RUN] " if dry_run else ""

    if not force and _is_downloaded(local_path):
        logger.info(f"{tag}SKIP (already on disk) | {spec.hf_id}")
        return True

    auth_note = " [requires HF_TOKEN]" if spec.requires_auth else ""
    logger.info(f"{tag}DOWNLOAD | {spec.hf_id} (~{spec.size_gb:.1f} GB){auth_note}")
    logger.info(f"          -> {local_path}")

    if dry_run:
        return True

    token = _get_token()
    if spec.requires_auth and not token:
        logger.error(
            f"MISSING HF_TOKEN for gated model {spec.hf_id}.\n"
            "  1. Accept the license at huggingface.co\n"
            "  2. export HF_TOKEN=hf_... and retry\n"
            "  Skipping."
        )
        return False

    try:
        from huggingface_hub import snapshot_download

        local_path.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        snapshot_download(
            repo_id=spec.hf_id,
            local_dir=str(local_path),
            ignore_patterns=spec.ignore_patterns + [
                "*.msgpack",
                "flax_model*",
                "tf_model*",
                "rust_model.ot",
                # Skip legacy pytorch checkpoints when safetensors are available
                # NOTE: pix2struct-base and matcha only have pytorch_model.bin,
                # so we don't add it to the global ignore. Use per-model ignore_patterns.
            ],
            token=token,
            resume_download=True,
        )

        elapsed = time.time() - t0
        logger.info(f"DONE | {spec.hf_id} | {elapsed/60:.1f} min | {local_path}")
        return True

    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "GatedRepo" in msg or "Access to model" in msg:
            logger.error(
                f"License not accepted for {spec.hf_id}.\n"
                "  Visit the model page, accept the license, wait ~1 hour, then retry."
            )
        elif "401" in msg or "token" in msg.lower():
            logger.error(
                f"Auth error for {spec.hf_id}. Set HF_TOKEN or run: huggingface-cli login"
            )
        else:
            logger.error(f"FAILED | {spec.hf_id} | {exc}")
        return False


# =============================================================================
# Table display
# =============================================================================


def print_model_table() -> None:
    print(f"\n{'='*78}")
    print(f"  Model Registry - Geo-SLM Chart Analysis (Stage 3-5)")
    print(f"{'='*78}")
    print(f"  {'HF Model ID':<42} {'GRP':<4} {'SIZE':>6}  {'KEY':<12}  STATUS")
    print(f"  {'-'*42} {'-'*4} {'-'*6}  {'-'*12}  {'-'*14}")
    for spec in MODEL_REGISTRY:
        status = "ON DISK" if _is_downloaded(PROJECT_ROOT / spec.local_path) else "not downloaded"
        auth = "*" if spec.requires_auth else " "
        key = spec.short_key or "-"
        print(
            f"  {spec.hf_id:<42} {spec.group:<4} {spec.size_gb:>5.1f}GB{auth} {key:<12}  {status}"
        )
    print(f"\n  * = requires HF_TOKEN (gated Llama models)")
    print(f"  Models root: {PROJECT_ROOT / 'models'}")
    print(f"{'='*78}\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Stage 3-5 models for Geo-SLM Chart Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group",
        choices=["vlm", "slm", "all"],
        default=None,
        help="Model group: vlm=Stage3 extractors, slm=Stage4 SLMs, all=both",
    )
    parser.add_argument(
        "--hf-id",
        default=None,
        help="Download a single model by HuggingFace repo ID (e.g. google/deplot)",
    )
    # Legacy short-key compat
    parser.add_argument(
        "--model",
        choices=list(_BY_SHORT_KEY.keys()),
        default=None,
        help="[Legacy] Short model key: qwen-0.5b | qwen-1.5b | qwen-7b | llama-1b | llama-3b",
    )
    # Legacy --all compat
    parser.add_argument("--all", action="store_true", help="[Legacy] Download all SLM models")
    # Legacy --list compat
    parser.add_argument("--list", action="store_true", help="List models with status and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download if already on disk")
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip models >7 GB (e.g. Qwen2.5-7B) for quick setup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print_model_table()
        return

    # -- Resolve model list --
    specs: List[ModelSpec] = []

    if args.hf_id:
        if args.hf_id not in _BY_HF_ID:
            logger.error(f"Unknown HF ID '{args.hf_id}'. Use --list to see registered models.")
            sys.exit(1)
        specs = [_BY_HF_ID[args.hf_id]]
    elif args.model:
        specs = [_BY_SHORT_KEY[args.model]]
    elif args.all:
        # Legacy --all: download all SLM models (backward compat)
        specs = [m for m in MODEL_REGISTRY if m.group == "slm"]
    elif args.group:
        specs = MODEL_REGISTRY[:] if args.group == "all" else [m for m in MODEL_REGISTRY if m.group == args.group]
    else:
        # Default: show table and prompt
        print_model_table()
        print("No action requested. Use --group vlm|slm|all or --hf-id. See --help.")
        sys.exit(0)

    if args.skip_large:
        skipped = [m.hf_id for m in specs if m.large]
        specs = [m for m in specs if not m.large]
        if skipped:
            logger.info(f"--skip-large: skipping {skipped}")

    if not specs:
        logger.info("No models to download.")
        return

    specs.sort(key=lambda m: m.priority)

    # -- Summary --
    to_dl = [m for m in specs if not _is_downloaded(PROJECT_ROOT / m.local_path)]
    total_gb = sum(m.size_gb for m in to_dl)
    print(f"\n{'='*60}")
    print(f"  Download Plan")
    print(f"{'='*60}")
    print(f"  Total in scope   : {len(specs)}")
    print(f"  Already on disk  : {len(specs) - len(to_dl)}")
    print(f"  To download      : {len(to_dl)}")
    print(f"  Estimated size   : {total_gb:.1f} GB")
    print(f"  Dry run          : {args.dry_run}")
    print(f"{'='*60}\n")

    # -- Run --
    failed: List[str] = []
    for spec in specs:
        ok = download_model(spec, dry_run=args.dry_run, force=args.force)
        if not ok:
            failed.append(spec.hf_id)

    print(f"\n{'='*60}")
    if args.dry_run:
        print("  Dry run complete. Remove --dry-run to start downloads.")
    elif failed:
        logger.error(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("  All downloads complete.")
        print(f"  Models: {PROJECT_ROOT / 'models'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
