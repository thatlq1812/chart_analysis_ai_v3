#!/usr/bin/env python3
"""
SLM Evaluation Benchmark

Evaluates fine-tuned SLM models on the chart analysis test set.
Produces structured JSON results compatible with thesis comparison tables.

Metrics:
    - Exact Match (EM): Normalized string match
    - Contains Match: Reference answer found in prediction
    - Numeric Accuracy: Numeric extraction within tolerance
    - BLEU-1: Unigram overlap for longer answers
    - Latency: Per-sample inference time
    - VRAM Peak: GPU memory high-water mark

Supports:
    - Local LoRA models (Qwen, Llama + adapter)
    - Local merged models
    - Base models (zero-shot)
    - Gemini API (cloud comparison)

Usage:
    # Evaluate Llama-3.2-1B fine-tuned LoRA:
    python scripts/evaluation/evaluate_slm.py \
        --base-model models/slm/llama-3.2-1b-instruct \
        --lora-path models/slm/llama-3.2-1b-chart-lora-v3/final \
        --test-data data/slm_training_v3/test.json \
        --output models/evaluation/llama-1b-lora-v3.json \
        --max-samples 500

    # Evaluate base model (zero-shot):
    python scripts/evaluation/evaluate_slm.py \
        --base-model models/slm/llama-3.2-1b-instruct \
        --test-data data/slm_training_v3/test.json \
        --output models/evaluation/llama-1b-base.json \
        --max-samples 500

    # Evaluate Qwen fine-tuned:
    python scripts/evaluation/evaluate_slm.py \
        --base-model models/slm/qwen2.5-1.5b-instruct \
        --lora-path models/slm/qwen2.5-1.5b-instruct-chart-lora/final \
        --test-data data/slm_training_v3/test.json \
        --output models/evaluation/qwen-1.5b-lora.json

    # Quick smoke test (10 samples):
    python scripts/evaluation/evaluate_slm.py \
        --base-model models/slm/llama-3.2-1b-instruct \
        --lora-path models/slm/llama-3.2-1b-chart-lora-v3/final \
        --test-data data/slm_training_v3/test.json \
        --output models/evaluation/smoke.json \
        --max-samples 10

    # Stratified sample across question types and chart types:
    python scripts/evaluation/evaluate_slm.py \
        --base-model models/slm/llama-3.2-1b-instruct \
        --lora-path models/slm/llama-3.2-1b-chart-lora-v3/final \
        --test-data data/slm_training_v3/test.json \
        --output models/evaluation/llama-1b-stratified.json \
        --max-samples 500 --stratified
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation for short answers
    text = text.rstrip(".")
    return text


def exact_match(prediction: str, reference: str) -> bool:
    """Normalized exact string match."""
    return normalize_text(prediction) == normalize_text(reference)


def contains_match(prediction: str, reference: str) -> bool:
    """Check if the reference answer is contained in the prediction."""
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    if not ref_norm:
        return False
    return ref_norm in pred_norm


def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text."""
    # Match integers, decimals, negative numbers, scientific notation
    pattern = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
    matches = re.findall(pattern, text.lower())
    results = []
    for m in matches:
        try:
            results.append(float(m))
        except ValueError:
            pass
    return results


def numeric_accuracy(prediction: str, reference: str, tolerance: float = 0.05) -> float:
    """
    Compare numeric values between prediction and reference.

    Returns fraction of reference numbers matched within tolerance.
    If no numbers in reference, returns -1.0 (not applicable).
    """
    ref_nums = extract_numbers(reference)
    if not ref_nums:
        return -1.0  # Not applicable

    pred_nums = extract_numbers(prediction)
    if not pred_nums:
        return 0.0

    matched = 0
    for ref_val in ref_nums:
        for pred_val in pred_nums:
            if abs(ref_val) < 1e-9:
                if abs(pred_val) < 1e-9:
                    matched += 1
                    break
            elif abs(pred_val - ref_val) / (abs(ref_val) + 1e-9) <= tolerance:
                matched += 1
                break

    return matched / len(ref_nums)


def bleu_1(prediction: str, reference: str) -> float:
    """
    Compute unigram BLEU score (BLEU-1).

    Simple token-overlap metric for longer text answers.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)

    clipped = 0
    for token, count in pred_counts.items():
        clipped += min(count, ref_counts.get(token, 0))

    precision = clipped / len(pred_tokens) if pred_tokens else 0.0

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))

    return bp * precision


def answer_length_category(text: str) -> str:
    """Classify answer length for stratified analysis."""
    length = len(text.strip())
    if length <= 5:
        return "short"
    elif length <= 50:
        return "medium"
    else:
        return "long"


# =============================================================================
# Model Loading
# =============================================================================


def load_local_model(
    base_model_path: str,
    lora_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device: str = "auto",
) -> Tuple[Any, Any, Any]:
    """
    Load a local HuggingFace model with optional LoRA adapter.

    Args:
        base_model_path: Path to base model or HuggingFace model ID
        lora_path: Optional path to LoRA adapter directory
        load_in_4bit: Use 4-bit quantization
        device: Device map strategy

    Returns:
        Tuple of (pipeline, tokenizer, model_name)
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )

    model_name = Path(base_model_path).name
    if lora_path:
        model_name += f"+LoRA({Path(lora_path).parent.name})"

    logger.info(f"Loading model | base={base_model_path} | lora={lora_path}")

    # Load tokenizer from LoRA path (has chat template) or base
    tokenizer_path = lora_path if lora_path else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Quantization config
    quant_config = None
    if load_in_4bit:
        import torch as _torch

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True,
    )

    # Apply LoRA if provided
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
        logger.info(f"LoRA adapter loaded | path={lora_path}")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Clear default max_length to avoid conflict with max_new_tokens
    if hasattr(model, "generation_config") and model.generation_config.max_length:
        model.generation_config.max_length = None

    # Log VRAM usage
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"VRAM allocated after load: {vram_mb:.0f} MB")

    return pipe, tokenizer, model_name


# =============================================================================
# Inference
# =============================================================================


def run_inference(
    pipe: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> Tuple[str, float]:
    """
    Run single inference and return (answer, latency_seconds).

    Args:
        pipe: HuggingFace text-generation pipeline
        tokenizer: Tokenizer instance
        system_prompt: System prompt
        user_prompt: User prompt with chart context
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (generated_text, latency_in_seconds)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    t1 = time.perf_counter()

    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        answer = generated[-1].get("content", "")
    else:
        answer = str(generated)

    return answer.strip(), t1 - t0


# =============================================================================
# Data Loading & Sampling
# =============================================================================


def load_test_data(
    test_path: str,
    max_samples: int = 0,
    stratified: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load test data with optional stratified sampling.

    Args:
        test_path: Path to test.json
        max_samples: Max samples (0 = all)
        stratified: If True, sample proportionally by question_type + chart_type
        seed: Random seed for reproducibility

    Returns:
        List of test samples
    """
    import random

    logger.info(f"Loading test data | path={test_path}")
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Total test samples: {len(data)}")

    if max_samples <= 0 or max_samples >= len(data):
        return data

    if stratified:
        return _stratified_sample(data, max_samples, seed)
    else:
        random.seed(seed)
        return random.sample(data, max_samples)


def _stratified_sample(
    data: List[Dict[str, Any]],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Stratified sampling by (question_type, chart_type) combination.

    Ensures proportional representation across all categories.
    """
    import random

    random.seed(seed)

    # Group by (question_type, chart_type)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in data:
        meta = sample.get("metadata", {})
        key = f"{meta.get('question_type', 'unknown')}_{meta.get('chart_type', 'unknown')}"
        groups[key].append(sample)

    # Calculate proportional allocation
    total = len(data)
    sampled: List[Dict[str, Any]] = []

    for key, group in groups.items():
        proportion = len(group) / total
        n = max(1, round(proportion * max_samples))
        n = min(n, len(group))
        sampled.extend(random.sample(group, n))

    # Trim if oversampled
    if len(sampled) > max_samples:
        random.shuffle(sampled)
        sampled = sampled[:max_samples]

    logger.info(
        f"Stratified sampling | groups={len(groups)} | "
        f"selected={len(sampled)} / {max_samples}"
    )
    return sampled


# =============================================================================
# Evaluation Loop
# =============================================================================


def evaluate(
    pipe: Any,
    tokenizer: Any,
    test_data: List[Dict[str, Any]],
    model_name: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Run evaluation on the test set.

    Args:
        pipe: Model pipeline
        tokenizer: Tokenizer
        test_data: List of test samples
        model_name: Model identifier string
        max_new_tokens: Max generation tokens
        temperature: Sampling temperature

    Returns:
        Dict with aggregate metrics, per-type breakdowns, and sample-level results
    """
    import torch

    results: List[Dict[str, Any]] = []
    total = len(test_data)

    # Track metrics
    em_correct = 0
    contains_correct = 0
    numeric_scores: List[float] = []
    bleu_scores: List[float] = []
    latencies: List[float] = []

    # Per-question-type metrics
    qt_metrics: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"em": 0, "contains": 0, "numeric": [], "bleu": [], "count": 0}
    )
    # Per-chart-type metrics
    ct_metrics: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"em": 0, "contains": 0, "numeric": [], "bleu": [], "count": 0}
    )

    logger.info(f"Evaluating {total} samples...")
    start_all = time.time()

    for i, sample in enumerate(test_data):
        convs = sample["conversations"]
        meta = sample.get("metadata", {})
        qt = meta.get("question_type", "unknown")
        ct = meta.get("chart_type", "unknown")

        system_prompt = convs[0]["content"]
        user_prompt = convs[1]["content"]
        reference = convs[2]["content"]

        # Run inference
        try:
            prediction, latency = run_inference(
                pipe,
                tokenizer,
                system_prompt,
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning(f"Inference failed | sample={i} | error={exc}")
            prediction = ""
            latency = 0.0

        # Compute metrics
        is_em = exact_match(prediction, reference)
        is_contains = contains_match(prediction, reference)
        num_acc = numeric_accuracy(prediction, reference)
        b1 = bleu_1(prediction, reference)
        ans_cat = answer_length_category(reference)

        if is_em:
            em_correct += 1
        if is_contains:
            contains_correct += 1
        if num_acc >= 0:
            numeric_scores.append(num_acc)
        bleu_scores.append(b1)
        latencies.append(latency)

        # Per-type tracking
        for bucket, key in [(qt_metrics, qt), (ct_metrics, ct)]:
            bucket[key]["count"] += 1
            if is_em:
                bucket[key]["em"] += 1
            if is_contains:
                bucket[key]["contains"] += 1
            if num_acc >= 0:
                bucket[key]["numeric"].append(num_acc)
            bucket[key]["bleu"].append(b1)

        # Store sample result
        results.append(
            {
                "index": i,
                "chart_type": ct,
                "question_type": qt,
                "curriculum_stage": meta.get("curriculum_stage", 0),
                "reference": reference,
                "prediction": prediction,
                "exact_match": is_em,
                "contains_match": is_contains,
                "numeric_accuracy": round(num_acc, 4) if num_acc >= 0 else None,
                "bleu_1": round(b1, 4),
                "answer_category": ans_cat,
                "latency_s": round(latency, 4),
            }
        )

        # Progress logging
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_all
            eta = (elapsed / (i + 1)) * (total - i - 1)
            logger.info(
                f"Progress: {i+1}/{total} | "
                f"EM={em_correct}/{i+1} ({em_correct/(i+1)*100:.1f}%) | "
                f"Contains={contains_correct}/{i+1} ({contains_correct/(i+1)*100:.1f}%) | "
                f"ETA={eta:.0f}s"
            )

    total_time = time.time() - start_all

    # Aggregate metrics
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    aggregate = {
        "exact_match": round(em_correct / total, 4) if total else 0,
        "contains_match": round(contains_correct / total, 4) if total else 0,
        "numeric_accuracy": (
            round(sum(numeric_scores) / len(numeric_scores), 4)
            if numeric_scores
            else None
        ),
        "bleu_1": round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0,
        "mean_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else 0,
        "p50_latency_s": round(sorted(latencies)[len(latencies) // 2], 4) if latencies else 0,
        "p95_latency_s": (
            round(sorted(latencies)[int(len(latencies) * 0.95)], 4) if latencies else 0
        ),
        "total_samples": total,
        "total_time_s": round(total_time, 2),
        "vram_peak_mb": round(vram_mb, 0),
    }

    # Per-type aggregation
    def _agg_bucket(bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for key, vals in bucket.items():
            n = vals["count"]
            out[key] = {
                "count": n,
                "exact_match": round(vals["em"] / n, 4) if n else 0,
                "contains_match": round(vals["contains"] / n, 4) if n else 0,
                "numeric_accuracy": (
                    round(sum(vals["numeric"]) / len(vals["numeric"]), 4)
                    if vals["numeric"]
                    else None
                ),
                "bleu_1": (
                    round(sum(vals["bleu"]) / len(vals["bleu"]), 4)
                    if vals["bleu"]
                    else 0
                ),
            }
        return dict(sorted(out.items(), key=lambda x: -x[1]["count"]))

    return {
        "model_name": model_name,
        "evaluated_at": datetime.now().isoformat(),
        "test_data_path": str(test_data[0].get("_source_path", "unknown")) if test_data else "",
        "config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "aggregate": aggregate,
        "by_question_type": _agg_bucket(qt_metrics),
        "by_chart_type": _agg_bucket(ct_metrics),
        "samples": results,
    }


# =============================================================================
# Report formatting
# =============================================================================


def print_report(eval_result: Dict[str, Any]) -> None:
    """Print a human-readable evaluation report to stdout."""
    agg = eval_result["aggregate"]
    model_name = eval_result["model_name"]

    print()
    print("=" * 70)
    print(f"  EVALUATION REPORT: {model_name}")
    print("=" * 70)
    print()

    print("--- Aggregate Metrics ---")
    print(f"  Exact Match Rate:    {agg['exact_match']*100:.2f}%")
    print(f"  Contains Match Rate: {agg['contains_match']*100:.2f}%")
    if agg["numeric_accuracy"] is not None:
        print(f"  Numeric Accuracy:    {agg['numeric_accuracy']*100:.2f}%")
    else:
        print(f"  Numeric Accuracy:    N/A (no numeric questions)")
    print(f"  BLEU-1:              {agg['bleu_1']:.4f}")
    print()
    print(f"  Mean Latency:        {agg['mean_latency_s']:.3f}s")
    print(f"  P50 Latency:         {agg['p50_latency_s']:.3f}s")
    print(f"  P95 Latency:         {agg['p95_latency_s']:.3f}s")
    print(f"  VRAM Peak:           {agg['vram_peak_mb']:.0f} MB")
    print(f"  Total Time:          {agg['total_time_s']:.1f}s ({agg['total_samples']} samples)")
    print()

    # By question type
    print("--- By Question Type ---")
    print(f"  {'Type':<20} {'Count':>6} {'EM%':>8} {'Contains%':>10} {'BLEU-1':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for qt, vals in eval_result.get("by_question_type", {}).items():
        print(
            f"  {qt:<20} {vals['count']:>6} "
            f"{vals['exact_match']*100:>7.1f}% "
            f"{vals['contains_match']*100:>9.1f}% "
            f"{vals['bleu_1']:>8.4f}"
        )
    print()

    # By chart type
    print("--- By Chart Type ---")
    print(f"  {'Type':<12} {'Count':>6} {'EM%':>8} {'Contains%':>10} {'BLEU-1':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for ct, vals in eval_result.get("by_chart_type", {}).items():
        print(
            f"  {ct:<12} {vals['count']:>6} "
            f"{vals['exact_match']*100:>7.1f}% "
            f"{vals['contains_match']*100:>9.1f}% "
            f"{vals['bleu_1']:>8.4f}"
        )
    print()

    # Show some error samples
    samples = eval_result.get("samples", [])
    errors = [s for s in samples if not s["contains_match"]]
    if errors:
        print("--- Sample Errors (first 5) ---")
        for s in errors[:5]:
            print(f"  [{s['question_type']}|{s['chart_type']}]")
            print(f"    Reference:  {s['reference'][:100]}")
            print(f"    Prediction: {s['prediction'][:100]}")
            print()

    print("=" * 70)


# =============================================================================
# Comparison table generator
# =============================================================================


def generate_comparison_table(result_paths: List[str], output_path: Optional[str] = None) -> str:
    """
    Generate a markdown comparison table from multiple evaluation results.

    Args:
        result_paths: Paths to evaluation JSON files
        output_path: Optional path to write the markdown table

    Returns:
        Markdown formatted comparison table
    """
    results = []
    for path in result_paths:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    lines = [
        "# SLM Model Comparison",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Metric | " + " | ".join(r["model_name"] for r in results) + " |",
        "| --- | " + " | ".join("---" for _ in results) + " |",
    ]

    metric_rows = [
        ("Exact Match", "exact_match", True),
        ("Contains Match", "contains_match", True),
        ("Numeric Accuracy", "numeric_accuracy", True),
        ("BLEU-1", "bleu_1", False),
        ("Mean Latency (s)", "mean_latency_s", False),
        ("P95 Latency (s)", "p95_latency_s", False),
        ("VRAM Peak (MB)", "vram_peak_mb", False),
        ("Samples", "total_samples", False),
    ]

    for label, key, is_pct in metric_rows:
        vals = []
        for r in results:
            v = r["aggregate"].get(key)
            if v is None:
                vals.append("N/A")
            elif is_pct:
                vals.append(f"{v*100:.1f}%")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}" if v < 10 else f"{v:.0f}")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    table = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table)
        logger.info(f"Comparison table written | path={output_path}")

    return table


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SLM models on chart analysis test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=False,
        default=None,
        help="Path to base model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (optional)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/slm_training_v3/test.json",
        help="Path to test JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation results JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling across question/chart types",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use fp16)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Paths to evaluation JSONs for comparison table",
    )
    parser.add_argument(
        "--compare-output",
        type=str,
        default=None,
        help="Output path for comparison markdown table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Mode 1: Generate comparison table from existing results
    if args.compare:
        table = generate_comparison_table(
            args.compare,
            output_path=args.compare_output,
        )
        print(table)
        return

    # Mode 2: Run evaluation
    if not args.base_model:
        parser_err = "error: --base-model is required when not using --compare mode"
        print(parser_err)
        raise SystemExit(2)

    # Determine output path
    if args.output is None:
        model_tag = Path(args.base_model).name
        lora_tag = f"_lora_{Path(args.lora_path).parent.name}" if args.lora_path else "_base"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"models/evaluation/{model_tag}{lora_tag}_{ts}.json"

    # Load model
    pipe, tokenizer, model_name = load_local_model(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        load_in_4bit=not args.no_4bit,
    )
    logger.info(f"Model ready | name={model_name}")

    # Load test data
    test_data = load_test_data(
        args.test_data,
        max_samples=args.max_samples,
        stratified=args.stratified,
        seed=args.seed,
    )

    # Tag source path on samples for reporting
    for sample in test_data:
        sample["_source_path"] = args.test_data

    # Run evaluation
    eval_result = evaluate(
        pipe=pipe,
        tokenizer=tokenizer,
        test_data=test_data,
        model_name=model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Add metadata
    eval_result["test_data_path"] = args.test_data
    eval_result["args"] = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "max_samples": args.max_samples,
        "stratified": args.stratified,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "load_in_4bit": not args.no_4bit,
        "seed": args.seed,
    }

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved | path={output_path}")

    # Print report
    print_report(eval_result)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
