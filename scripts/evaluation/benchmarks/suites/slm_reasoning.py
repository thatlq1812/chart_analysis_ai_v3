"""
SLM Reasoning Benchmark Suite

Evaluates a fine-tuned SLM on chart QA tasks using the chart_qa_v2 dataset.
Requires a trained LoRA adapter in models/slm/.

This suite is designed to run AFTER SLM training completes and measures
the model's ability to answer questions about charts given:
  - A VLM-extracted table (from DePlot/MatCha)
  - A natural language question (from Gemini QA pairs)

Metrics (ChartQA standard):
    exact_match   - Exact answer match (case-insensitive)
    contains      - Ground truth answer appears in prediction
    numeric_acc   - Numeric relative error within 5% (ChartQA standard)
    anls_mean     - Mean ANLS across all QA pairs
    by_q_type     - Breakdown by question type (extraction, trend, comparison, etc.)

Usage:
    runner.run_suite("slm_reasoning",
        model_path="models/slm/qwen2.5-7b-chart-lora-v4/final",
        qa_dir="data/academic_dataset/chart_qa_v2/generated",
    )
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..metrics import anls, exact_match, contains_match, numeric_accuracy
from ..registry import REGISTRY, BenchmarkResult, BenchmarkSuite, PerChartResult

logger = logging.getLogger(__name__)

# Question types for stratified analysis
Q_TYPE_GROUPS = {
    "extraction": {"value", "max", "min", "range", "threshold"},
    "structural": {"structural", "layout", "element_count"},
    "trend": {"trend", "comparison", "interpolation"},
    "reasoning": {"why_reasoning", "multi_hop", "percentage_change", "prediction"},
}


@REGISTRY.register("slm_reasoning")
class SLMReasoningSuite(BenchmarkSuite):
    """
    Benchmark SLM chart reasoning on QA pairs.

    Requires:
      1. Trained LoRA adapter (model_path)
      2. VLM cache for chart tables (vlm_cache_dir) OR runs DePlot on-the-fly
      3. QA pairs (qa_dir = data/academic_dataset/chart_qa_v2/generated/)
    """

    description = "SLM chart QA accuracy (EM, ANLS, Numeric) on Gemini QA pairs"

    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        model_path: Optional[str] = None,
        qa_dir: Optional[Path] = None,
        vlm_cache_dir: Optional[Path] = None,
        device: str = "cuda",
        max_new_tokens: int = 256,
        n_questions_per_chart: int = 5,
        **kwargs: Any,
    ) -> BenchmarkResult:
        qa_dir = qa_dir or (PROJECT_ROOT / "data" / "academic_dataset" / "chart_qa_v2" / "generated")
        vlm_cache_dir = vlm_cache_dir or (PROJECT_ROOT / "data" / "academic_dataset" / "stage3_vlm_cache")
        config = {
            "model_path": model_path or "not_set",
            "qa_dir": str(qa_dir),
            "vlm_cache_dir": str(vlm_cache_dir),
            "device": device,
            "max_new_tokens": max_new_tokens,
            "n_questions_per_chart": n_questions_per_chart,
        }

        if not model_path:
            logger.warning(
                "No model_path provided. SLM benchmark requires a trained adapter. "
                "Run SLM training first. Skipping inference."
            )
            return BenchmarkResult(
                suite_name="slm_reasoning",
                run_id="",
                config=config,
                aggregate={"status": 0.0},
                metadata={"skipped": True, "reason": "no model_path"},
                n_success=0,
                n_error=0,
            )

        logger.info(f"SLM Reasoning benchmark | model={model_path} | charts={len(chart_ids)}")

        # Load SLM
        model, tokenizer = self._load_slm(model_path, device)

        per_chart_results: List[PerChartResult] = []
        all_em: List[float] = []
        all_anls: List[float] = []
        all_contains: List[float] = []
        all_numeric: List[float] = []
        q_type_scores: Dict[str, List[float]] = {}
        latencies: List[float] = []

        for chart_id in chart_ids:
            ann = self._load_annotation(chart_id, annotations_dir)
            if ann is None:
                continue

            chart_type = ann.get("chart_type", "unknown")
            difficulty = ann.get("difficulty", "unknown")

            # Load QA pairs for this chart
            qa_pairs = self._load_qa_pairs(chart_id, chart_type, qa_dir, n_questions_per_chart)
            if not qa_pairs:
                logger.debug(f"No QA pairs | chart_id={chart_id}")
                continue

            # Load or extract VLM table
            table_str = self._load_vlm_table(chart_id, vlm_cache_dir)
            if not table_str:
                logger.warning(f"No VLM table cached | chart_id={chart_id} | skipping. "
                               "Run batch VLM extraction first.")
                continue

            chart_scores: List[float] = []
            qa_details: List[Dict] = []

            for qa in qa_pairs:
                question = qa.get("question", "")
                answer_gt = str(qa.get("answer", ""))
                q_type = qa.get("question_type", "unknown")

                try:
                    t0 = time.time()
                    pred = self._ask_slm(
                        model, tokenizer, table_str, question,
                        max_new_tokens=max_new_tokens,
                    )
                    latency = time.time() - t0
                    latencies.append(latency)

                    em = 1.0 if exact_match(pred, answer_gt) else 0.0
                    anls_score = anls(pred, answer_gt)
                    contains_score = 1.0 if contains_match(pred, answer_gt) else 0.0
                    num_score = 1.0 if numeric_accuracy(pred, answer_gt, tolerance=0.05) else 0.0

                    all_em.append(em)
                    all_anls.append(anls_score)
                    all_contains.append(contains_score)
                    all_numeric.append(num_score)
                    chart_scores.append(anls_score)

                    # Per question-type tracking
                    q_group = self._get_q_group(q_type)
                    q_type_scores.setdefault(q_group, []).append(anls_score)

                    qa_details.append({
                        "question": question,
                        "answer_gt": answer_gt,
                        "answer_pred": pred,
                        "q_type": q_type,
                        "em": em,
                        "anls": round(anls_score, 4),
                        "contains": contains_score,
                        "numeric": num_score,
                        "latency_s": round(latency, 2),
                    })

                except Exception as e:
                    logger.error(
                        f"SLM inference failed | chart_id={chart_id} | q={question[:40]} | error={e}"
                    )

            mean_anls = sum(chart_scores) / len(chart_scores) if chart_scores else 0.0
            logger.info(
                f"SLM | chart_id={chart_id} | chart_type={chart_type} | "
                f"qa_count={len(qa_pairs)} | mean_anls={mean_anls:.3f}"
            )

            per_chart_results.append(
                PerChartResult(
                    chart_id=chart_id,
                    chart_type=chart_type,
                    difficulty=difficulty,
                    success=True,
                    scores={"anls_mean": round(mean_anls, 4)},
                    details={"qa_pairs": qa_details[:10]},  # store first 10 for report
                )
            )

        # Aggregate
        def safe_mean(lst: List[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        aggregate: Dict[str, float] = {
            "exact_match": safe_mean(all_em),
            "anls_mean": safe_mean(all_anls),
            "contains": safe_mean(all_contains),
            "numeric_acc": safe_mean(all_numeric),
            "mean_latency_s": safe_mean(latencies),
        }
        for q_group, scores in q_type_scores.items():
            aggregate[f"anls_{q_group}"] = safe_mean(scores)

        n_success = len(per_chart_results)
        return BenchmarkResult(
            suite_name="slm_reasoning",
            run_id="",
            config=config,
            per_chart=per_chart_results,
            aggregate=aggregate,
            metadata={
                "n_charts": len(chart_ids),
                "n_qa_total": len(all_em),
                "model_path": model_path,
            },
            n_success=n_success,
            n_error=len(chart_ids) - n_success,
        )

    # -------------------------------------------------------------------------
    # SLM inference
    # -------------------------------------------------------------------------

    def _load_slm(self, model_path: str, device: str):
        """Load LoRA-merged SLM for inference."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            base_path = Path(model_path)
            adapter_path = base_path if (base_path / "adapter_config.json").exists() else None

            # Load base model name from adapter_config if present
            if adapter_path:
                adapter_cfg = json.loads((adapter_path / "adapter_config.json").read_text())
                base_model_name = adapter_cfg.get("base_model_name_or_path", "")
            else:
                base_model_name = model_path

            logger.info(f"Loading tokenizer from {base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            logger.info(f"Loading base model: {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
            )

            if adapter_path:
                logger.info(f"Loading LoRA adapter from {adapter_path}")
                model = PeftModel.from_pretrained(model, str(adapter_path))
                model = model.merge_and_unload()

            model.eval()
            logger.info("SLM loaded and ready")
            return model, tokenizer

        except Exception as e:
            raise RuntimeError(f"Failed to load SLM: {e}") from e

    def _ask_slm(
        self,
        model: Any,
        tokenizer: Any,
        table_str: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Build ChatML prompt and run SLM inference."""
        import torch

        system = (
            "You are a chart reasoning expert. "
            "Analyze chart data from the provided table and answer questions accurately. "
            "For numeric answers, provide the exact number. "
            "For trend questions, be concise and specific."
        )
        user_content = f"[TABLE]\n{table_str}\n\n[QUESTION]\n{question}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        # Use chat template if available, otherwise build manually
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_qa_pairs(
        self,
        chart_id: str,
        chart_type: str,
        qa_dir: Path,
        n: int,
    ) -> List[Dict]:
        """Load QA pairs for a chart from chart_qa_v2 directory."""
        # Try direct chart_id file first
        direct = qa_dir / chart_type / f"{chart_id}.json"
        if direct.exists():
            try:
                data = json.loads(direct.read_text(encoding="utf-8"))
                pairs = data.get("qa_pairs", data.get("pairs", []))
                return pairs[:n]
            except Exception:
                pass
        # Search in all type subdirs
        for type_dir in qa_dir.iterdir():
            if not type_dir.is_dir():
                continue
            p = type_dir / f"{chart_id}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    pairs = data.get("qa_pairs", data.get("pairs", []))
                    return pairs[:n]
                except Exception:
                    pass
        return []

    def _load_vlm_table(self, chart_id: str, vlm_cache_dir: Path) -> Optional[str]:
        """Load cached VLM table output for a chart."""
        subdirs = [vlm_cache_dir] + (list(vlm_cache_dir.iterdir()) if vlm_cache_dir.exists() else [])
        for subdir in subdirs:
            if not isinstance(subdir, Path):
                continue
            p = subdir / f"{chart_id}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    return data.get("table_str") or data.get("raw_output") or ""
                except Exception:
                    pass
        return None

    @staticmethod
    def _get_q_group(q_type: str) -> str:
        """Map question type string to Q_TYPE_GROUPS key."""
        q_lower = q_type.lower()
        for group, types in Q_TYPE_GROUPS.items():
            if q_lower in types or any(t in q_lower for t in types):
                return group
        return "other"
