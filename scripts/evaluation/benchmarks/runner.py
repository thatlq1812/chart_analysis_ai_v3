"""
Benchmark Runner

Unified execution engine for all benchmark suites.
Handles:
  - Loading the benchmark chart set (from manifest)
  - Running one or more suites sequentially
  - Structured logging (stdout + file)
  - Saving results per run + updating global runs_registry.json
  - Comparison table across multiple suite runs

Usage:
    runner = BenchmarkRunner()
    result = runner.run_suite("vlm_extraction", models=["deplot", "matcha"])
    runner.run_suite("ocr_quality")
    runner.print_registry()
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from .registry import REGISTRY, BenchmarkResult

# Auto-import all suites so they register themselves
from .suites import vlm_extraction  # noqa: F401
from .suites import ocr_quality     # noqa: F401
from .suites import classifier      # noqa: F401
from .suites import slm_reasoning   # noqa: F401
from .suites import baseline_vlm    # noqa: F401
from .suites import e2e_pipeline    # noqa: F401
from .suites import ablation        # noqa: F401

logger = logging.getLogger(__name__)

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results" / "runs"
REGISTRY_FILE = BENCHMARK_DIR / "results" / "runs_registry.json"


def _setup_logging(log_dir: Path) -> None:
    """Configure logging to stdout + rotating log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


class BenchmarkRunner:
    """
    Unified benchmark runner for all suites.

    Example:
        runner = BenchmarkRunner()
        runner.run_suite("vlm_extraction", models=["deplot", "matcha"])
        runner.run_suite("ocr_quality")
        runner.print_registry()
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        annotations_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        n_charts: Optional[int] = None,
    ) -> None:
        self.manifest_path = manifest_path or (BENCHMARK_DIR / "benchmark_manifest.json")
        self.images_dir = images_dir or (PROJECT_ROOT / "data" / "academic_dataset" / "classified_charts")
        self.annotations_dir = annotations_dir or (BENCHMARK_DIR / "annotations")
        self.results_dir = results_dir or RESULTS_DIR
        self.n_charts = n_charts  # None = use all from manifest

        self._manifest: Optional[Dict] = None
        self._chart_ids: Optional[List[str]] = None

    def _load_manifest(self) -> List[str]:
        """Load chart IDs from benchmark manifest."""
        if self._chart_ids is not None:
            return self._chart_ids
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._manifest = data
        # Manifest has strata_spec with chart_ids per type
        chart_ids: List[str] = []
        if "chart_ids" in data:
            chart_ids = data["chart_ids"]
        elif "strata_spec" in data:
            # Collect chart_ids from annotation files
            for ann_file in self.annotations_dir.glob("*.json"):
                chart_ids.append(ann_file.stem)
        if self.n_charts is not None:
            chart_ids = chart_ids[: self.n_charts]
        self._chart_ids = chart_ids
        logger.info(f"Loaded {len(chart_ids)} charts from manifest | path={self.manifest_path}")
        return chart_ids

    def run_suite(
        self,
        suite_name: str,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Run a single benchmark suite and save results.

        Args:
            suite_name: Registered suite name (e.g. "vlm_extraction")
            **kwargs: Passed to suite.run() (model names, device, etc.)

        Returns:
            BenchmarkResult
        """
        suite_cls = REGISTRY.get(suite_name)
        suite = suite_cls()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{suite_name}_{timestamp}"
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        _setup_logging(run_dir)

        logger.info(f"=== Benchmark Suite: {suite_name} | run_id={run_id} ===")
        if kwargs:
            logger.info(f"Config: {kwargs}")

        chart_ids = self._load_manifest()
        logger.info(f"Charts: {len(chart_ids)}")

        t0 = time.time()
        result = suite.run(
            chart_ids=chart_ids,
            images_dir=self.images_dir,
            annotations_dir=self.annotations_dir,
            **kwargs,
        )
        result.run_id = run_id
        result.total_latency_s = time.time() - t0

        # Save results
        result.save(run_dir)
        logger.info(
            f"Results saved | dir={run_dir} | "
            f"success={result.n_success} | errors={result.n_error} | "
            f"time={result.total_latency_s:.1f}s"
        )

        # Print aggregate scores
        logger.info("--- Aggregate Scores ---")
        for k, v in sorted(result.aggregate.items()):
            logger.info(f"  {k}: {v:.4f}")

        # Update registry
        self._update_registry(result, run_dir)

        return result

    def run_suites(
        self,
        suite_names: List[str],
        **shared_kwargs: Any,
    ) -> List[BenchmarkResult]:
        """Run multiple suites sequentially."""
        results = []
        for name in suite_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting suite: {name}")
            result = self.run_suite(name, **shared_kwargs)
            results.append(result)
        self._print_comparison(results)
        return results

    def _update_registry(self, result: BenchmarkResult, run_dir: Path) -> None:
        """Append run summary to runs_registry.json."""
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        registry: List[Dict] = []
        if REGISTRY_FILE.exists():
            try:
                registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            except Exception:
                registry = []

        entry = {
            "run_id": result.run_id,
            "suite_name": result.suite_name,
            "timestamp": result.run_id.split("_", 1)[-1] if "_" in result.run_id else "",
            "n_success": result.n_success,
            "n_error": result.n_error,
            "total_latency_s": round(result.total_latency_s, 1),
            "aggregate": {k: round(v, 4) for k, v in result.aggregate.items()},
            "config": result.config,
            "result_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        }
        registry.append(entry)
        REGISTRY_FILE.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Registry updated | {REGISTRY_FILE}")

    def _print_comparison(self, results: List[BenchmarkResult]) -> None:
        """Print a comparison table across all results."""
        if not results:
            return
        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON SUMMARY")
        print("=" * 70)
        all_keys = sorted({k for r in results for k in r.aggregate})
        header = f"{'Suite':<25} " + " ".join(f"{k[:12]:<14}" for k in all_keys)
        print(header)
        print("-" * len(header))
        for r in results:
            scores = " ".join(
                f"{r.aggregate.get(k, 0.0):<14.4f}" for k in all_keys
            )
            print(f"{r.suite_name:<25} {scores}")
        print("=" * 70)

    def print_registry(self) -> None:
        """Print a table of all past benchmark runs."""
        if not REGISTRY_FILE.exists():
            print("No benchmark runs found.")
            return
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        print(f"\n{'='*80}")
        print(f"BENCHMARK RUNS REGISTRY ({len(registry)} runs)")
        print(f"{'='*80}")
        print(f"{'run_id':<45} {'success':>8} {'errors':>6} {'time':>7}")
        print("-" * 80)
        for entry in registry[-20:]:  # Show last 20 runs
            print(
                f"{entry['run_id']:<45} "
                f"{entry['n_success']:>8} "
                f"{entry['n_error']:>6} "
                f"{entry['total_latency_s']:>6.1f}s"
            )
            if entry.get("aggregate"):
                scores_str = "  " + " | ".join(
                    f"{k}={v:.3f}" for k, v in sorted(entry["aggregate"].items())
                )
                print(scores_str)
        print("=" * 80)
