"""
Benchmark Suite Registry

Defines the ABC all benchmark suites must implement, the BenchmarkResult
schema, and the global REGISTRY that maps suite names to suite classes.

All benchmark suites are auto-registered via the @register decorator.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class PerChartResult:
    """Per-chart result for a single benchmark suite run."""

    chart_id: str
    chart_type: str
    difficulty: str
    success: bool
    error: Optional[str] = None
    latency_s: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """
    Complete result from a single benchmark suite run.

    Saved to: data/benchmark/results/runs/{suite_name}_{timestamp}/results.json
    """

    suite_name: str
    run_id: str                          # "{suite_name}_{YYYYMMDD_HHMMSS}"
    config: Dict[str, Any]              # What was run (model names, flags, etc.)
    per_chart: List[PerChartResult] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_latency_s: float = 0.0
    n_success: int = 0
    n_error: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, output_dir: Path) -> Path:
        """Save results as JSON + Markdown report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "results.json"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path = output_dir / "report.md"
        md_path.write_text(self._to_markdown(), encoding="utf-8")
        return json_path

    def _to_markdown(self) -> str:
        lines = [
            f"# Benchmark Report: {self.suite_name}",
            f"",
            f"**Run ID**: {self.run_id}",
            f"**Charts**: {self.n_success} success / {self.n_error} errors",
            f"**Total time**: {self.total_latency_s:.1f}s",
            f"",
            f"## Config",
            f"```json",
            json.dumps(self.config, indent=2, ensure_ascii=False),
            f"```",
            f"",
            f"## Aggregate Scores",
            f"",
            "| Metric | Score |",
            "| --- | --- |",
        ]
        for k, v in sorted(self.aggregate.items()):
            lines.append(f"| {k} | {v:.4f} |")

        lines += [
            f"",
            f"## Per-Chart Results",
            f"",
            "| chart_id | type | difficulty | success | latency | " +
            " | ".join(sorted(self.aggregate.keys())[:5]) + " |",
            "| --- | --- | --- | --- | --- |" + " --- |" * min(5, len(self.aggregate)),
        ]
        for r in self.per_chart:
            score_cols = " | ".join(
                f"{r.scores.get(k, 0.0):.3f}"
                for k in sorted(self.aggregate.keys())[:5]
            )
            ok = "OK" if r.success else f"ERR: {r.error or ''}"
            lines.append(
                f"| {r.chart_id[:40]} | {r.chart_type} | {r.difficulty} | "
                f"{ok} | {r.latency_s:.1f}s | {score_cols} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base suite ABC
# ---------------------------------------------------------------------------

class BenchmarkSuite(ABC):
    """
    Abstract base class for all benchmark suites.

    Subclasses register themselves with @REGISTRY.register("suite_name").
    The runner calls run() and receives a BenchmarkResult.
    """

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def run(
        self,
        chart_ids: List[str],
        images_dir: Path,
        annotations_dir: Path,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """
        Execute benchmark on a list of chart IDs.

        Args:
            chart_ids: List of chart_id strings to evaluate
            images_dir: Directory containing chart image files
            annotations_dir: Directory containing GT annotation JSON files
            **kwargs: Suite-specific kwargs (model names, device, etc.)

        Returns:
            BenchmarkResult with per-chart and aggregate scores
        """

    def _load_annotation(self, chart_id: str, annotations_dir: Path) -> Optional[Dict]:
        """Load and return annotation dict for a chart_id, or None on error."""
        ann_path = annotations_dir / f"{chart_id}.json"
        if not ann_path.exists():
            return None
        try:
            return json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _find_image(self, chart_id: str, images_dir: Path) -> Optional[Path]:
        """Find image file for chart_id, checking multiple extensions."""
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = images_dir / f"{chart_id}{ext}"
            if p.exists():
                return p
            # Also check one level deep (type subdirs)
            for sub in images_dir.iterdir():
                if sub.is_dir():
                    p2 = sub / f"{chart_id}{ext}"
                    if p2.exists():
                        return p2
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SuiteRegistry:
    """Registry mapping suite names to BenchmarkSuite classes."""

    def __init__(self) -> None:
        self._suites: Dict[str, Type[BenchmarkSuite]] = {}

    def register(self, name: str):
        """Decorator: @REGISTRY.register("vlm_extraction")"""
        def decorator(cls: Type[BenchmarkSuite]) -> Type[BenchmarkSuite]:
            cls.name = name
            self._suites[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type[BenchmarkSuite]:
        if name not in self._suites:
            available = list(self._suites.keys())
            raise KeyError(f"Unknown suite '{name}'. Available: {available}")
        return self._suites[name]

    def list_names(self) -> List[str]:
        return list(self._suites.keys())

    def __repr__(self) -> str:
        return f"SuiteRegistry({list(self._suites.keys())})"


# Global registry instance
REGISTRY = SuiteRegistry()
