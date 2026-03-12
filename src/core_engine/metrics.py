"""
Pipeline Metrics

Lightweight dataclasses for collecting per-stage and per-run metrics.
Populated by PipelineBuilder.run() and stored in PipelineResult.extra.

These metrics power:
  - Live progress reporting in the API layer
  - Benchmark comparisons between adapter configurations
  - Auto-generated system statistics reports
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageMetrics:
    """
    Execution metrics for a single pipeline stage.

    Collected automatically by PipelineBuilder during run().
    """

    stage_key: str
    """Stage identifier, e.g. 's2_detection'."""

    adapter_name: str
    """Name of the adapter that was used, e.g. 'yolov8'."""

    start_time: float = field(default_factory=time.time)
    """Unix timestamp when the stage began (set by PipelineBuilder)."""

    end_time: Optional[float] = None
    """Unix timestamp when the stage finished."""

    success: bool = False
    """True if the stage completed without raising an exception."""

    output_count: int = 0
    """
    Semantic output count:
    - Stage 1: number of images extracted
    - Stage 2: number of charts detected
    - Stage 3: number of chart metadata objects
    - Stage 4: number of refined chart objects
    - Stage 5: number of insights generated
    """

    error_message: Optional[str] = None
    """Error description if success=False."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Adapter-specific counters (e.g. skipped_low_confidence for Stage 2)."""

    @property
    def duration_ms(self) -> Optional[int]:
        """Wall-clock duration in milliseconds, or None if not complete."""
        if self.end_time is None:
            return None
        return int((self.end_time - self.start_time) * 1000)

    def finish(self, success: bool, output_count: int = 0, **extra: Any) -> None:
        """
        Mark the stage as finished.

        Args:
            success:      True if the stage succeeded.
            output_count: Semantic output item count.
            **extra:      Additional adapter-specific values.
        """
        self.end_time = time.time()
        self.success = success
        self.output_count = output_count
        self.extra.update(extra)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_key,
            "adapter": self.adapter_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "output_count": self.output_count,
            "error": self.error_message,
            **self.extra,
        }


@dataclass
class PipelineMetrics:
    """
    Aggregated execution metrics for a complete pipeline run.

    Attached to PipelineResult.metrics.
    Serialisable to JSON for API responses and benchmark reports.
    """

    session_id: str
    source_file: str
    adapter_config: Dict[str, str] = field(default_factory=dict)
    """Maps stage_key -> adapter_name for this run, e.g. {'s2_detection': 'yolov8'}."""

    stages: List[StageMetrics] = field(default_factory=list)
    """Ordered list of per-stage metrics."""

    total_start_time: float = field(default_factory=time.time)
    total_end_time: Optional[float] = None

    @property
    def total_duration_ms(self) -> Optional[int]:
        if self.total_end_time is None:
            return None
        return int((self.total_end_time - self.total_start_time) * 1000)

    @property
    def succeeded(self) -> bool:
        return all(m.success for m in self.stages)

    @property
    def failed_stage(self) -> Optional[str]:
        for m in self.stages:
            if not m.success:
                return m.stage_key
        return None

    def finish(self) -> None:
        """Mark the full pipeline as finished."""
        self.total_end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_file": self.source_file,
            "adapter_config": self.adapter_config,
            "total_duration_ms": self.total_duration_ms,
            "succeeded": self.succeeded,
            "failed_stage": self.failed_stage,
            "stages": [s.to_dict() for s in self.stages],
        }
