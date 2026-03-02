"""
Stage 5: Insight & Reporting

Final pipeline stage. Takes refined chart data from Stage 4 and produces:
1. Insight generation: trend, comparison, anomaly, summary per chart
2. Schema validation: ensure output conforms to FinalChartResult
3. JSON serialization with traceability metadata
4. Text summary report for the full session

Input:  Stage4Output (list of RefinedChartData)
Output: PipelineResult (list of FinalChartResult + session summary)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..exceptions import StageProcessingError
from ..schemas.enums import ChartType
from ..schemas.stage_outputs import (
    ChartInsight,
    DataSeries,
    FinalChartResult,
    PipelineResult,
    RefinedChartData,
    SessionInfo,
    Stage4Output,
)
from .base import BaseStage

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ReportingConfig(BaseModel):
    """Configuration for Stage 5: Reporting."""

    # Insight generation
    enable_insights: bool = Field(
        default=True,
        description="Generate trend/comparison/anomaly insights",
    )
    max_insights_per_chart: int = Field(
        default=5,
        ge=1,
        description="Maximum number of insights to generate per chart",
    )
    min_series_for_comparison: int = Field(
        default=2,
        description="Minimum series count to generate comparison insights",
    )
    anomaly_z_score_threshold: float = Field(
        default=2.0,
        description="Z-score threshold to flag a data point as anomalous",
    )

    # Output options
    save_json: bool = Field(default=True, description="Save output JSON to disk")
    save_report: bool = Field(default=True, description="Save text report to disk")
    save_markdown: bool = Field(default=True, description="Save Markdown report to disk")
    save_csv: bool = Field(default=False, description="Save data as CSV files")
    output_dir: str = Field(
        default="data/output",
        description="Directory for output files",
    )

    # Validation
    require_description: bool = Field(
        default=True,
        description="Fail chart if description is empty",
    )


# =============================================================================
# Stage 5 Implementation
# =============================================================================


class Stage5Reporting(BaseStage):
    """
    Stage 5 Orchestrator: Insight & Reporting.

    Generates human-readable insights from structured chart data,
    validates output schema, and writes JSON + text reports.

    Example:
        config = ReportingConfig()
        stage = Stage5Reporting(config)
        result = stage.process(stage4_output)
        print(result.summary)
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        config = config or ReportingConfig()
        super().__init__(config)
        self.config: ReportingConfig = config
        self._start_time: Optional[float] = None

    def process(self, input_data: Stage4Output) -> PipelineResult:
        """
        Generate insights and produce the final pipeline result.

        Args:
            input_data: Stage4Output with refined chart data

        Returns:
            PipelineResult with insights, JSON export, and session summary

        Raises:
            StageProcessingError: If a critical validation fails
        """
        if not self.validate_input(input_data):
            raise StageProcessingError(
                message="Invalid input: expected Stage4Output",
                stage="s5_reporting",
                recoverable=False,
            )

        self._start_time = time.time()
        session = input_data.session

        logger.info(
            f"Reporting started | session={session.session_id} | "
            f"charts={len(input_data.charts)}"
        )

        final_charts: List[FinalChartResult] = []
        all_warnings: List[str] = []

        for chart in input_data.charts:
            # Validate chart data
            chart_warnings = self._validate_chart(chart, session.session_id)
            all_warnings.extend(chart_warnings)
            for w in chart_warnings:
                logger.warning(w)

            try:
                final = self._process_single_chart(chart, session)
                final_charts.append(final)
            except Exception as exc:
                logger.error(
                    f"Reporting failed for chart | "
                    f"chart_id={chart.chart_id} | error={exc}"
                )
                # Include chart with empty insights rather than dropping it
                final_charts.append(
                    FinalChartResult(
                        chart_id=chart.chart_id,
                        chart_type=chart.chart_type,
                        title=chart.title,
                        data=chart,
                        insights=[
                            ChartInsight(
                                insight_type="error",
                                text=f"Insight generation failed: {exc}",
                                confidence=0.0,
                            )
                        ],
                        source_info={"error": str(exc)},
                    )
                )

        elapsed = time.time() - (self._start_time or time.time())
        summary = self._generate_session_summary(session, final_charts, elapsed)

        result = PipelineResult(
            session=session,
            charts=final_charts,
            summary=summary,
            processing_time_seconds=round(elapsed, 3),
            model_versions=self._collect_model_versions(),
            warnings=all_warnings,
        )

        # Write output files
        if self.config.save_json or self.config.save_report or self.config.save_markdown or self.config.save_csv:
            self._write_outputs(result)

        logger.info(
            f"Reporting complete | session={session.session_id} | "
            f"charts={len(final_charts)} | elapsed={elapsed:.2f}s"
        )

        return result

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input is Stage4Output."""
        return isinstance(input_data, Stage4Output)

    # -------------------------------------------------------------------------
    # Data validation
    # -------------------------------------------------------------------------

    def _validate_chart(
        self,
        chart: RefinedChartData,
        session_id: str,
    ) -> List[str]:
        """
        Validate a single chart's data integrity.

        Checks:
        - Required fields (chart_id, chart_type)
        - Description present (if required by config)
        - Series consistency: all series should have > 0 points
        - Value sanity: no NaN or infinity in data points

        Args:
            chart: Refined chart data from Stage 4
            session_id: Session identifier for log context

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings: List[str] = []

        # Required field: chart_id must be non-empty
        if not chart.chart_id or not chart.chart_id.strip():
            warnings.append(
                f"Chart has empty chart_id | session={session_id}"
            )

        # Description required check
        if self.config.require_description and not chart.description:
            warnings.append(
                f"Chart missing description | chart_id={chart.chart_id} | "
                f"session={session_id}"
            )

        if not chart.series:
            warnings.append(
                f"Chart has no data series | chart_id={chart.chart_id} | "
                f"session={session_id}"
            )
            return warnings

        for series in chart.series:
            # Empty series
            if not series.points:
                warnings.append(
                    f"Empty series | chart_id={chart.chart_id} | "
                    f"series={series.name} | session={session_id}"
                )
                continue

            # Check for non-finite values
            import math

            for point in series.points:
                if math.isnan(point.value) or math.isinf(point.value):
                    warnings.append(
                        f"Non-finite value detected | chart_id={chart.chart_id} | "
                        f"series={series.name} | label={point.label} | "
                        f"value={point.value} | session={session_id}"
                    )
                    break  # One warning per series is enough

            # Check confidence ranges
            low_conf = [
                p for p in series.points if p.confidence < 0.5
            ]
            if low_conf:
                warnings.append(
                    f"Low confidence points | chart_id={chart.chart_id} | "
                    f"series={series.name} | "
                    f"count={len(low_conf)}/{len(series.points)} | "
                    f"session={session_id}"
                )

        return warnings

    # -------------------------------------------------------------------------
    # Single-chart processing
    # -------------------------------------------------------------------------

    def _process_single_chart(
        self,
        chart: RefinedChartData,
        session: SessionInfo,
    ) -> FinalChartResult:
        """
        Build FinalChartResult for a single chart.

        Args:
            chart: Refined chart data from Stage 4
            session: Session metadata for traceability

        Returns:
            FinalChartResult with insights and provenance
        """
        insights: List[ChartInsight] = []

        if self.config.enable_insights:
            insights = self._generate_insights(chart)

        source_info = self._build_source_info(chart, session)

        return FinalChartResult(
            chart_id=chart.chart_id,
            chart_type=chart.chart_type,
            title=chart.title,
            data=chart,
            insights=insights[: self.config.max_insights_per_chart],
            source_info=source_info,
        )

    # -------------------------------------------------------------------------
    # Insight generation
    # -------------------------------------------------------------------------

    def _generate_insights(self, chart: RefinedChartData) -> List[ChartInsight]:
        """
        Generate insights for a single chart.

        Generates up to max_insights_per_chart across categories:
        - summary: Overall description
        - trend: Increasing/decreasing/stable trend per series
        - comparison: Relative comparison across series
        - anomaly: Outlier data points

        Args:
            chart: Refined chart data

        Returns:
            List of ChartInsight objects (may be empty if no data)
        """
        insights: List[ChartInsight] = []

        if not chart.series:
            return insights

        # 1. Summary insight
        insights.append(self._summary_insight(chart))

        # 2. Trend insights (per series)
        for series in chart.series:
            trend = self._detect_trend(series)
            if trend:
                insights.append(trend)

        # 3. Comparison insight (if multiple series)
        if len(chart.series) >= self.config.min_series_for_comparison:
            comparison = self._comparison_insight(chart)
            if comparison:
                insights.append(comparison)

        # 4. Anomaly insights
        anomalies = self._detect_anomalies(chart)
        insights.extend(anomalies)

        return insights

    def _summary_insight(self, chart: RefinedChartData) -> ChartInsight:
        """Generate a summary insight from chart metadata."""
        total_points = sum(len(s.points) for s in chart.series)
        series_names = [s.name for s in chart.series if s.name != "Unknown"]

        if chart.description:
            text = chart.description
        elif series_names:
            text = (
                f"{chart.chart_type.value.capitalize()} chart with "
                f"{len(chart.series)} series "
                f"({', '.join(series_names[:3])}) "
                f"containing {total_points} data points."
            )
        else:
            text = (
                f"{chart.chart_type.value.capitalize()} chart "
                f"with {total_points} data points."
            )

        return ChartInsight(
            insight_type="summary",
            text=text,
            confidence=0.9 if chart.description else 0.6,
        )

    def _detect_trend(self, series: DataSeries) -> Optional[ChartInsight]:
        """
        Detect increasing / decreasing / stable trend in a series.

        Uses linear regression slope sign for direction.

        Args:
            series: Data series with numeric points

        Returns:
            ChartInsight if trend detected, None if too few points
        """
        points = [p for p in series.points if p.value is not None]
        if len(points) < 3:
            return None

        values = [p.value for p in points]
        n = len(values)
        x = list(range(n))

        # Simple linear regression slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return None

        slope = numerator / denominator
        relative_slope = slope / (abs(y_mean) + 1e-9)

        if relative_slope > 0.05:
            direction = "increasing"
            confidence = min(0.9, 0.5 + abs(relative_slope))
        elif relative_slope < -0.05:
            direction = "decreasing"
            confidence = min(0.9, 0.5 + abs(relative_slope))
        else:
            direction = "stable"
            confidence = 0.7

        first_val = values[0]
        last_val = values[-1]
        pct_change = ((last_val - first_val) / (abs(first_val) + 1e-9)) * 100

        text = (
            f"Series '{series.name}' shows a {direction} trend "
            f"from {first_val:.2g} to {last_val:.2g} "
            f"({pct_change:+.1f}%)."
        )

        return ChartInsight(
            insight_type="trend",
            text=text,
            confidence=round(confidence, 2),
        )

    def _comparison_insight(
        self, chart: RefinedChartData
    ) -> Optional[ChartInsight]:
        """
        Compare series maxima and identify the dominant series.

        Args:
            chart: Chart with multiple series

        Returns:
            ChartInsight for comparison, or None if values unavailable
        """
        series_maxima: List[Tuple[str, float]] = []

        for s in chart.series:
            if s.points:
                max_val = max(p.value for p in s.points)
                series_maxima.append((s.name, max_val))

        if not series_maxima:
            return None

        series_maxima.sort(key=lambda x: x[1], reverse=True)
        dominant_name, dominant_val = series_maxima[0]
        lowest_name, lowest_val = series_maxima[-1]

        if abs(dominant_val) < 1e-9:
            return None

        ratio = dominant_val / (abs(lowest_val) + 1e-9)

        text = (
            f"'{dominant_name}' has the highest peak value ({dominant_val:.3g}), "
            f"{ratio:.1f}x higher than '{lowest_name}' ({lowest_val:.3g})."
        )

        return ChartInsight(insight_type="comparison", text=text, confidence=0.85)

    def _detect_anomalies(
        self, chart: RefinedChartData
    ) -> List[ChartInsight]:
        """
        Flag data points that deviate significantly from series mean.

        Uses z-score method: flag points where |z| > anomaly_z_score_threshold.

        Args:
            chart: Chart data with series

        Returns:
            List of anomaly insights (may be empty)
        """
        anomalies: List[ChartInsight] = []

        for series in chart.series:
            if len(series.points) < 4:
                continue

            values = [p.value for p in series.points]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5

            if std < 1e-9:
                continue

            for point in series.points:
                z = abs(point.value - mean) / std
                if z > self.config.anomaly_z_score_threshold:
                    text = (
                        f"Outlier detected in series '{series.name}': "
                        f"label='{point.label}', value={point.value:.3g} "
                        f"(z-score={z:.1f}, mean={mean:.3g})."
                    )
                    anomalies.append(
                        ChartInsight(
                            insight_type="anomaly",
                            text=text,
                            confidence=round(min(0.95, 0.5 + z * 0.1), 2),
                        )
                    )

        return anomalies

    # -------------------------------------------------------------------------
    # Provenance / traceability
    # -------------------------------------------------------------------------

    def _build_source_info(
        self,
        chart: RefinedChartData,
        session: SessionInfo,
    ) -> Dict[str, Any]:
        """
        Build traceability metadata for a chart result.

        Args:
            chart: Refined chart data
            session: Session info with source file

        Returns:
            Dict with provenance fields
        """
        return {
            "session_id": session.session_id,
            "source_file": str(session.source_file),
            "chart_id": chart.chart_id,
            "chart_type": chart.chart_type.value,
            "corrections_applied": len(chart.correction_log),
            "correction_log": chart.correction_log,
            "total_series": len(chart.series),
            "total_data_points": sum(len(s.points) for s in chart.series),
        }

    # -------------------------------------------------------------------------
    # Session summary
    # -------------------------------------------------------------------------

    def _generate_session_summary(
        self,
        session: SessionInfo,
        charts: List[FinalChartResult],
        elapsed: float,
    ) -> str:
        """
        Generate a human-readable session summary.

        Args:
            session: Session metadata
            charts: All final chart results
            elapsed: Total processing time in seconds

        Returns:
            Multi-line summary string
        """
        if not charts:
            return (
                f"Session {session.session_id}: No charts extracted from "
                f"'{session.source_file}'."
            )

        # Count chart types
        type_counts: Dict[str, int] = {}
        for c in charts:
            t = c.chart_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        type_summary = ", ".join(
            f"{count} {ctype}" for ctype, count in sorted(type_counts.items())
        )
        total_points = sum(
            sum(len(s.points) for s in c.data.series) for c in charts
        )
        total_insights = sum(len(c.insights) for c in charts)

        lines = [
            f"Session: {session.session_id}",
            f"Source: {session.source_file}",
            f"Charts extracted: {len(charts)} ({type_summary})",
            f"Total data points: {total_points}",
            f"Insights generated: {total_insights}",
            f"Processing time: {elapsed:.2f}s",
        ]

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Output writing
    # -------------------------------------------------------------------------

    def _write_outputs(self, result: PipelineResult) -> None:
        """
        Write JSON and text report to the configured output directory.

        Args:
            result: Complete pipeline result
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        session_id = result.session.session_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{session_id}_{ts}"

        if self.config.save_json:
            json_path = output_dir / f"{base_name}.json"
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
                logger.info(f"JSON output written | path={json_path}")
            except Exception as exc:
                logger.warning(f"Failed to write JSON output | error={exc}")

        if self.config.save_report:
            report_path = output_dir / f"{base_name}_report.txt"
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(self._format_text_report(result))
                logger.info(f"Text report written | path={report_path}")
            except Exception as exc:
                logger.warning(f"Failed to write text report | error={exc}")

        if self.config.save_markdown:
            md_path = output_dir / f"{base_name}_report.md"
            try:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(self._format_markdown_report(result))
                logger.info(f"Markdown report written | path={md_path}")
            except Exception as exc:
                logger.warning(f"Failed to write Markdown report | error={exc}")

        if self.config.save_csv:
            csv_path = output_dir / f"{base_name}_data.csv"
            try:
                self._write_csv(result, csv_path)
                logger.info(f"CSV data written | path={csv_path}")
            except Exception as exc:
                logger.warning(f"Failed to write CSV | error={exc}")

    def _format_text_report(self, result: PipelineResult) -> str:
        """
        Format a readable text report.

        Args:
            result: Pipeline result

        Returns:
            Multi-section text report
        """
        lines: List[str] = [
            "=" * 70,
            "CHART ANALYSIS REPORT",
            "=" * 70,
            "",
            result.summary,
            "",
        ]

        for i, chart in enumerate(result.charts, 1):
            lines += [
                "-" * 50,
                f"Chart {i}: {chart.chart_id}",
                f"  Type : {chart.chart_type.value}",
                f"  Title: {chart.title or '(no title)'}",
            ]

            if chart.data.x_axis_label:
                lines.append(f"  X-Axis: {chart.data.x_axis_label}")
            if chart.data.y_axis_label:
                lines.append(f"  Y-Axis: {chart.data.y_axis_label}")

            lines.append("")
            lines.append("  Data Series:")
            for series in chart.data.series:
                values_str = (
                    ", ".join(f"{p.label}={p.value:.3g}" for p in series.points[:5])
                    + (" ..." if len(series.points) > 5 else "")
                )
                lines.append(f"    - {series.name}: {values_str}")

            if chart.insights:
                lines.append("")
                lines.append("  Insights:")
                for insight in chart.insights:
                    lines.append(f"    [{insight.insight_type.upper()}] {insight.text}")

            lines.append("")

        lines += [
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            "=" * 70,
        ]

        return "\n".join(lines)

    def _format_markdown_report(self, result: PipelineResult) -> str:
        """
        Generate a Markdown-formatted report suitable for documentation or thesis.

        Args:
            result: Pipeline result

        Returns:
            Markdown string with tables, headings, and structured data
        """
        lines: List[str] = [
            "# Chart Analysis Report",
            "",
            f"**Session:** {result.session.session_id}  ",
            f"**Source:** {result.session.source_file}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Processing Time:** {result.processing_time_seconds:.2f}s  ",
            "",
            "---",
            "",
            "## Summary",
            "",
        ]

        # Summary table
        total_points = sum(
            sum(len(s.points) for s in c.data.series) for c in result.charts
        )
        total_insights = sum(len(c.insights) for c in result.charts)

        lines += [
            "| Property | Value |",
            "| --- | --- |",
            f"| Charts Extracted | {len(result.charts)} |",
            f"| Total Data Points | {total_points} |",
            f"| Insights Generated | {total_insights} |",
            f"| Processing Time | {result.processing_time_seconds:.2f}s |",
        ]

        # Chart type distribution
        type_counts: Dict[str, int] = {}
        for c in result.charts:
            t = c.chart_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        if type_counts:
            lines += [
                "",
                "### Chart Types",
                "",
                "| Type | Count |",
                "| --- | --- |",
            ]
            for ctype, count in sorted(type_counts.items()):
                lines.append(f"| {ctype} | {count} |")

        lines += ["", "---", ""]

        # Per-chart details
        for i, chart in enumerate(result.charts, 1):
            lines += [
                f"## Chart {i}: {chart.title or '(untitled)'}",
                "",
                f"**ID:** {chart.chart_id}  ",
                f"**Type:** {chart.chart_type.value}  ",
            ]

            if chart.data.x_axis_label:
                lines.append(f"**X-Axis:** {chart.data.x_axis_label}  ")
            if chart.data.y_axis_label:
                lines.append(f"**Y-Axis:** {chart.data.y_axis_label}  ")
            lines.append("")

            # Data table
            if chart.data.series:
                lines.append("### Data")
                lines.append("")

                for series in chart.data.series:
                    if not series.points:
                        continue

                    lines.append(f"**Series: {series.name}** ({len(series.points)} points)")
                    lines.append("")
                    lines += [
                        "| Label | Value | Confidence |",
                        "| --- | --- | --- |",
                    ]
                    for p in series.points[:20]:
                        lines.append(
                            f"| {p.label} | {p.value:.4g} | {p.confidence:.2f} |"
                        )
                    if len(series.points) > 20:
                        lines.append(f"| ... | *{len(series.points) - 20} more* | |")
                    lines.append("")

            # Insights
            if chart.insights:
                lines.append("### Insights")
                lines.append("")
                for insight in chart.insights:
                    icon = {
                        "trend": "**[TREND]**",
                        "comparison": "**[COMPARE]**",
                        "anomaly": "**[ANOMALY]**",
                        "summary": "**[SUMMARY]**",
                    }.get(insight.insight_type, f"**[{insight.insight_type.upper()}]**")
                    lines.append(
                        f"- {icon} {insight.text} *(confidence: {insight.confidence:.2f})*"
                    )
                lines.append("")

            # Source info
            if chart.source_info:
                lines.append("### Traceability")
                lines.append("")
                lines += [
                    "| Field | Value |",
                    "| --- | --- |",
                ]
                for k, v in chart.source_info.items():
                    if k == "correction_log":
                        lines.append(f"| {k} | {len(v)} corrections |")
                    else:
                        lines.append(f"| {k} | {v} |")
                lines.append("")

            lines += ["---", ""]

        # Warnings section
        if result.warnings:
            lines += [
                "## Warnings",
                "",
            ]
            for w in result.warnings:
                lines.append(f"- {w}")
            lines += ["", "---", ""]

        # Footer
        lines += [
            f"*Report generated by Geo-SLM Chart Analysis Pipeline*",
        ]

        return "\n".join(lines)

    def _write_csv(self, result: PipelineResult, csv_path: Path) -> None:
        """
        Write extracted chart data as a flat CSV file.

        Each row represents a single data point with full context.

        Args:
            result: Pipeline result
            csv_path: Output CSV file path
        """
        import csv

        fieldnames = [
            "session_id",
            "chart_id",
            "chart_type",
            "chart_title",
            "series_name",
            "label",
            "value",
            "unit",
            "confidence",
            "x_axis_label",
            "y_axis_label",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for chart in result.charts:
                for series in chart.data.series:
                    for point in series.points:
                        writer.writerow(
                            {
                                "session_id": result.session.session_id,
                                "chart_id": chart.chart_id,
                                "chart_type": chart.chart_type.value,
                                "chart_title": chart.title or "",
                                "series_name": series.name,
                                "label": point.label,
                                "value": point.value,
                                "unit": point.unit or "",
                                "confidence": point.confidence,
                                "x_axis_label": chart.data.x_axis_label or "",
                                "y_axis_label": chart.data.y_axis_label or "",
                            }
                        )

    @staticmethod
    def _collect_model_versions() -> Dict[str, str]:
        """
        Collect version identifiers for models used in the session.

        Returns:
            Dict mapping model role -> version string
        """
        versions: Dict[str, str] = {}
        try:
            import torch  # type: ignore[import]

            versions["torch"] = torch.__version__
        except ImportError:
            pass
        try:
            from ultralytics import __version__ as yolo_ver  # type: ignore[import]

            versions["yolo"] = yolo_ver
        except ImportError:
            pass
        return versions
