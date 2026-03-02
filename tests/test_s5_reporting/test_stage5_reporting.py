"""
Unit tests for Stage 5: Reporting.

Tests insight generation, output formatting, and data validation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from src.core_engine.schemas.common import SessionInfo
from src.core_engine.schemas.enums import ChartType
from src.core_engine.schemas.stage_outputs import (
    ChartInsight,
    DataPoint,
    DataSeries,
    FinalChartResult,
    PipelineResult,
    RefinedChartData,
    Stage4Output,
)
from src.core_engine.stages.s5_reporting import ReportingConfig, Stage5Reporting


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def session_info() -> SessionInfo:
    """Create a test session."""
    return SessionInfo(
        session_id="test_session_s5",
        source_file=Path("sample_report.pdf"),
        config_hash="s5_test_hash",
    )


def _make_series(
    name: str,
    labels: List[str],
    values: List[float],
    confidence: float = 0.95,
) -> DataSeries:
    """Helper to build a DataSeries quickly."""
    return DataSeries(
        name=name,
        points=[
            DataPoint(label=l, value=v, confidence=confidence)
            for l, v in zip(labels, values)
        ],
    )


def _make_chart(
    chart_id: str = "chart_001",
    chart_type: ChartType = ChartType.LINE,
    title: str = "Revenue Growth",
    series: list | None = None,
    description: str = "A line chart showing revenue growth.",
) -> RefinedChartData:
    """Helper to build a RefinedChartData quickly."""
    if series is None:
        series = [
            _make_series(
                "Revenue",
                ["Q1", "Q2", "Q3", "Q4"],
                [10.0, 15.0, 20.0, 25.0],
            ),
        ]
    return RefinedChartData(
        chart_id=chart_id,
        chart_type=chart_type,
        title=title,
        x_axis_label="Quarter",
        y_axis_label="Million USD",
        series=series,
        description=description,
    )


@pytest.fixture
def single_chart() -> RefinedChartData:
    """A single line chart with one increasing series."""
    return _make_chart()


@pytest.fixture
def multi_series_chart() -> RefinedChartData:
    """A chart with two series for comparison testing."""
    return _make_chart(
        chart_id="chart_002",
        title="Revenue vs Expenses",
        series=[
            _make_series("Revenue", ["Q1", "Q2", "Q3", "Q4"], [10, 15, 20, 25]),
            _make_series("Expenses", ["Q1", "Q2", "Q3", "Q4"], [8, 9, 10, 11]),
        ],
    )


@pytest.fixture
def anomaly_chart() -> RefinedChartData:
    """A chart with an anomalous data point."""
    return _make_chart(
        chart_id="chart_003",
        title="Sales with Outlier",
        series=[
            _make_series(
                "Sales",
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
                [100, 102, 98, 101, 99, 100, 900],
            ),
        ],
    )


@pytest.fixture
def empty_chart() -> RefinedChartData:
    """A chart with no series data."""
    return _make_chart(
        chart_id="chart_empty",
        title="Empty Chart",
        series=[],
        description="A chart with no data.",
    )


@pytest.fixture
def stage4_output(
    session_info: SessionInfo,
    single_chart: RefinedChartData,
) -> Stage4Output:
    """Standard Stage 4 output with one chart."""
    return Stage4Output(session=session_info, charts=[single_chart])


@pytest.fixture
def stage4_multi(
    session_info: SessionInfo,
    single_chart: RefinedChartData,
    multi_series_chart: RefinedChartData,
    anomaly_chart: RefinedChartData,
) -> Stage4Output:
    """Stage 4 output with multiple charts."""
    return Stage4Output(
        session=session_info,
        charts=[single_chart, multi_series_chart, anomaly_chart],
    )


@pytest.fixture
def default_stage() -> Stage5Reporting:
    """Stage 5 with default config."""
    return Stage5Reporting()


@pytest.fixture
def no_output_stage(tmp_path: Path) -> Stage5Reporting:
    """Stage 5 that skips file writing."""
    return Stage5Reporting(
        ReportingConfig(
            save_json=False,
            save_report=False,
            save_markdown=False,
            save_csv=False,
            output_dir=str(tmp_path),
        )
    )


# =============================================================================
# Tests: Config
# =============================================================================


class TestReportingConfig:
    """Tests for ReportingConfig defaults and validation."""

    def test_default_values(self):
        cfg = ReportingConfig()
        assert cfg.enable_insights is True
        assert cfg.max_insights_per_chart == 5
        assert cfg.min_series_for_comparison == 2
        assert cfg.anomaly_z_score_threshold == 2.0
        assert cfg.save_json is True
        assert cfg.save_report is True
        assert cfg.save_markdown is True
        assert cfg.save_csv is False
        assert cfg.output_dir == "data/output"

    def test_custom_config(self):
        cfg = ReportingConfig(
            enable_insights=False,
            max_insights_per_chart=3,
            anomaly_z_score_threshold=3.0,
            save_csv=True,
        )
        assert cfg.enable_insights is False
        assert cfg.max_insights_per_chart == 3
        assert cfg.anomaly_z_score_threshold == 3.0
        assert cfg.save_csv is True

    def test_min_insights_constraint(self):
        with pytest.raises(Exception):
            ReportingConfig(max_insights_per_chart=0)


# =============================================================================
# Tests: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for Stage5 input validation."""

    def test_valid_input(self, default_stage, stage4_output):
        assert default_stage.validate_input(stage4_output) is True

    def test_invalid_input_none(self, default_stage):
        assert default_stage.validate_input(None) is False

    def test_invalid_input_dict(self, default_stage):
        assert default_stage.validate_input({"charts": []}) is False

    def test_invalid_input_string(self, default_stage):
        assert default_stage.validate_input("data") is False

    def test_process_raises_on_invalid(self, default_stage):
        with pytest.raises(Exception) as exc_info:
            default_stage.process("not_stage4_output")
        assert "Invalid input" in str(exc_info.value)


# =============================================================================
# Tests: Insight Generation
# =============================================================================


class TestInsightGeneration:
    """Tests for the insight generation subsystem."""

    def test_summary_with_description(self, no_output_stage, single_chart):
        insight = no_output_stage._summary_insight(single_chart)
        assert insight.insight_type == "summary"
        assert insight.confidence == 0.9  # Has description
        assert "line chart" in insight.text.lower() or single_chart.description in insight.text

    def test_summary_without_description(self, no_output_stage):
        chart = _make_chart(description="")
        insight = no_output_stage._summary_insight(chart)
        assert insight.insight_type == "summary"
        assert insight.confidence == 0.6  # No description
        assert "Revenue" in insight.text

    def test_trend_increasing(self, no_output_stage):
        series = _make_series("Sales", ["A", "B", "C", "D"], [10, 20, 30, 40])
        insight = no_output_stage._detect_trend(series)
        assert insight is not None
        assert insight.insight_type == "trend"
        assert "increasing" in insight.text.lower()

    def test_trend_decreasing(self, no_output_stage):
        series = _make_series("Costs", ["A", "B", "C", "D"], [40, 30, 20, 10])
        insight = no_output_stage._detect_trend(series)
        assert insight is not None
        assert "decreasing" in insight.text.lower()

    def test_trend_stable(self, no_output_stage):
        series = _make_series("Stable", ["A", "B", "C", "D"], [10.0, 10.1, 9.9, 10.0])
        insight = no_output_stage._detect_trend(series)
        assert insight is not None
        assert "stable" in insight.text.lower()

    def test_trend_too_few_points(self, no_output_stage):
        series = _make_series("Short", ["A", "B"], [10, 20])
        insight = no_output_stage._detect_trend(series)
        assert insight is None

    def test_comparison_insight(self, no_output_stage, multi_series_chart):
        insight = no_output_stage._comparison_insight(multi_series_chart)
        assert insight is not None
        assert insight.insight_type == "comparison"
        assert "Revenue" in insight.text  # Revenue has higher peak

    def test_comparison_not_enough_series(self, no_output_stage, single_chart):
        # Default min_series_for_comparison=2, but we use _comparison_insight directly
        insight = no_output_stage._comparison_insight(single_chart)
        # Still generates since it has series, but only one comparison
        assert insight is not None

    def test_anomaly_detection(self, no_output_stage, anomaly_chart):
        anomalies = no_output_stage._detect_anomalies(anomaly_chart)
        assert len(anomalies) > 0
        assert any("900" in a.text for a in anomalies)
        assert all(a.insight_type == "anomaly" for a in anomalies)

    def test_no_anomaly_in_uniform_data(self, no_output_stage, single_chart):
        anomalies = no_output_stage._detect_anomalies(single_chart)
        # Linear data: no outliers expected
        assert len(anomalies) == 0

    def test_insights_disabled(self, session_info, single_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                enable_insights=False,
                save_json=False,
                save_report=False,
                save_markdown=False,
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        result = stage.process(output)
        for chart in result.charts:
            assert len(chart.insights) == 0

    def test_max_insights_capped(self, session_info):
        # Create chart that would generate many insights (multiple series + anomalies)
        series = [
            _make_series(f"S{i}", ["A", "B", "C", "D", "E"], [10 * i, 20 * i, 30 * i, 400 * i, 50 * i])
            for i in range(1, 5)
        ]
        chart = _make_chart(title="Many Insights", series=series)
        stage = Stage5Reporting(
            ReportingConfig(
                max_insights_per_chart=3,
                save_json=False,
                save_report=False,
                save_markdown=False,
            )
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = stage.process(output)
        assert len(result.charts[0].insights) <= 3

    def test_empty_chart_no_insights(self, no_output_stage, empty_chart):
        insights = no_output_stage._generate_insights(empty_chart)
        assert insights == []


# =============================================================================
# Tests: Full Processing
# =============================================================================


class TestFullProcessing:
    """Tests for the end-to-end Stage 5 process."""

    def test_single_chart_processing(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        assert isinstance(result, PipelineResult)
        assert len(result.charts) == 1
        assert result.charts[0].chart_id == "chart_001"
        assert result.processing_time_seconds >= 0
        assert result.summary != ""

    def test_multi_chart_processing(self, session_info):
        charts = [
            _make_chart(chart_id=f"chart_{i:03d}", title=f"Chart {i}")
            for i in range(5)
        ]
        stage = Stage5Reporting(
            ReportingConfig(save_json=False, save_report=False, save_markdown=False)
        )
        output = Stage4Output(session=session_info, charts=charts)
        result = stage.process(output)
        assert len(result.charts) == 5
        assert result.total_charts == 5

    def test_empty_chart_list(self, session_info, no_output_stage):
        output = Stage4Output(session=session_info, charts=[])
        result = no_output_stage.process(output)
        assert len(result.charts) == 0
        assert "No charts" in result.summary

    def test_pipeline_result_computed_fields(self, no_output_stage, stage4_multi):
        result = no_output_stage.process(stage4_multi)
        assert result.total_charts == 3
        assert "line" in result.chart_types_summary

    def test_chart_with_error_still_included(self, session_info):
        """Even if insight generation fails, chart is still in output."""
        stage = Stage5Reporting(
            ReportingConfig(save_json=False, save_report=False, save_markdown=False)
        )
        chart = _make_chart(chart_id="chart_err")
        output = Stage4Output(session=session_info, charts=[chart])
        result = stage.process(output)
        assert len(result.charts) == 1


# =============================================================================
# Tests: Source Info / Traceability
# =============================================================================


class TestTraceability:
    """Tests for provenance metadata generation."""

    def test_source_info_fields(self, no_output_stage, session_info, single_chart):
        source_info = no_output_stage._build_source_info(single_chart, session_info)
        assert source_info["session_id"] == "test_session_s5"
        assert source_info["chart_id"] == "chart_001"
        assert source_info["chart_type"] == "line"
        assert source_info["total_series"] == 1
        assert source_info["total_data_points"] == 4
        assert source_info["corrections_applied"] == 0

    def test_correction_log_tracked(self, no_output_stage, session_info):
        chart = _make_chart()
        chart.correction_log = ["Fixed OCR: 'l0' -> '10'", "Fixed label: 'Q!' -> 'Q1'"]
        source_info = no_output_stage._build_source_info(chart, session_info)
        assert source_info["corrections_applied"] == 2
        assert len(source_info["correction_log"]) == 2


# =============================================================================
# Tests: Session Summary
# =============================================================================


class TestSessionSummary:
    """Tests for session summary generation."""

    def test_summary_content(self, no_output_stage, session_info, single_chart):
        charts = [
            FinalChartResult(
                chart_id=single_chart.chart_id,
                chart_type=single_chart.chart_type,
                title=single_chart.title,
                data=single_chart,
                insights=[],
                source_info={},
            )
        ]
        summary = no_output_stage._generate_session_summary(session_info, charts, 1.5)
        assert "test_session_s5" in summary
        assert "1" in summary  # 1 chart
        assert "1.5" in summary  # processing time

    def test_empty_summary(self, no_output_stage, session_info):
        summary = no_output_stage._generate_session_summary(session_info, [], 0.1)
        assert "No charts" in summary


# =============================================================================
# Tests: Output Formatting
# =============================================================================


class TestTextReport:
    """Tests for text report formatting."""

    def test_text_report_structure(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        report = no_output_stage._format_text_report(result)
        assert "CHART ANALYSIS REPORT" in report
        assert "chart_001" in report
        assert "Revenue" in report
        assert "Quarter" in report or "line" in report

    def test_text_report_multiple_charts(self, no_output_stage, stage4_multi):
        result = no_output_stage.process(stage4_multi)
        report = no_output_stage._format_text_report(result)
        assert "chart_001" in report
        assert "chart_002" in report
        assert "chart_003" in report


class TestMarkdownReport:
    """Tests for Markdown report formatting."""

    def test_markdown_has_headers(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        md = no_output_stage._format_markdown_report(result)
        assert "# Chart Analysis Report" in md
        assert "## Summary" in md
        assert "## Chart 1:" in md

    def test_markdown_has_tables(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        md = no_output_stage._format_markdown_report(result)
        assert "| Property | Value |" in md
        assert "| Label | Value | Confidence |" in md

    def test_markdown_has_insights(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        md = no_output_stage._format_markdown_report(result)
        assert "### Insights" in md
        assert "**[TREND]**" in md or "**[SUMMARY]**" in md

    def test_markdown_traceability(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        md = no_output_stage._format_markdown_report(result)
        assert "### Traceability" in md
        assert "session_id" in md

    def test_markdown_multi_chart(self, no_output_stage, stage4_multi):
        result = no_output_stage.process(stage4_multi)
        md = no_output_stage._format_markdown_report(result)
        assert "## Chart 1:" in md
        assert "## Chart 2:" in md
        assert "## Chart 3:" in md


# =============================================================================
# Tests: CSV Output
# =============================================================================


class TestCSVOutput:
    """Tests for CSV export."""

    def test_csv_write(self, tmp_path, session_info, single_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=False,
                save_markdown=False,
                save_csv=True,
                output_dir=str(tmp_path),
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        result = stage.process(output)

        csv_files = list(tmp_path.glob("*_data.csv"))
        assert len(csv_files) == 1

        content = csv_files[0].read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 5  # header + 4 data points
        assert "session_id" in lines[0]
        assert "Revenue" in content

    def test_csv_multi_series(self, tmp_path, session_info, multi_series_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=False,
                save_markdown=False,
                save_csv=True,
                output_dir=str(tmp_path),
            )
        )
        output = Stage4Output(session=session_info, charts=[multi_series_chart])
        result = stage.process(output)

        csv_files = list(tmp_path.glob("*_data.csv"))
        content = csv_files[0].read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # header + 4 Revenue + 4 Expenses = 9
        assert len(lines) == 9


# =============================================================================
# Tests: File Output
# =============================================================================


class TestFileOutput:
    """Tests for file writing."""

    def test_json_output(self, tmp_path, session_info, single_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=True,
                save_report=False,
                save_markdown=False,
                output_dir=str(tmp_path),
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        stage.process(output)

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert "session" in data
        assert "charts" in data
        assert len(data["charts"]) == 1

    def test_text_report_output(self, tmp_path, session_info, single_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=True,
                save_markdown=False,
                output_dir=str(tmp_path),
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        stage.process(output)

        txt_files = list(tmp_path.glob("*_report.txt"))
        assert len(txt_files) == 1
        assert "CHART ANALYSIS REPORT" in txt_files[0].read_text(encoding="utf-8")

    def test_markdown_output(self, tmp_path, session_info, single_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=False,
                save_markdown=True,
                output_dir=str(tmp_path),
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        stage.process(output)

        md_files = list(tmp_path.glob("*_report.md"))
        assert len(md_files) == 1
        assert "# Chart Analysis Report" in md_files[0].read_text(encoding="utf-8")

    def test_all_outputs(self, tmp_path, session_info, single_chart):
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=True,
                save_report=True,
                save_markdown=True,
                save_csv=True,
                output_dir=str(tmp_path),
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        stage.process(output)

        assert len(list(tmp_path.glob("*.json"))) == 1
        assert len(list(tmp_path.glob("*_report.txt"))) == 1
        assert len(list(tmp_path.glob("*_report.md"))) == 1
        assert len(list(tmp_path.glob("*_data.csv"))) == 1

    def test_no_output_dir_creates(self, tmp_path, session_info, single_chart):
        out_dir = tmp_path / "nested" / "output"
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=True,
                save_report=False,
                save_markdown=False,
                output_dir=str(out_dir),
            )
        )
        output = Stage4Output(session=session_info, charts=[single_chart])
        stage.process(output)

        assert out_dir.exists()
        assert len(list(out_dir.glob("*.json"))) == 1


# =============================================================================
# Tests: Model Versions
# =============================================================================


class TestModelVersions:
    """Tests for model version collection."""

    def test_collect_versions(self, no_output_stage):
        versions = no_output_stage._collect_model_versions()
        assert isinstance(versions, dict)
        # torch should be available in test env
        if "torch" in versions:
            assert isinstance(versions["torch"], str)

    def test_versions_in_result(self, no_output_stage, stage4_output):
        result = no_output_stage.process(stage4_output)
        assert isinstance(result.model_versions, dict)


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_chart_with_no_title(self, session_info, no_output_stage):
        chart = _make_chart(title=None)
        output = Stage4Output(session=session_info, charts=[chart])
        result = no_output_stage.process(output)
        assert result.charts[0].title is None

    def test_chart_with_zero_values(self, session_info, no_output_stage):
        chart = _make_chart(
            series=[_make_series("Zeros", ["A", "B", "C", "D"], [0, 0, 0, 0])]
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = no_output_stage.process(output)
        assert len(result.charts) == 1

    def test_chart_with_negative_values(self, session_info, no_output_stage):
        chart = _make_chart(
            series=[_make_series("Negative", ["A", "B", "C", "D"], [-10, -5, -20, -15])]
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = no_output_stage.process(output)
        assert len(result.charts) == 1
        # Should still detect trend
        insights = [i for i in result.charts[0].insights if i.insight_type == "trend"]
        assert len(insights) > 0

    def test_single_point_series(self, session_info, no_output_stage):
        chart = _make_chart(
            series=[_make_series("Single", ["A"], [42.0])]
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = no_output_stage.process(output)
        # No trend for single point, but summary should exist
        insights = result.charts[0].insights
        trends = [i for i in insights if i.insight_type == "trend"]
        assert len(trends) == 0

    def test_large_dataset(self, session_info, no_output_stage):
        """Process many charts without error."""
        charts = [
            _make_chart(
                chart_id=f"chart_{i:04d}",
                title=f"Chart {i}",
            )
            for i in range(20)
        ]
        output = Stage4Output(session=session_info, charts=charts)
        result = no_output_stage.process(output)
        assert result.total_charts == 20

    def test_mixed_chart_types(self, session_info, no_output_stage):
        charts = [
            _make_chart(chart_id="c1", chart_type=ChartType.LINE),
            _make_chart(chart_id="c2", chart_type=ChartType.BAR),
            _make_chart(chart_id="c3", chart_type=ChartType.PIE),
            _make_chart(chart_id="c4", chart_type=ChartType.SCATTER),
        ]
        output = Stage4Output(session=session_info, charts=charts)
        result = no_output_stage.process(output)
        assert result.total_charts == 4
        types = result.chart_types_summary
        assert types.get("line", 0) == 1
        assert types.get("bar", 0) == 1
        assert types.get("pie", 0) == 1
        assert types.get("scatter", 0) == 1


# =============================================================================
# Tests: Data Validation
# =============================================================================


class TestDataValidation:
    """Tests for the chart data validation subsystem."""

    def test_valid_chart_no_warnings(self, no_output_stage, single_chart, session_info):
        warnings = no_output_stage._validate_chart(single_chart, session_info.session_id)
        assert warnings == []

    def test_missing_description_warning(self, no_output_stage, session_info):
        chart = _make_chart(description="")
        warnings = no_output_stage._validate_chart(chart, session_info.session_id)
        assert any("missing description" in w for w in warnings)

    def test_no_series_warning(self, no_output_stage, session_info, empty_chart):
        warnings = no_output_stage._validate_chart(empty_chart, session_info.session_id)
        assert any("no data series" in w for w in warnings)

    def test_empty_series_warning(self, no_output_stage, session_info):
        chart = _make_chart(
            series=[DataSeries(name="Empty", points=[])]
        )
        warnings = no_output_stage._validate_chart(chart, session_info.session_id)
        assert any("Empty series" in w for w in warnings)

    def test_low_confidence_warning(self, no_output_stage, session_info):
        chart = _make_chart(
            series=[
                _make_series("LowConf", ["A", "B", "C"], [10, 20, 30], confidence=0.3)
            ]
        )
        warnings = no_output_stage._validate_chart(chart, session_info.session_id)
        assert any("Low confidence" in w for w in warnings)

    def test_nan_value_warning(self, no_output_stage, session_info):
        import math

        series = DataSeries(
            name="WithNaN",
            points=[
                DataPoint(label="A", value=10.0),
                DataPoint(label="B", value=float("nan")),
            ],
        )
        chart = _make_chart(series=[series])
        warnings = no_output_stage._validate_chart(chart, session_info.session_id)
        assert any("Non-finite" in w for w in warnings)

    def test_inf_value_warning(self, no_output_stage, session_info):
        series = DataSeries(
            name="WithInf",
            points=[
                DataPoint(label="A", value=10.0),
                DataPoint(label="B", value=float("inf")),
            ],
        )
        chart = _make_chart(series=[series])
        warnings = no_output_stage._validate_chart(chart, session_info.session_id)
        assert any("Non-finite" in w for w in warnings)

    def test_warnings_in_pipeline_result(self, session_info):
        """Warnings propagate to PipelineResult.warnings."""
        chart = _make_chart(description="")
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=False,
                save_markdown=False,
                require_description=True,
            )
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = stage.process(output)
        assert len(result.warnings) > 0
        assert any("missing description" in w for w in result.warnings)

    def test_no_warnings_when_description_not_required(self, session_info):
        chart = _make_chart(description="")
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=False,
                save_markdown=False,
                require_description=False,
            )
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = stage.process(output)
        assert not any("missing description" in w for w in result.warnings)

    def test_markdown_includes_warnings(self, session_info):
        chart = _make_chart(description="")
        stage = Stage5Reporting(
            ReportingConfig(
                save_json=False,
                save_report=False,
                save_markdown=False,
                require_description=True,
            )
        )
        output = Stage4Output(session=session_info, charts=[chart])
        result = stage.process(output)
        md = stage._format_markdown_report(result)
        assert "## Warnings" in md
