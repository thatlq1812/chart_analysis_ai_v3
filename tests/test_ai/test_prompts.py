"""
Tests for core_engine.ai.prompts
"""

import json

from core_engine.ai.prompts import (
    CHART_REASONING_SYSTEM,
    DATA_VALIDATION_SYSTEM,
    DESCRIPTION_GEN_SYSTEM,
    OCR_CORRECTION_SYSTEM,
    format_description_user,
    format_ocr_correction_user,
    format_reasoning_user,
)


class TestSystemPrompts:
    def test_chart_reasoning_system_is_string(self) -> None:
        assert isinstance(CHART_REASONING_SYSTEM, str)
        assert len(CHART_REASONING_SYSTEM) > 50

    def test_ocr_correction_system_mentions_json(self) -> None:
        assert "JSON" in OCR_CORRECTION_SYSTEM

    def test_description_gen_system_is_string(self) -> None:
        assert isinstance(DESCRIPTION_GEN_SYSTEM, str)

    def test_data_validation_system_mentions_valid(self) -> None:
        assert "valid" in DATA_VALIDATION_SYSTEM


class TestFormatReasoningUser:
    def test_returns_string(self) -> None:
        result = format_reasoning_user(
            chart_type="bar",
            ocr_texts=[{"text": "Q1", "confidence": 0.9}],
            detected_elements=[{"type": "bar", "value": 10}],
            axis_info={"x_min": 0, "x_max": 100},
        )
        assert isinstance(result, str)

    def test_contains_chart_type(self) -> None:
        result = format_reasoning_user(
            chart_type="line",
            ocr_texts=[],
            detected_elements=[],
            axis_info={},
        )
        assert "line" in result

    def test_contains_valid_json(self) -> None:
        result = format_reasoning_user(
            chart_type="bar",
            ocr_texts=[{"text": "Jan", "confidence": 0.8}],
            detected_elements=[],
            axis_info={"y_min": 0.0, "y_max": 100.0},
        )
        # Extract the JSON portion after the first newline
        json_part = result.split("\n\n", 1)[1]
        parsed = json.loads(json_part)
        assert parsed["chart_type"] == "bar"
        assert parsed["ocr_texts"][0]["text"] == "Jan"

    def test_color_map_optional(self) -> None:
        result_without = format_reasoning_user("scatter", [], [], {})
        result_with = format_reasoning_user(
            "scatter", [], [], {}, color_map={"blue": "Series A"}
        )
        assert "color_map" not in result_without
        assert "color_map" in result_with


class TestFormatOcrCorrectionUser:
    def test_returns_string(self) -> None:
        result = format_ocr_correction_user(
            tokens=["Q1", "1O.5", "Jan"],
            chart_type="bar",
        )
        assert isinstance(result, str)

    def test_contains_tokens(self) -> None:
        result = format_ocr_correction_user(
            tokens=["Q1", "Q2"],
            chart_type="bar",
        )
        assert "Q1" in result
        assert "Q2" in result

    def test_axis_range_optional(self) -> None:
        result_without = format_ocr_correction_user(["val"], "bar")
        result_with = format_ocr_correction_user(
            ["val"], "bar", axis_range={"y_min": 0, "y_max": 100}
        )
        assert "axis_range" not in result_without
        assert "axis_range" in result_with


class TestFormatDescriptionUser:
    def test_returns_string(self) -> None:
        result = format_description_user(
            chart_type="line",
            title="Monthly Trend",
            x_label="Month",
            y_label="Revenue",
            series_summary=[{"name": "2024", "min": 10, "max": 50, "mean": 30}],
        )
        assert isinstance(result, str)

    def test_contains_chart_type(self) -> None:
        result = format_description_user(
            chart_type="pie",
            title=None,
            x_label=None,
            y_label=None,
            series_summary=[],
        )
        assert "pie" in result

    def test_null_title_serialized(self) -> None:
        result = format_description_user(
            chart_type="bar",
            title=None,
            x_label="X",
            y_label="Y",
            series_summary=[],
        )
        assert "null" in result.lower() or '"title": null' in result
