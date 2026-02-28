"""
Tests for core_engine.ai.task_types
"""

import pytest

from core_engine.ai.task_types import TaskType


class TestTaskType:
    def test_all_values_are_strings(self) -> None:
        for task in TaskType:
            assert isinstance(task.value, str)

    def test_chart_reasoning_value(self) -> None:
        assert TaskType.CHART_REASONING.value == "chart_reasoning"

    def test_ocr_correction_value(self) -> None:
        assert TaskType.OCR_CORRECTION.value == "ocr_correction"

    def test_description_gen_value(self) -> None:
        assert TaskType.DESCRIPTION_GEN.value == "description_gen"

    def test_data_validation_value(self) -> None:
        assert TaskType.DATA_VALIDATION.value == "data_validation"

    def test_from_string(self) -> None:
        assert TaskType("chart_reasoning") is TaskType.CHART_REASONING
        assert TaskType("ocr_correction") is TaskType.OCR_CORRECTION

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            TaskType("nonexistent_task")

    def test_four_task_types_defined(self) -> None:
        assert len(TaskType) == 4

    def test_str_subclass(self) -> None:
        # TaskType(str, Enum) can be used as plain str
        assert TaskType.CHART_REASONING == "chart_reasoning"
