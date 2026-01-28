"""
Unit tests for Stage 4: Prompt Builder

Tests canonical context building and prompt generation.
"""

import json
import pytest

from src.core_engine.schemas.common import BoundingBox, Color, Point
from src.core_engine.schemas.enums import ChartType
from src.core_engine.schemas.stage_outputs import (
    AxisInfo,
    ChartElement,
    DataPoint,
    DataSeries,
    ExtractionConfidence,
    OCRText,
    RawMetadata,
)
from src.core_engine.stages.s4_reasoning import (
    GeminiPromptBuilder,
    PromptConfig,
    PromptTask,
    OutputFormat,
    CanonicalContext,
    create_prompt_builder,
)


class TestCanonicalContext:
    """Test CanonicalContext data structure."""
    
    def test_basic_creation(self):
        """Test creating basic context."""
        context = CanonicalContext(
            chart_id="test_001",
            chart_type="bar",
        )
        
        assert context.chart_id == "test_001"
        assert context.chart_type == "bar"
        assert context.ocr_texts == []
        assert context.elements == []
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = CanonicalContext(
            chart_id="test_001",
            chart_type="bar",
            x_axis={"detected": True, "range": [0, 100]},
            extraction_confidence=0.85,
        )
        
        d = context.to_dict()
        
        assert d["chart_id"] == "test_001"
        assert d["chart_type"] == "bar"
        assert d["axes"]["x"]["detected"] is True
        assert d["quality"]["confidence"] == 0.85


class TestPromptConfig:
    """Test PromptConfig validation."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PromptConfig()
        
        assert config.default_task == PromptTask.FULL_REASONING
        assert config.output_format == OutputFormat.JSON
        assert config.max_ocr_items == 50
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PromptConfig(
            max_ocr_items=30,
            min_ocr_confidence=0.5,
            use_few_shot=True,
        )
        
        assert config.max_ocr_items == 30
        assert config.min_ocr_confidence == 0.5
        assert config.use_few_shot is True


class TestGeminiPromptBuilder:
    """Test GeminiPromptBuilder class."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample RawMetadata for testing."""
        return RawMetadata(
            chart_id="test_chart_001",
            chart_type=ChartType.BAR,
            texts=[
                OCRText(
                    text="Quarterly Revenue",
                    bbox=BoundingBox(x_min=100, y_min=10, x_max=300, y_max=40, confidence=0.9),
                    confidence=0.95,
                    role="title",
                ),
                OCRText(
                    text="Q1",
                    bbox=BoundingBox(x_min=80, y_min=280, x_max=120, y_max=300, confidence=0.9),
                    confidence=0.88,
                    role="xlabel",
                ),
                OCRText(
                    text="loo",
                    bbox=BoundingBox(x_min=10, y_min=200, x_max=40, y_max=220, confidence=0.9),
                    confidence=0.75,
                    role="value",
                ),
                OCRText(
                    text="Revenue ($M)",
                    bbox=BoundingBox(x_min=10, y_min=140, x_max=45, y_max=160, confidence=0.9),
                    confidence=0.90,
                    role="ylabel",
                ),
            ],
            elements=[
                ChartElement(
                    element_type="bar",
                    bbox=BoundingBox(x_min=100, y_min=100, x_max=150, y_max=250, confidence=0.9),
                    center=Point(x=125, y=175),
                    color=Color(r=66, g=133, b=244),
                ),
                ChartElement(
                    element_type="bar",
                    bbox=BoundingBox(x_min=200, y_min=80, x_max=250, y_max=250, confidence=0.9),
                    center=Point(x=225, y=165),
                    color=Color(r=66, g=133, b=244),
                ),
            ],
            axis_info=AxisInfo(
                x_axis_detected=True,
                y_axis_detected=True,
                x_min=0,
                x_max=4,
                y_min=0,
                y_max=500,
                x_calibration_confidence=0.85,
                y_calibration_confidence=0.90,
            ),
            confidence=ExtractionConfidence(
                classification_confidence=0.95,
                ocr_mean_confidence=0.87,
                axis_calibration_confidence=0.88,
                element_detection_confidence=0.85,
                overall_confidence=0.89,
            ),
        )
    
    @pytest.fixture
    def sample_series(self):
        """Create sample DataSeries for testing."""
        return [
            DataSeries(
                name="Revenue",
                color=Color(r=66, g=133, b=244),
                points=[
                    DataPoint(label="Q1", value=125.0, confidence=0.85),
                    DataPoint(label="Q2", value=180.0, confidence=0.88),
                    DataPoint(label="Q3", value=220.0, confidence=0.82),
                    DataPoint(label="Q4", value=310.0, confidence=0.90),
                ],
            ),
        ]
    
    def test_initialization(self):
        """Test builder initialization."""
        builder = GeminiPromptBuilder()
        
        assert len(builder._templates) >= 4
        assert "reasoning" in builder._templates
    
    def test_factory_function(self):
        """Test create_prompt_builder factory."""
        builder = create_prompt_builder()
        
        assert isinstance(builder, GeminiPromptBuilder)
    
    def test_build_canonical_context(self, sample_metadata, sample_series):
        """Test building canonical context."""
        builder = GeminiPromptBuilder()
        
        context = builder.build_canonical_context(
            metadata=sample_metadata,
            mapped_series=sample_series,
            image_width=500,
            image_height=400,
        )
        
        assert isinstance(context, CanonicalContext)
        assert context.chart_id == "test_chart_001"
        assert context.chart_type == "bar"
        assert len(context.ocr_texts) == 4
        assert len(context.estimated_series) == 1
    
    def test_build_reasoning_prompt(self, sample_metadata, sample_series):
        """Test building full reasoning prompt."""
        builder = GeminiPromptBuilder()
        
        prompt = builder.build_reasoning_prompt(
            metadata=sample_metadata,
            mapped_series=sample_series,
        )
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        
        # Check key sections present
        assert "INPUT CONTEXT" in prompt
        assert "TASKS" in prompt
        assert "OUTPUT FORMAT" in prompt
        
        # Check chart info included
        assert "bar" in prompt.lower()
        assert "test_chart_001" in prompt
    
    def test_build_ocr_correction_prompt(self, sample_metadata):
        """Test building OCR correction prompt."""
        builder = GeminiPromptBuilder()
        
        prompt = builder.build_ocr_correction_prompt(
            texts=sample_metadata.texts,
            chart_type=sample_metadata.chart_type,
        )
        
        assert isinstance(prompt, str)
        assert "OCR" in prompt
        assert "loo" in prompt  # The OCR error
        assert "100" in prompt  # Common correction
    
    def test_build_description_prompt(self, sample_series):
        """Test building description prompt."""
        builder = GeminiPromptBuilder()
        
        prompt = builder.build_description_prompt(
            chart_type=ChartType.BAR,
            title="Quarterly Revenue",
            series=sample_series,
            x_label="Quarter",
            y_label="Revenue ($M)",
        )
        
        assert isinstance(prompt, str)
        assert "bar" in prompt.lower()
        assert "Quarterly Revenue" in prompt
    
    def test_build_value_extraction_prompt(self, sample_metadata, sample_series):
        """Test building value extraction prompt."""
        builder = GeminiPromptBuilder()
        
        context = builder.build_canonical_context(
            sample_metadata, sample_series
        )
        
        prompt = builder.build_value_extraction_prompt(context)
        
        assert isinstance(prompt, str)
        assert "VALUE" in prompt
        assert "JSON" in prompt
    
    def test_build_trend_analysis_prompt(self, sample_series):
        """Test building trend analysis prompt."""
        builder = GeminiPromptBuilder()
        
        prompt = builder.build_trend_analysis_prompt(
            series=sample_series,
            chart_type=ChartType.LINE,
        )
        
        assert isinstance(prompt, str)
        assert "trend" in prompt.lower()
    
    def test_low_confidence_texts_filtered(self, sample_metadata):
        """Test that low confidence texts can be filtered."""
        config = PromptConfig(min_ocr_confidence=0.8)
        builder = GeminiPromptBuilder(config)
        
        context = builder.build_canonical_context(sample_metadata)
        
        # Only texts with conf >= 0.8 should be included
        # From fixture: 0.95, 0.88, 0.75, 0.90 -> 0.75 filtered
        assert len(context.ocr_texts) == 3
    
    def test_max_items_limit(self):
        """Test that max items are respected."""
        config = PromptConfig(max_ocr_items=5)  # Min allowed is 5
        builder = GeminiPromptBuilder(config)
        
        # Create metadata with many texts
        texts = [
            OCRText(
                text=f"Text {i}",
                bbox=BoundingBox(x_min=0, y_min=i*10, x_max=50, y_max=i*10+10, confidence=0.9),
                confidence=0.9,
            )
            for i in range(10)
        ]
        
        metadata = RawMetadata(
            chart_id="test",
            chart_type=ChartType.BAR,
            texts=texts,
        )
        
        context = builder.build_canonical_context(metadata)
        
        assert len(context.ocr_texts) == 5
    
    def test_constraints_included_by_default(self, sample_metadata):
        """Test that constraints are included by default."""
        builder = GeminiPromptBuilder()
        
        prompt = builder.build_reasoning_prompt(sample_metadata)
        
        assert "CONSTRAINTS" in prompt
        assert "ONLY" in prompt  # Anti-hallucination
    
    def test_constraints_can_be_disabled(self, sample_metadata):
        """Test that constraints can be disabled."""
        config = PromptConfig(include_constraints=False)
        builder = GeminiPromptBuilder(config)
        
        prompt = builder.build_reasoning_prompt(sample_metadata)
        
        # Should not have the explicit constraints section
        # (basic rules in output format remain)
        assert "CONSTRAINTS (CRITICAL)" not in prompt
    
    def test_prompt_includes_warnings(self):
        """Test that warnings are included in prompt."""
        metadata = RawMetadata(
            chart_id="test",
            chart_type=ChartType.BAR,
            warnings=["Low OCR confidence", "Axis calibration uncertain"],
        )
        
        builder = GeminiPromptBuilder()
        context = builder.build_canonical_context(metadata)
        
        assert len(context.warnings) == 2
    
    def test_axis_labels_extracted(self, sample_metadata):
        """Test that axis labels are correctly identified."""
        builder = GeminiPromptBuilder()
        
        # Use internal method
        x_label = builder._find_axis_label(sample_metadata.texts, "xlabel")
        y_label = builder._find_axis_label(sample_metadata.texts, "ylabel")
        
        assert x_label == "Q1"
        assert y_label == "Revenue ($M)"


class TestPromptTask:
    """Test PromptTask enum."""
    
    def test_all_tasks_defined(self):
        """Test all expected tasks are defined."""
        tasks = [
            PromptTask.FULL_REASONING,
            PromptTask.OCR_CORRECTION,
            PromptTask.VALUE_EXTRACTION,
            PromptTask.DESCRIPTION_ONLY,
            PromptTask.LEGEND_MAPPING,
            PromptTask.TREND_ANALYSIS,
        ]
        
        assert len(tasks) == 6


class TestOutputFormat:
    """Test OutputFormat enum."""
    
    def test_all_formats_defined(self):
        """Test all expected formats are defined."""
        formats = [
            OutputFormat.JSON,
            OutputFormat.TEXT,
            OutputFormat.MARKDOWN,
        ]
        
        assert len(formats) == 3
