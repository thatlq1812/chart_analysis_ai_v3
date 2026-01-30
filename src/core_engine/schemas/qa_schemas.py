"""
Chart QA Schema v2 - Enhanced Reasoning Structures

This module defines the schema for Chart Question-Answering with:
- Reasoning depth (multi-step, interpolation, inference)
- Visual grounding (coordinates, regions, references)
- Confidence and uncertainty handling
- Chart-type specific templates

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-01-28 | That Le | Research-grade QA schema |
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class QuestionType(str, Enum):
    """Classification of question types by cognitive demand."""
    
    # Basic (shallow) - OCR level
    STRUCTURAL = "structural"      # "What is the title?", "What are the axis labels?"
    EXTRACTION = "extraction"      # "What is the value at X?"
    COUNTING = "counting"          # "How many bars?", "How many data points?"
    
    # Intermediate - Requires visual understanding
    COMPARISON = "comparison"      # "Which is higher?", "Compare A vs B"
    TREND = "trend"                # "Is it increasing/decreasing?"
    RANGE = "range"                # "What is the min/max?", "What is the range?"
    
    # Advanced (deep) - Requires reasoning
    INTERPOLATION = "interpolation"    # "What is the value at X=0.35?"
    EXTRAPOLATION = "extrapolation"    # "What would be the value at X=1.5?"
    PERCENTAGE_CHANGE = "percentage_change"  # "How much did it change from A to B?"
    THRESHOLD = "threshold"            # "At what point does Y drop below 500?"
    OPTIMAL_POINT = "optimal_point"    # "Where is the trade-off optimal?"
    MULTI_HOP = "multi_hop"            # Requires combining multiple facts
    
    # Conceptual - Requires domain knowledge
    WHY_REASONING = "why_reasoning"    # "Why does the trend change?"
    CAPTION_AWARE = "caption_aware"    # Questions linking caption to chart
    AMBIGUITY = "ambiguity"            # "Cannot be determined from chart"


class ReasoningMethod(str, Enum):
    """Method used to arrive at the answer."""
    
    DIRECT_READ = "direct_read"         # Read directly from chart
    INTERPOLATION = "interpolation"     # Estimate between known points
    EXTRAPOLATION = "extrapolation"     # Estimate beyond known points
    CALCULATION = "calculation"         # Arithmetic on extracted values
    APPROXIMATION = "approximation"     # Visual estimation
    INFERENCE = "inference"             # Logical deduction
    COMPARISON = "comparison"           # Compare multiple elements
    AGGREGATION = "aggregation"         # Combine multiple data points
    PATTERN_RECOGNITION = "pattern"     # Identify visual patterns
    CANNOT_DETERMINE = "cannot_determine"  # Insufficient information


class ChartRegion(str, Enum):
    """Regions within a chart for visual grounding."""
    
    TITLE = "title"
    X_AXIS = "x_axis"
    Y_AXIS = "y_axis"
    LEGEND = "legend"
    PLOT_AREA = "plot_area"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    ENTIRE_CHART = "entire"


class ConfidenceLevel(str, Enum):
    """Confidence levels for answers."""
    
    HIGH = "high"        # > 0.9 - Direct read, clear values
    MEDIUM = "medium"    # 0.7-0.9 - Some estimation needed
    LOW = "low"          # 0.5-0.7 - Significant approximation
    UNCERTAIN = "uncertain"  # < 0.5 - Cannot reliably determine


# =============================================================================
# VISUAL GROUNDING
# =============================================================================

class PointReference(BaseModel):
    """Reference to a specific point on the chart."""
    
    x_value: Optional[Union[float, str]] = Field(
        None, description="X-axis value or label"
    )
    y_value: Optional[Union[float, str]] = Field(
        None, description="Y-axis value"
    )
    x_pixel: Optional[int] = Field(
        None, description="X pixel coordinate (optional)"
    )
    y_pixel: Optional[int] = Field(
        None, description="Y pixel coordinate (optional)"
    )


class RegionReference(BaseModel):
    """Reference to a region on the chart."""
    
    region: ChartRegion = Field(..., description="Named region")
    description: Optional[str] = Field(
        None, description="Human-readable description"
    )


class VisualGrounding(BaseModel):
    """Visual grounding information for an answer."""
    
    chart_type: str = Field(..., description="Type of chart")
    regions_referenced: List[ChartRegion] = Field(
        default_factory=list, 
        description="Chart regions used to answer"
    )
    points_referenced: List[PointReference] = Field(
        default_factory=list,
        description="Specific data points referenced"
    )
    series_referenced: List[str] = Field(
        default_factory=list,
        description="Data series names referenced"
    )
    tick_marks_used: List[Union[float, str]] = Field(
        default_factory=list,
        description="Axis tick marks used for reading"
    )


# =============================================================================
# REASONING STRUCTURE
# =============================================================================

class ReasoningStep(BaseModel):
    """A single step in the reasoning chain."""
    
    step_number: int = Field(..., ge=1)
    action: str = Field(..., description="What was done in this step")
    observation: str = Field(..., description="What was observed/extracted")
    intermediate_result: Optional[str] = Field(
        None, description="Intermediate value if any"
    )


class InferenceInfo(BaseModel):
    """Information about the inference process."""
    
    method: ReasoningMethod = Field(..., description="Method used")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score 0-1"
    )
    confidence_level: ConfidenceLevel = Field(..., description="Categorical confidence")
    reasoning_steps: List[ReasoningStep] = Field(
        default_factory=list,
        description="Chain of reasoning steps"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during reasoning"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations of this answer"
    )


# =============================================================================
# QA PAIR v2
# =============================================================================

class QAPairV2(BaseModel):
    """Enhanced QA pair with reasoning depth and visual grounding."""
    
    # Core QA
    question: str = Field(..., description="The question asked")
    answer: str = Field(..., description="Human-readable answer")
    
    # Classification
    question_type: QuestionType = Field(..., description="Type of question")
    difficulty: int = Field(
        ..., ge=1, le=5, 
        description="Difficulty 1-5 (1=basic OCR, 5=complex reasoning)"
    )
    
    # Structured answer (for training)
    answer_value: Optional[Union[float, str, List[Any]]] = Field(
        None, description="Extracted answer value (for evaluation)"
    )
    answer_unit: Optional[str] = Field(
        None, description="Unit of measurement if applicable"
    )
    
    # Visual grounding
    visual_grounding: Optional[VisualGrounding] = Field(
        None, description="Visual evidence for the answer"
    )
    
    # Reasoning
    inference: Optional[InferenceInfo] = Field(
        None, description="How the answer was derived"
    )
    
    # Metadata
    requires_caption: bool = Field(
        False, description="Whether question requires caption context"
    )
    is_answerable: bool = Field(
        True, description="Whether question can be answered from chart alone"
    )
    unanswerable_reason: Optional[str] = Field(
        None, description="Why question cannot be answered if is_answerable=False"
    )


# =============================================================================
# CHART QA SAMPLE v2
# =============================================================================

class ChartVerification(BaseModel):
    """Verification info from Gemini about chart validity and type."""
    
    is_valid_chart: bool = Field(True, description="Is this a valid chart/graph?")
    actual_chart_type: str = Field(..., description="Gemini's detected chart type")
    chart_quality: str = Field("medium", description="high|medium|low|unreadable")
    verification_notes: Optional[str] = Field(None, description="Notes on type mismatch or quality issues")
    type_matches_folder: bool = Field(True, description="Does Gemini type match folder type?")


class ChartQASampleV2(BaseModel):
    """Complete QA sample for a chart image."""
    
    # Identity
    image_id: str = Field(..., description="Unique image identifier")
    image_path: str = Field(..., description="Relative path to image")
    chart_type: str = Field(..., description="Folder/labeled chart type")
    
    # Gemini verification (for data cleaning)
    verification: Optional[ChartVerification] = Field(
        None, description="Gemini's verification of chart type and quality"
    )
    
    # Context
    caption: Optional[str] = Field(None, description="Figure caption if available")
    context_text: Optional[str] = Field(None, description="Surrounding text context")
    paper_id: Optional[str] = Field(None, description="Source paper ID")
    
    # QA pairs
    qa_pairs: List[QAPairV2] = Field(
        default_factory=list, description="List of QA pairs"
    )
    
    # Statistics
    qa_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of questions by type"
    )
    avg_difficulty: float = Field(
        0.0, description="Average difficulty of questions"
    )
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    generator_version: str = Field("2.0.0", description="QA generator version")
    generator_model: str = Field("", description="Model used for generation")


# =============================================================================
# QUESTION TEMPLATES BY CHART TYPE
# =============================================================================

# Templates for generating diverse, deep questions
QUESTION_TEMPLATES = {
    "line": {
        QuestionType.STRUCTURAL: [
            "What is the label for the {axis}-axis?",
            "What is the title of this chart?",
            "How many data series are shown?",
        ],
        QuestionType.EXTRACTION: [
            "What is the value when {x_var} = {x_val}?",
            "What is the {y_var} at the {position} point?",
        ],
        QuestionType.TREND: [
            "What is the overall trend of the data?",
            "Is the {series} increasing or decreasing between {x1} and {x2}?",
            "Where does the trend change direction?",
        ],
        QuestionType.INTERPOLATION: [
            "Estimate the value when {x_var} = {x_val} (between tick marks)",
            "What would be the approximate {y_var} at {x_val}?",
        ],
        QuestionType.PERCENTAGE_CHANGE: [
            "What is the percentage change in {y_var} from {x1} to {x2}?",
            "By what factor does {y_var} decrease from {x1} to {x2}?",
        ],
        QuestionType.THRESHOLD: [
            "At what {x_var} does {y_var} first drop below {threshold}?",
            "Where does the curve cross the {value} line?",
        ],
        QuestionType.OPTIMAL_POINT: [
            "At what point does the rate of change become negligible?",
            "What appears to be the optimal trade-off point?",
            "Where is the 'elbow' or knee point in this curve?",
        ],
        QuestionType.WHY_REASONING: [
            "Why does the {y_var} decrease as {x_var} increases?",
            "What could explain the {pattern} observed in the data?",
        ],
    },
    "bar": {
        QuestionType.STRUCTURAL: [
            "What categories are shown on the {axis}-axis?",
            "How many bars are displayed?",
            "What does each bar color represent?",
        ],
        QuestionType.EXTRACTION: [
            "What is the value for '{category}'?",
            "What is the height of the {position} bar?",
        ],
        QuestionType.COMPARISON: [
            "Which category has the highest value?",
            "How much larger is '{cat1}' compared to '{cat2}'?",
            "Rank the categories from highest to lowest.",
        ],
        QuestionType.PERCENTAGE_CHANGE: [
            "What percentage of the total does '{category}' represent?",
            "How much higher is '{cat1}' than '{cat2}' in percentage terms?",
        ],
        QuestionType.MULTI_HOP: [
            "What is the sum of the top 3 categories?",
            "What is the average value across all categories?",
        ],
    },
    "scatter": {
        QuestionType.TREND: [
            "Is there a positive or negative correlation?",
            "What is the general relationship between {x_var} and {y_var}?",
        ],
        QuestionType.EXTRACTION: [
            "How many data points are in the {region} region?",
            "What is the approximate {y_var} when {x_var} = {x_val}?",
        ],
        QuestionType.COMPARISON: [
            "Which cluster has more data points?",
            "Are there any outliers visible?",
        ],
        QuestionType.INTERPOLATION: [
            "Based on the trend, estimate {y_var} when {x_var} = {x_val}",
        ],
    },
    "pie": {
        QuestionType.EXTRACTION: [
            "What percentage does '{category}' represent?",
            "What is the largest segment?",
        ],
        QuestionType.COMPARISON: [
            "Is '{cat1}' larger than '{cat2}'?",
            "How many times larger is the biggest segment than the smallest?",
        ],
        QuestionType.MULTI_HOP: [
            "What is the combined percentage of '{cat1}' and '{cat2}'?",
            "What percentage do the top 2 categories represent together?",
        ],
    },
    "heatmap": {
        QuestionType.EXTRACTION: [
            "What is the value at row '{row}', column '{col}'?",
            "What is the maximum value in the heatmap?",
        ],
        QuestionType.COMPARISON: [
            "Which cell has the highest/lowest intensity?",
            "Compare the values in row '{row1}' vs row '{row2}'",
        ],
        QuestionType.TREND: [
            "Is there a pattern along the diagonal?",
            "Which row/column shows the most variation?",
        ],
    },
}


# =============================================================================
# DIFFICULTY MAPPING
# =============================================================================

QUESTION_DIFFICULTY = {
    QuestionType.STRUCTURAL: 1,
    QuestionType.EXTRACTION: 2,
    QuestionType.COUNTING: 1,
    QuestionType.COMPARISON: 2,
    QuestionType.TREND: 2,
    QuestionType.RANGE: 2,
    QuestionType.INTERPOLATION: 3,
    QuestionType.EXTRAPOLATION: 4,
    QuestionType.PERCENTAGE_CHANGE: 3,
    QuestionType.THRESHOLD: 4,
    QuestionType.OPTIMAL_POINT: 5,
    QuestionType.MULTI_HOP: 4,
    QuestionType.WHY_REASONING: 5,
    QuestionType.CAPTION_AWARE: 4,
    QuestionType.AMBIGUITY: 3,
}


# =============================================================================
# SHARD SCHEMA v2
# =============================================================================

class ShardMetadataV2(BaseModel):
    """Metadata for a QA shard file."""
    
    schema_version: str = Field("2.0.0")
    chart_type: str
    shard_index: int
    sample_count: int
    
    # QA statistics
    total_qa_pairs: int = 0
    qa_type_distribution: Dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: Dict[int, int] = Field(default_factory=dict)
    avg_difficulty: float = 0.0
    
    # Quality metrics
    answerable_ratio: float = 1.0
    avg_confidence: float = 0.0


class ShardV2(BaseModel):
    """A shard file containing QA samples."""
    
    metadata: ShardMetadataV2
    samples: List[ChartQASampleV2]
