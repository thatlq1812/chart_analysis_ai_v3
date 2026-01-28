"""
Prompt Builder for Gemini API

Constructs structured prompts in Canonical Format for optimal
LLM reasoning on chart data.

Key features:
- Canonical Format: Structured context sections
- Anti-hallucination: Explicit constraints and grounding
- Multi-task prompts: OCR correction, value extraction, description
- Template-based: Jinja2-style variable substitution
- Token-efficient: Compact but complete context

Reference: docs/architecture/STAGE4_REASONING.md
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ...schemas.common import Color
from ...schemas.enums import ChartType
from ...schemas.stage_outputs import (
    AxisInfo,
    ChartElement,
    DataPoint,
    DataSeries,
    OCRText,
    RawMetadata,
)

logger = logging.getLogger(__name__)


class PromptTask(str, Enum):
    """Types of reasoning tasks."""
    FULL_REASONING = "full_reasoning"
    OCR_CORRECTION = "ocr_correction"
    VALUE_EXTRACTION = "value_extraction"
    DESCRIPTION_ONLY = "description_only"
    LEGEND_MAPPING = "legend_mapping"
    TREND_ANALYSIS = "trend_analysis"


class OutputFormat(str, Enum):
    """Expected output formats."""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"


class PromptConfig(BaseModel):
    """Configuration for prompt building."""
    
    # Task settings
    default_task: PromptTask = Field(
        default=PromptTask.FULL_REASONING,
        description="Default reasoning task"
    )
    
    # Output format
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Expected output format"
    )
    
    # Context limits
    max_ocr_items: int = Field(
        default=50,
        ge=5,
        description="Maximum OCR texts to include"
    )
    max_elements: int = Field(
        default=100,
        ge=10,
        description="Maximum elements to include"
    )
    max_series_points: int = Field(
        default=20,
        ge=5,
        description="Maximum points per series to include"
    )
    
    # Confidence thresholds
    min_ocr_confidence: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Minimum OCR confidence to include"
    )
    
    # Template settings
    use_few_shot: bool = Field(
        default=False,
        description="Include few-shot examples in prompt"
    )
    include_constraints: bool = Field(
        default=True,
        description="Include anti-hallucination constraints"
    )
    
    # Language
    output_language: str = Field(
        default="english",
        description="Output language for descriptions"
    )


@dataclass
class CanonicalContext:
    """
    Canonical Format context structure.
    
    This is the structured representation that gets serialized
    into the prompt. Each field maps to a specific section.
    """
    
    # Chart metadata
    chart_id: str
    chart_type: str
    image_dimensions: Optional[tuple] = None
    
    # Axis information
    x_axis: Dict[str, Any] = None
    y_axis: Dict[str, Any] = None
    
    # OCR data
    ocr_texts: List[Dict[str, Any]] = None
    
    # Geometric data
    elements: List[Dict[str, Any]] = None
    estimated_series: List[Dict[str, Any]] = None
    
    # Quality metrics
    extraction_confidence: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        self.x_axis = self.x_axis or {}
        self.y_axis = self.y_axis or {}
        self.ocr_texts = self.ocr_texts or []
        self.elements = self.elements or []
        self.estimated_series = self.estimated_series or []
        self.warnings = self.warnings or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chart_id": self.chart_id,
            "chart_type": self.chart_type,
            "image_dimensions": self.image_dimensions,
            "axes": {
                "x": self.x_axis,
                "y": self.y_axis,
            },
            "ocr_texts": self.ocr_texts,
            "elements": self.elements,
            "estimated_series": self.estimated_series,
            "quality": {
                "confidence": self.extraction_confidence,
                "warnings": self.warnings,
            },
        }


class GeminiPromptBuilder:
    """
    Builds structured prompts for Gemini API.
    
    Uses Canonical Format for clear, grounded context that
    minimizes hallucination risk.
    
    Example:
        builder = GeminiPromptBuilder()
        prompt = builder.build_reasoning_prompt(metadata, mapped_series)
    """
    
    # System instruction for Gemini
    SYSTEM_INSTRUCTION = """You are a senior data analyst specializing in chart analysis.
Your task is to analyze extracted chart data and provide accurate, grounded insights.

CRITICAL RULES:
1. ONLY use information provided in the INPUT CONTEXT
2. If data is uncertain, acknowledge the uncertainty
3. Do NOT invent values not present in the data
4. Flag potential OCR errors but preserve original if unsure
5. Output MUST be valid JSON when requested"""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize prompt builder.
        
        Args:
            config: Prompt configuration
        """
        self.config = config or PromptConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load templates
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from files or use defaults."""
        templates = {}
        
        # Default templates
        templates["reasoning"] = self._get_reasoning_template()
        templates["ocr_correction"] = self._get_ocr_correction_template()
        templates["description"] = self._get_description_template()
        templates["value_extraction"] = self._get_value_extraction_template()
        
        # Try loading from files
        prompts_dir = Path(__file__).parent / "prompts"
        for name in ["reasoning", "ocr_correction", "description", "value_extraction"]:
            template_path = prompts_dir / f"{name}.txt"
            if template_path.exists():
                try:
                    templates[name] = template_path.read_text(encoding="utf-8")
                    self.logger.debug(f"Loaded template: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load template {name}: {e}")
        
        return templates
    
    def build_canonical_context(
        self,
        metadata: RawMetadata,
        mapped_series: Optional[List[DataSeries]] = None,
        image_width: int = 0,
        image_height: int = 0,
    ) -> CanonicalContext:
        """
        Build canonical context from Stage 3 metadata.
        
        Args:
            metadata: Raw metadata from Stage 3
            mapped_series: Optional pre-mapped series from ValueMapper
            image_width: Image width
            image_height: Image height
        
        Returns:
            CanonicalContext ready for prompt building
        """
        # Process OCR texts
        ocr_texts = []
        for text in metadata.texts[:self.config.max_ocr_items]:
            if text.confidence >= self.config.min_ocr_confidence:
                ocr_texts.append({
                    "text": text.text,
                    "role": text.role or "unknown",
                    "confidence": round(text.confidence, 2),
                    "position": {
                        "x": text.bbox.center[0],
                        "y": text.bbox.center[1],
                    },
                })
        
        # Process axis info
        x_axis = {}
        y_axis = {}
        if metadata.axis_info:
            ai = metadata.axis_info
            if ai.x_axis_detected:
                x_axis = {
                    "detected": True,
                    "range": [ai.x_min, ai.x_max],
                    "label": self._find_axis_label(metadata.texts, "xlabel"),
                    "confidence": ai.x_calibration_confidence,
                }
            if ai.y_axis_detected:
                y_axis = {
                    "detected": True,
                    "range": [ai.y_min, ai.y_max],
                    "label": self._find_axis_label(metadata.texts, "ylabel"),
                    "confidence": ai.y_calibration_confidence,
                }
        
        # Process elements
        elements = []
        for elem in metadata.elements[:self.config.max_elements]:
            elem_dict = {
                "type": elem.element_type,
                "center": {"x": elem.center.x, "y": elem.center.y},
            }
            if elem.color:
                elem_dict["color"] = f"RGB({elem.color.r},{elem.color.g},{elem.color.b})"
            if elem.area_pixels:
                elem_dict["area"] = elem.area_pixels
            elements.append(elem_dict)
        
        # Process mapped series
        estimated_series = []
        if mapped_series:
            for series in mapped_series:
                series_dict = {
                    "name": series.name,
                    "points": [
                        {"label": p.label, "value": p.value, "confidence": p.confidence}
                        for p in series.points[:self.config.max_series_points]
                    ],
                }
                if series.color:
                    series_dict["color"] = f"RGB({series.color.r},{series.color.g},{series.color.b})"
                estimated_series.append(series_dict)
        
        # Extraction confidence
        conf = 0.5
        if metadata.confidence:
            conf = metadata.confidence.overall_confidence
        
        return CanonicalContext(
            chart_id=metadata.chart_id,
            chart_type=metadata.chart_type.value,
            image_dimensions=(image_width, image_height) if image_width > 0 else None,
            x_axis=x_axis,
            y_axis=y_axis,
            ocr_texts=ocr_texts,
            elements=elements,
            estimated_series=estimated_series,
            extraction_confidence=conf,
            warnings=metadata.warnings,
        )
    
    def build_reasoning_prompt(
        self,
        metadata: RawMetadata,
        mapped_series: Optional[List[DataSeries]] = None,
        image_width: int = 0,
        image_height: int = 0,
    ) -> str:
        """
        Build full reasoning prompt.
        
        This is the main entry point for comprehensive chart analysis.
        
        Args:
            metadata: Raw metadata from Stage 3
            mapped_series: Optional pre-mapped series
            image_width: Image width
            image_height: Image height
        
        Returns:
            Formatted prompt string
        """
        context = self.build_canonical_context(
            metadata, mapped_series, image_width, image_height
        )
        
        # Build context section
        context_str = self._format_context_section(context)
        
        # Build task section
        task_str = self._format_task_section(PromptTask.FULL_REASONING)
        
        # Build output format section
        output_str = self._format_output_section()
        
        # Build constraints section
        constraints_str = ""
        if self.config.include_constraints:
            constraints_str = self._format_constraints_section(context)
        
        # Combine sections
        prompt = f"""## INPUT CONTEXT

{context_str}

## TASKS

{task_str}

## OUTPUT FORMAT

{output_str}
{constraints_str}"""
        
        return prompt
    
    def build_ocr_correction_prompt(
        self,
        texts: List[OCRText],
        chart_type: ChartType,
    ) -> str:
        """
        Build prompt for OCR error correction.
        
        Args:
            texts: OCR text results
            chart_type: Chart type for context
        
        Returns:
            Formatted prompt string
        """
        # Format text list
        text_items = []
        for i, text in enumerate(texts[:self.config.max_ocr_items]):
            conf_str = f"[conf: {text.confidence:.0%}]"
            role_str = f"({text.role})" if text.role else ""
            text_items.append(f'{i+1}. "{text.text}" {role_str} {conf_str}')
        
        texts_section = "\n".join(text_items)
        
        prompt = f"""## OCR CORRECTION TASK

Chart Type: {chart_type.value}

### Detected Texts:
{texts_section}

### Common OCR Errors to Check:
- "loo", "l00", "1oo" → "100"
- "O" → "0" in numeric context
- "l", "I" → "1" in numeric context  
- "S" → "5" in numeric context
- "B" → "8" in numeric context
- Missing "%" after percentage values
- Decimal confusion: "1.0" vs "10"

### Task:
Identify and correct OCR errors. Only include items that need correction.

### Output (JSON):
```json
{{
    "corrections": [
        {{"index": 1, "original": "...", "corrected": "...", "reason": "..."}}
    ],
    "axis_labels": {{
        "x": "detected x-axis label or null",
        "y": "detected y-axis label or null"
    }},
    "title": "detected title or null"
}}
```

Return ONLY valid JSON."""
        
        return prompt
    
    def build_value_extraction_prompt(
        self,
        context: CanonicalContext,
    ) -> str:
        """
        Build prompt for value extraction refinement.
        
        Args:
            context: Canonical context with estimated values
        
        Returns:
            Formatted prompt string
        """
        # Format estimated series
        series_str = json.dumps(context.estimated_series, indent=2)
        
        # Format axis info
        axis_str = f"""X-Axis: {json.dumps(context.x_axis)}
Y-Axis: {json.dumps(context.y_axis)}"""
        
        prompt = f"""## VALUE EXTRACTION REFINEMENT

Chart Type: {context.chart_type}
Extraction Confidence: {context.extraction_confidence:.0%}

### Axis Information:
{axis_str}

### Estimated Data Series:
{series_str}

### Task:
1. Verify value consistency with axis ranges
2. Identify units (%, USD, Years, etc.)
3. Flag any suspicious values
4. Suggest corrections if values seem wrong

### Output (JSON):
```json
{{
    "units": {{
        "x": "unit or null",
        "y": "unit or null"
    }},
    "refined_series": [
        {{
            "name": "series name",
            "points": [
                {{"label": "...", "value": 123.4}}
            ]
        }}
    ],
    "flags": [
        {{"index": 0, "issue": "value outside axis range"}}
    ]
}}
```

Return ONLY valid JSON."""
        
        return prompt
    
    def build_description_prompt(
        self,
        chart_type: ChartType,
        title: Optional[str],
        series: List[DataSeries],
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> str:
        """
        Build prompt for description generation.
        
        Args:
            chart_type: Chart type
            title: Chart title
            series: Data series
            x_label: X-axis label
            y_label: Y-axis label
        
        Returns:
            Formatted prompt string
        """
        # Format series summary
        series_summary = []
        for s in series:
            if s.points:
                values = [p.value for p in s.points[:10]]
                min_val = min(values) if values else 0
                max_val = max(values) if values else 0
                avg_val = sum(values) / len(values) if values else 0
                series_summary.append(
                    f"- {s.name}: {len(s.points)} points, "
                    f"range [{min_val:.1f}, {max_val:.1f}], avg {avg_val:.1f}"
                )
        
        series_str = "\n".join(series_summary) if series_summary else "No data"
        
        prompt = f"""## DESCRIPTION GENERATION

### Chart Information:
- Type: {chart_type.value}
- Title: {title or 'Unknown'}
- X-axis: {x_label or 'Unknown'}
- Y-axis: {y_label or 'Unknown'}

### Data Summary:
{series_str}

### Task:
Generate a 2-3 sentence academic-style description that:
1. States what the chart shows
2. Describes key patterns or trends
3. Notes significant values (max, min, outliers)

Use formal language. Be factual and precise.
Base description ONLY on provided data.

### Output:
Return ONLY the description text (no JSON)."""
        
        return prompt
    
    def build_trend_analysis_prompt(
        self,
        series: List[DataSeries],
        chart_type: ChartType,
    ) -> str:
        """
        Build prompt for trend analysis.
        
        Args:
            series: Data series
            chart_type: Chart type
        
        Returns:
            Formatted prompt string
        """
        # Format data for analysis
        data_str = ""
        for s in series:
            points_str = ", ".join(
                f"({p.label}: {p.value})"
                for p in s.points[:15]
            )
            data_str += f"\n{s.name}: {points_str}"
        
        prompt = f"""## TREND ANALYSIS

Chart Type: {chart_type.value}

### Data:
{data_str}

### Task:
Analyze the data and identify:
1. Overall trend (increasing, decreasing, stable, cyclical)
2. Rate of change (gradual, steep, exponential)
3. Notable patterns (peaks, valleys, plateaus)
4. Comparisons between series (if multiple)

### Output (JSON):
```json
{{
    "overall_trend": "increasing|decreasing|stable|cyclical|mixed",
    "trend_strength": "strong|moderate|weak",
    "patterns": [
        {{"type": "peak", "at": "label", "value": 123}}
    ],
    "comparison": "description of series comparison or null",
    "summary": "one sentence summary"
}}
```

Return ONLY valid JSON."""
        
        return prompt
    
    # =========================================================================
    # Private Methods: Formatting
    # =========================================================================
    
    def _format_context_section(self, context: CanonicalContext) -> str:
        """Format the context section of the prompt."""
        sections = []
        
        # Chart info
        sections.append(f"### Chart Information\n"
                       f"- ID: {context.chart_id}\n"
                       f"- Type: {context.chart_type}\n"
                       f"- Extraction Confidence: {context.extraction_confidence:.0%}")
        
        # Axis info
        if context.x_axis or context.y_axis:
            axis_lines = ["### Axis Context"]
            if context.x_axis.get("detected"):
                x_range = context.x_axis.get("range", [None, None])
                x_label = context.x_axis.get("label", "Unknown")
                axis_lines.append(
                    f"- X-Axis: Label=\"{x_label}\", Range={x_range}"
                )
            if context.y_axis.get("detected"):
                y_range = context.y_axis.get("range", [None, None])
                y_label = context.y_axis.get("label", "Unknown")
                axis_lines.append(
                    f"- Y-Axis: Label=\"{y_label}\", Range={y_range}"
                )
            sections.append("\n".join(axis_lines))
        
        # OCR texts
        if context.ocr_texts:
            ocr_lines = ["### OCR Texts (with confidence)"]
            for i, t in enumerate(context.ocr_texts[:20]):
                role = t.get("role", "")
                role_str = f" [{role}]" if role and role != "unknown" else ""
                ocr_lines.append(
                    f'{i+1}. "{t["text"]}"{role_str} (conf: {t["confidence"]:.0%})'
                )
            if len(context.ocr_texts) > 20:
                ocr_lines.append(f"... and {len(context.ocr_texts) - 20} more")
            sections.append("\n".join(ocr_lines))
        
        # Estimated series
        if context.estimated_series:
            series_lines = ["### Estimated Data (from geometric mapping)"]
            for s in context.estimated_series:
                points_preview = ", ".join(
                    f"{p['label']}:{p['value']}"
                    for p in s.get("points", [])[:5]
                )
                color = s.get("color", "")
                color_str = f" ({color})" if color else ""
                series_lines.append(f"- {s['name']}{color_str}: {points_preview}...")
            sections.append("\n".join(series_lines))
        
        # Warnings
        if context.warnings:
            warn_lines = ["### Warnings"]
            for w in context.warnings[:5]:
                warn_lines.append(f"- {w}")
            sections.append("\n".join(warn_lines))
        
        return "\n\n".join(sections)
    
    def _format_task_section(self, task: PromptTask) -> str:
        """Format the task section of the prompt."""
        if task == PromptTask.FULL_REASONING:
            return """1. **OCR CORRECTION**: Fix common OCR errors in detected texts
   - "loo" → "100", "O" → "0" in numbers, etc.
   - Identify chart title, axis labels

2. **VALUE REFINEMENT**: Verify and correct estimated values
   - Check consistency with axis ranges
   - Identify measurement units (%, USD, Years, etc.)

3. **LEGEND MAPPING**: Associate colors with series names
   - Match legend text to element colors

4. **DESCRIPTION**: Generate academic-style summary
   - What the chart shows
   - Key trends and patterns
   - Notable values"""
        
        elif task == PromptTask.OCR_CORRECTION:
            return "Fix OCR errors and identify chart components (title, axis labels)."
        
        elif task == PromptTask.VALUE_EXTRACTION:
            return "Verify and refine extracted values against axis ranges."
        
        elif task == PromptTask.DESCRIPTION_ONLY:
            return "Generate an academic-style description of the chart."
        
        return "Analyze the chart data."
    
    def _format_output_section(self) -> str:
        """Format the output section of the prompt."""
        return """Return ONLY valid JSON in this format:
```json
{
    "title": "chart title or null",
    "x_axis_label": "x-axis label or null",
    "y_axis_label": "y-axis label or null",
    "units": {"x": "unit", "y": "unit"},
    "corrections": [
        {"original": "loo", "corrected": "100", "reason": "OCR error"}
    ],
    "series": [
        {
            "name": "Series name",
            "color_rgb": [r, g, b],
            "points": [
                {"label": "Q1", "value": 100},
                {"label": "Q2", "value": 150}
            ]
        }
    ],
    "description": "Academic description of the chart",
    "trend": "increasing|decreasing|stable|mixed",
    "confidence": 0.85
}
```"""
    
    def _format_constraints_section(self, context: CanonicalContext) -> str:
        """Format anti-hallucination constraints."""
        constraints = ["\n### CONSTRAINTS (CRITICAL)"]
        
        constraints.append("- ONLY use data from INPUT CONTEXT")
        constraints.append("- Do NOT invent values not present in the data")
        constraints.append("- If uncertain, set confidence < 0.7")
        
        # Add specific constraints based on data quality
        if context.extraction_confidence < 0.6:
            constraints.append("- LOW CONFIDENCE extraction: be extra cautious")
        
        if context.warnings:
            constraints.append(f"- {len(context.warnings)} warning(s) detected: verify carefully")
        
        return "\n".join(constraints)
    
    def _find_axis_label(
        self,
        texts: List[OCRText],
        role: str,
    ) -> Optional[str]:
        """Find axis label from OCR texts."""
        for text in texts:
            if text.role == role:
                return text.text
        return None
    
    # =========================================================================
    # Template Methods (Defaults)
    # =========================================================================
    
    def _get_reasoning_template(self) -> str:
        """Get default reasoning template."""
        return """You are a senior data analyst. Analyze the chart data below.

{context}

{tasks}

{output_format}

{constraints}"""
    
    def _get_ocr_correction_template(self) -> str:
        """Get default OCR correction template."""
        return """Fix OCR errors in these texts from a {chart_type} chart:

{texts}

Common errors: loo→100, O→0, l→1, S→5

Output JSON with corrections."""
    
    def _get_description_template(self) -> str:
        """Get default description template."""
        return """Generate a 2-3 sentence academic description for this {chart_type} chart.

Title: {title}
Data: {data_summary}

Be factual and precise. Base ONLY on provided data."""
    
    def _get_value_extraction_template(self) -> str:
        """Get default value extraction template."""
        return """Verify these extracted values against the axis ranges:

Axes: X={x_range}, Y={y_range}
Data: {estimated_values}

Flag any values outside ranges. Identify units."""


def create_prompt_builder(
    config: Optional[PromptConfig] = None,
) -> GeminiPromptBuilder:
    """
    Factory function to create a prompt builder.
    
    Args:
        config: Optional configuration
    
    Returns:
        Configured GeminiPromptBuilder
    """
    return GeminiPromptBuilder(config)
