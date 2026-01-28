"""
Gemini Reasoning Engine

Google Gemini API integration for Stage 4 reasoning.
Supports both text-only and vision (multimodal) reasoning.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...schemas.common import Color
from ...schemas.enums import ChartType
from ...schemas.stage_outputs import (
    DataPoint,
    DataSeries,
    OCRText,
    RawMetadata,
    RefinedChartData,
)
from .reasoning_engine import ReasoningEngine, ReasoningResult

logger = logging.getLogger(__name__)


class GeminiConfig(BaseModel):
    """Configuration for Gemini reasoning engine."""
    
    # API settings
    api_key: Optional[str] = Field(
        default=None,
        description="Google API key (reads from GOOGLE_API_KEY env if not set)"
    )
    model_name: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model to use"
    )
    
    # Generation settings
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=100, le=8192)
    top_p: float = Field(default=0.95, ge=0, le=1)
    
    # Vision settings
    use_vision: bool = Field(
        default=False,
        description="Use vision model with image input"
    )
    
    # Prompt settings
    system_prompt: str = Field(
        default=(
            "You are an expert chart analysis AI. "
            "You analyze charts and extract structured data with high accuracy. "
            "You correct OCR errors and generate clear descriptions."
        )
    )
    
    # Retry settings
    max_retries: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1.0, ge=0)


class GeminiReasoningEngine(ReasoningEngine):
    """
    Google Gemini API reasoning engine.
    
    Features:
    - OCR error correction using context
    - Value extraction and mapping
    - Legend-color association
    - Academic-style description generation
    
    Example:
        config = GeminiConfig(api_key="...")
        engine = GeminiReasoningEngine(config)
        result = engine.reason(metadata)
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize Gemini reasoning engine.
        
        Args:
            config: Gemini configuration
        """
        self.config = config or GeminiConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get API key
        self.api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            # Try loading from secrets file
            secrets_path = Path(__file__).parent.parent.parent.parent.parent / "config/secrets/.env"
            if secrets_path.exists():
                self.api_key = self._load_api_key_from_file(secrets_path)
        
        if not self.api_key:
            self.logger.warning(
                "No Gemini API key found. "
                "Set GOOGLE_API_KEY environment variable or pass api_key in config."
            )
            self._client = None
            self._model = None
        else:
            self._initialize_client()
        
        self.logger.info(
            f"GeminiReasoningEngine initialized | "
            f"model={self.config.model_name} | "
            f"vision={self.config.use_vision}"
        )
    
    def _load_api_key_from_file(self, env_path: Path) -> Optional[str]:
        """Load API key from .env file."""
        try:
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception as e:
            self.logger.warning(f"Failed to load API key from {env_path}: {e}")
        return None
    
    def _initialize_client(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            
            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            
            # Create model
            self._model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                system_instruction=self.config.system_prompt,
            )
            
            self._client = genai
            self.logger.info(f"Gemini client initialized | model={self.config.model_name}")
            
        except ImportError:
            self.logger.error(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
            self._client = None
            self._model = None
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            self._client = None
            self._model = None
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return self._model is not None
    
    def reason(
        self,
        metadata: RawMetadata,
        image_path: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Perform semantic reasoning on chart metadata.
        
        Args:
            metadata: Raw metadata from Stage 3
            image_path: Optional path to chart image for vision models
        
        Returns:
            ReasoningResult with refined data
        """
        if not self.is_available():
            return self._fallback_reasoning(metadata)
        
        chart_id = metadata.chart_id
        self.logger.info(f"Reasoning started | chart_id={chart_id}")
        
        try:
            # Build prompt
            prompt = self._build_reasoning_prompt(metadata)
            
            # Call Gemini
            if self.config.use_vision and image_path:
                response = self._call_with_image(prompt, image_path)
            else:
                response = self._call_text_only(prompt)
            
            # Parse response
            result = self._parse_reasoning_response(response, metadata)
            
            corrections_count = len(result.corrections) if result.corrections else 0
            self.logger.info(
                f"Reasoning complete | chart_id={chart_id} | "
                f"corrections={corrections_count}"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Reasoning failed | chart_id={chart_id} | error={e}")
            return ReasoningResult(
                success=False,
                error_message=str(e),
                description="Reasoning failed - using raw data",
            )
    
    def correct_ocr(
        self,
        texts: List[OCRText],
        chart_type: ChartType,
    ) -> tuple[List[OCRText], List[Dict[str, str]]]:
        """
        Correct OCR errors using Gemini.
        
        Args:
            texts: OCR text results
            chart_type: Detected chart type
        
        Returns:
            Tuple of (corrected texts, list of corrections)
        """
        if not self.is_available() or not texts:
            return texts, []
        
        # Build correction prompt
        prompt = self._build_ocr_correction_prompt(texts, chart_type)
        
        try:
            response = self._call_text_only(prompt)
            corrected, corrections = self._parse_ocr_correction(response, texts)
            return corrected, corrections
        except Exception as e:
            self.logger.warning(f"OCR correction failed: {e}")
            return texts, []
    
    def generate_description(
        self,
        chart_type: ChartType,
        title: Optional[str],
        series: List[DataSeries],
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> str:
        """
        Generate academic-style chart description.
        
        Args:
            chart_type: Type of chart
            title: Chart title
            series: Data series
            x_label: X-axis label
            y_label: Y-axis label
        
        Returns:
            Academic-style description string
        """
        if not self.is_available():
            return self._generate_fallback_description(
                chart_type, title, series, x_label, y_label
            )
        
        prompt = self._build_description_prompt(
            chart_type, title, series, x_label, y_label
        )
        
        try:
            response = self._call_text_only(prompt)
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Description generation failed: {e}")
            return self._generate_fallback_description(
                chart_type, title, series, x_label, y_label
            )
    
    # =========================================================================
    # Private Methods: API Calls
    # =========================================================================
    
    def _call_text_only(self, prompt: str) -> str:
        """Call Gemini with text-only prompt."""
        response = self._model.generate_content(prompt)
        return response.text
    
    def _call_with_image(self, prompt: str, image_path: str) -> str:
        """Call Gemini with image and text prompt."""
        from PIL import Image
        
        image = Image.open(image_path)
        response = self._model.generate_content([prompt, image])
        return response.text
    
    # =========================================================================
    # Private Methods: Prompt Building
    # =========================================================================
    
    def _build_reasoning_prompt(self, metadata: RawMetadata) -> str:
        """Build the main reasoning prompt."""
        # Format OCR texts
        ocr_texts = []
        for i, text in enumerate(metadata.texts):
            role_str = f" (role: {text.role})" if text.role else ""
            ocr_texts.append(f"  {i+1}. \"{text.text}\"{role_str} [conf: {text.confidence:.2f}]")
        ocr_section = "\n".join(ocr_texts) if ocr_texts else "  No text detected"
        
        # Format elements
        elements = []
        for i, elem in enumerate(metadata.elements):
            color_str = ""
            if elem.color:
                color_str = f" color=RGB({elem.color.r},{elem.color.g},{elem.color.b})"
            elements.append(
                f"  {i+1}. {elem.element_type} at ({elem.center.x}, {elem.center.y}){color_str}"
            )
        elem_section = "\n".join(elements) if elements else "  No elements detected"
        
        prompt = f"""Analyze this {metadata.chart_type.value} chart and extract structured data.

## Input Data

### OCR Texts:
{ocr_section}

### Detected Elements:
{elem_section}

## Tasks

1. **Fix OCR Errors**: Common errors include:
   - "loo" or "l00" -> "100"
   - "O" -> "0" in numeric context
   - "l" or "I" -> "1" in numeric context
   - "S" -> "5" in numeric context
   - Missing "%" after percentage values

2. **Identify Components**:
   - Chart title
   - X-axis label
   - Y-axis label
   - Legend items (map to colors if possible)

3. **Extract Data Series**: For each series/category, extract:
   - Series name
   - Data points (label, value)

## Output Format (JSON)

```json
{{
    "title": "Chart title or null",
    "x_axis_label": "X-axis label or null",
    "y_axis_label": "Y-axis label or null",
    "corrections": [
        {{"original": "loo", "corrected": "100", "reason": "common OCR error"}}
    ],
    "series": [
        {{
            "name": "Series name",
            "color_rgb": [r, g, b],
            "points": [
                {{"label": "Q1", "value": 100}},
                {{"label": "Q2", "value": 150}}
            ]
        }}
    ],
    "description": "Brief academic description of the chart"
}}
```

Return ONLY valid JSON, no additional text.
"""
        return prompt
    
    def _build_ocr_correction_prompt(
        self,
        texts: List[OCRText],
        chart_type: ChartType,
    ) -> str:
        """Build OCR correction prompt."""
        text_list = [f'"{t.text}" (conf: {t.confidence:.2f})' for t in texts]
        
        return f"""You are correcting OCR errors from a {chart_type.value} chart.

Input texts:
{chr(10).join(f"- {t}" for t in text_list)}

Common OCR errors:
- "loo", "l00", "1oo" -> "100"
- "O" -> "0" in numbers
- "l", "I" -> "1" in numbers
- "S" -> "5" in numbers

Return JSON:
```json
{{
    "corrections": [
        {{"original": "...", "corrected": "...", "reason": "..."}}
    ]
}}
```

Only include items that need correction. Return empty list if no corrections needed.
"""
    
    def _build_description_prompt(
        self,
        chart_type: ChartType,
        title: Optional[str],
        series: List[DataSeries],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> str:
        """Build description generation prompt."""
        # Format series info
        series_info = []
        for s in series:
            if s.points:
                values = [f"{p.label}: {p.value}" for p in s.points[:5]]
                series_info.append(f"- {s.name}: {', '.join(values)}")
        
        return f"""Generate a brief academic-style description for this chart.

Chart Type: {chart_type.value}
Title: {title or 'Unknown'}
X-axis: {x_label or 'Unknown'}
Y-axis: {y_label or 'Unknown'}

Data Series:
{chr(10).join(series_info) if series_info else 'No data available'}

Write 2-3 sentences describing:
1. What the chart shows
2. Key patterns or trends (if any)
3. Notable values (highest/lowest)

Use academic language. Be concise and factual.
"""
    
    # =========================================================================
    # Private Methods: Response Parsing
    # =========================================================================
    
    def _parse_reasoning_response(
        self,
        response: str,
        metadata: RawMetadata,
    ) -> ReasoningResult:
        """Parse Gemini reasoning response."""
        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try parsing entire response as JSON
            json_str = response
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return ReasoningResult(
                success=False,
                error_message=f"JSON parse error: {e}",
                raw_response=response,
            )
        
        # Extract fields
        title = data.get("title")
        x_label = data.get("x_axis_label")
        y_label = data.get("y_axis_label")
        corrections = data.get("corrections", [])
        description = data.get("description", "")
        
        # Parse series
        series = []
        for s_data in data.get("series", []):
            points = [
                DataPoint(
                    label=str(p.get("label", "")),
                    value=float(p.get("value", 0)),
                    confidence=0.9,
                )
                for p in s_data.get("points", [])
            ]
            
            color = None
            color_rgb = s_data.get("color_rgb")
            if color_rgb and isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
                r, g, b = color_rgb
                color = Color(r=int(r), g=int(g), b=int(b))
            
            series.append(DataSeries(
                name=s_data.get("name", "Unknown"),
                color=color,
                points=points,
            ))
        
        return ReasoningResult(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            series=series,
            description=description,
            corrections=corrections,
            confidence=0.85,
            raw_response=response,
            success=True,
        )
    
    def _parse_ocr_correction(
        self,
        response: str,
        original_texts: List[OCRText],
    ) -> tuple[List[OCRText], List[Dict[str, str]]]:
        """Parse OCR correction response."""
        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        try:
            data = json.loads(json_str)
            corrections = data.get("corrections", [])
        except json.JSONDecodeError:
            return original_texts, []
        
        # Build correction map
        correction_map = {c["original"]: c["corrected"] for c in corrections}
        
        # Apply corrections
        corrected_texts = []
        for text in original_texts:
            if text.text in correction_map:
                corrected = OCRText(
                    text=correction_map[text.text],
                    bbox=text.bbox,
                    confidence=text.confidence,
                    role=text.role,
                )
                corrected_texts.append(corrected)
            else:
                corrected_texts.append(text)
        
        return corrected_texts, corrections
    
    # =========================================================================
    # Fallback Methods (when API unavailable)
    # =========================================================================
    
    def _fallback_reasoning(self, metadata: RawMetadata) -> ReasoningResult:
        """Fallback reasoning when API is unavailable."""
        self.logger.warning("Using fallback reasoning (API unavailable)")
        
        # Extract what we can from raw data
        grouped = self._extract_text_by_role(metadata.texts)
        
        title = grouped["title"][0] if grouped["title"] else None
        x_label = grouped["xlabel"][0] if grouped["xlabel"] else None
        y_label = grouped["ylabel"][0] if grouped["ylabel"] else None
        
        # Create single series with raw values
        points = []
        for text in grouped.get("value", []):
            try:
                value = float(text.replace(",", "").replace("%", ""))
                points.append(DataPoint(label="", value=value, confidence=0.5))
            except ValueError:
                pass
        
        series = [DataSeries(name="Data", points=points)] if points else []
        
        description = self._generate_fallback_description(
            metadata.chart_type, title, series, x_label, y_label
        )
        
        return ReasoningResult(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            series=series,
            description=description,
            corrections=[],
            confidence=0.5,
            reasoning_log=["Used fallback reasoning (API unavailable)"],
            success=True,
        )
    
    def _generate_fallback_description(
        self,
        chart_type: ChartType,
        title: Optional[str],
        series: List[DataSeries],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> str:
        """Generate simple description without API."""
        parts = [f"This is a {chart_type.value} chart"]
        
        if title:
            parts.append(f'titled "{title}"')
        
        if x_label:
            parts.append(f"with x-axis showing {x_label}")
        
        if y_label:
            parts.append(f"and y-axis showing {y_label}")
        
        if series and series[0].points:
            n_points = sum(len(s.points) for s in series)
            parts.append(f"containing {n_points} data points")
        
        return ". ".join(parts) + "."
