"""
Reasoning Engine Abstract Interface

Base class for all reasoning engines (Gemini, Local SLM, etc.)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...schemas.enums import ChartType
from ...schemas.stage_outputs import (
    DataPoint,
    DataSeries,
    OCRText,
    RawMetadata,
    RefinedChartData,
)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result from reasoning engine."""
    
    # Core outputs
    title: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    series: List[DataSeries] = field(default_factory=list)
    description: str = ""
    
    # Corrections made
    corrections: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    reasoning_log: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    
    # Error handling
    success: bool = True
    error_message: Optional[str] = None


class ReasoningEngine(ABC):
    """
    Abstract base class for reasoning engines.
    
    All reasoning engines must implement:
    - reason(): Main reasoning method
    - correct_ocr(): OCR error correction
    - generate_description(): Description generation
    """
    
    def __init__(self, config: Optional[BaseModel] = None):
        """
        Initialize reasoning engine.
        
        Args:
            config: Engine-specific configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def correct_ocr(
        self,
        texts: List[OCRText],
        chart_type: ChartType,
    ) -> tuple[List[OCRText], List[Dict[str, str]]]:
        """
        Correct OCR errors using context.
        
        Args:
            texts: OCR text results
            chart_type: Detected chart type
        
        Returns:
            Tuple of (corrected texts, list of corrections made)
        """
        pass
    
    @abstractmethod
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
        pass
    
    def _extract_text_by_role(
        self,
        texts: List[OCRText],
    ) -> Dict[str, List[str]]:
        """
        Group OCR texts by their detected role.
        
        Args:
            texts: List of OCR text results
        
        Returns:
            Dict mapping role to list of text strings
        """
        grouped: Dict[str, List[str]] = {
            "title": [],
            "xlabel": [],
            "ylabel": [],
            "legend": [],
            "value": [],
            "unknown": [],
        }
        
        for text in texts:
            role = text.role or "unknown"
            if role in grouped:
                grouped[role].append(text.text)
            else:
                grouped["unknown"].append(text.text)
        
        return grouped
    
    def _find_title(self, texts: List[OCRText]) -> Optional[str]:
        """Find the most likely title from OCR texts."""
        grouped = self._extract_text_by_role(texts)
        
        # Priority: explicit title role > top position text
        if grouped["title"]:
            # Join multiple title lines
            return " ".join(grouped["title"])
        
        # Fallback: look for longest text in top region
        # (This is a heuristic, actual logic may vary)
        return None
    
    def _find_axis_labels(
        self,
        texts: List[OCRText],
    ) -> tuple[Optional[str], Optional[str]]:
        """Find X and Y axis labels from OCR texts."""
        grouped = self._extract_text_by_role(texts)
        
        x_label = grouped["xlabel"][0] if grouped["xlabel"] else None
        y_label = grouped["ylabel"][0] if grouped["ylabel"] else None
        
        return x_label, y_label
