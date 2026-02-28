"""
AI Router-Based Reasoning Engine

Implements the ReasoningEngine interface using the new AIRouter layer.
This is the replacement for GeminiReasoningEngine when using multi-provider routing.

Instead of calling Gemini directly, this engine:
1. Builds prompts using the existing prompt system
2. Routes the request through AIRouter (tries local_slm -> gemini -> openai)
3. Parses the standardized AIResponse into ReasoningResult

Usage in Stage 4:
    config = ReasoningConfig(engine="router")
    stage = Stage4Reasoning(config)
    # Stage will auto-create AIRouterEngine backed by the configured router
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...ai.router import AIRouter
from ...ai.task_types import TaskType
from ...ai.exceptions import AIProviderExhaustedError
from ...ai.prompts import (
    CHART_REASONING_SYSTEM,
    OCR_CORRECTION_SYSTEM,
    DESCRIPTION_GEN_SYSTEM,
    format_reasoning_user,
    format_ocr_correction_user,
    format_description_user,
)
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


class AIRouterEngine(ReasoningEngine):
    """
    ReasoningEngine implementation backed by AIRouter.

    Routes requests through the multi-provider fallback chain.
    Parses provider responses identically to GeminiReasoningEngine.

    Attributes:
        provider_id: "router" -- not a single provider
    """

    def __init__(self, router: Optional[AIRouter] = None) -> None:
        """
        Initialize with an AIRouter instance.

        Args:
            router: Configured AIRouter. If None, creates a default router
                    using environment variable API keys.
        """
        super().__init__()
        self._router = router or AIRouter()
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_available(self) -> bool:
        """
        Check if at least one provider is configured.

        Always returns True -- availability is checked per-request
        via the router's health checks.
        """
        return bool(self._router._adapters)

    # -------------------------------------------------------------------------
    # ReasoningEngine interface
    # -------------------------------------------------------------------------

    def reason(
        self,
        metadata: RawMetadata,
        image_path: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Route chart reasoning through the AI provider chain.

        Args:
            metadata: Raw metadata from Stage 3
            image_path: Optional image path passed to vision providers

        Returns:
            ReasoningResult populated from the AI response
        """
        chart_id = metadata.chart_id
        self.logger.info(
            f"AIRouterEngine.reason | chart_id={chart_id} | "
            f"chart_type={metadata.chart_type.value}"
        )

        # Build prompts
        ocr_texts_payload = [
            {"text": t.text, "role": t.role or "unknown", "confidence": t.confidence}
            for t in metadata.texts
        ]
        elements_payload = [
            {
                "type": e.element_type,
                "center": {"x": e.center.x, "y": e.center.y},
                "color": (
                    {"r": e.color.r, "g": e.color.g, "b": e.color.b}
                    if e.color
                    else None
                ),
            }
            for e in metadata.elements
        ]
        axis_payload: Dict[str, Any] = {}
        if metadata.axis_info:
            ai = metadata.axis_info
            if ai.x_axis_detected:
                axis_payload["x_axis"] = {
                    "min_val": ai.x_min,
                    "max_val": ai.x_max,
                    "scale_factor": ai.x_scale_factor,
                    "calibration_confidence": ai.x_calibration_confidence,
                }
            if ai.y_axis_detected:
                axis_payload["y_axis"] = {
                    "min_val": ai.y_min,
                    "max_val": ai.y_max,
                    "scale_factor": ai.y_scale_factor,
                    "calibration_confidence": ai.y_calibration_confidence,
                }

        user_prompt = format_reasoning_user(
            chart_type=metadata.chart_type.value,
            ocr_texts=ocr_texts_payload,
            detected_elements=elements_payload,
            axis_info=axis_payload,
        )

        try:
            ai_response = self._router.route_sync(
                TaskType.CHART_REASONING,
                CHART_REASONING_SYSTEM,
                user_prompt,
                image_path=image_path,
            )

            if not ai_response.success:
                self.logger.warning(
                    f"AIRouterEngine | failed | chart_id={chart_id} | "
                    f"provider={ai_response.provider} | "
                    f"error={ai_response.error_message}"
                )
                return ReasoningResult(
                    success=False,
                    error_message=ai_response.error_message,
                )

            result = self._parse_reasoning_response(ai_response.content, metadata)
            result.reasoning_log.append(
                f"provider={ai_response.provider} | model={ai_response.model_used}"
            )
            self.logger.info(
                f"AIRouterEngine | success | chart_id={chart_id} | "
                f"provider={ai_response.provider} | "
                f"confidence={ai_response.confidence:.2f}"
            )
            return result

        except AIProviderExhaustedError as exc:
            self.logger.error(
                f"AIRouterEngine | all providers failed | chart_id={chart_id} | "
                f"errors={exc.errors}"
            )
            return ReasoningResult(
                success=False,
                error_message=str(exc),
                description="Reasoning failed - all providers exhausted",
            )

    def correct_ocr(
        self,
        texts: List[OCRText],
        chart_type: ChartType,
    ) -> tuple[List[OCRText], List[Dict[str, Any]]]:
        """
        Route OCR correction through the AI provider chain.

        Args:
            texts: OCR text results to correct
            chart_type: Detected chart type for context

        Returns:
            Tuple of (corrected texts, list of correction dicts)
        """
        if not texts:
            return texts, []

        tokens = [t.text for t in texts]
        user_prompt = format_ocr_correction_user(
            tokens=tokens,
            chart_type=chart_type.value,
        )

        try:
            ai_response = self._router.route_sync(
                TaskType.OCR_CORRECTION,
                OCR_CORRECTION_SYSTEM,
                user_prompt,
            )

            if not ai_response.success:
                return texts, []

            return self._parse_ocr_correction(ai_response.content, texts)

        except (AIProviderExhaustedError, Exception) as exc:
            self.logger.warning(
                f"AIRouterEngine.correct_ocr | failed | error={exc}"
            )
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
        Route description generation through the AI provider chain.

        Args:
            chart_type: Detected chart type
            title: Chart title
            series: Data series
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            Academic-style description string
        """
        series_summary = [
            {
                "name": s.name,
                "n_points": len(s.points),
                "values": [p.value for p in s.points[:5]],
            }
            for s in series
        ]

        user_prompt = format_description_user(
            chart_type=chart_type.value,
            title=title,
            x_label=x_label,
            y_label=y_label,
            series_summary=series_summary,
        )

        try:
            ai_response = self._router.route_sync(
                TaskType.DESCRIPTION_GEN,
                DESCRIPTION_GEN_SYSTEM,
                user_prompt,
            )
            if ai_response.success:
                return ai_response.content.strip()
        except Exception as exc:
            self.logger.warning(
                f"AIRouterEngine.generate_description | failed | error={exc}"
            )

        # Fallback: minimal description
        return self._fallback_description(chart_type, title, series, x_label, y_label)

    # -------------------------------------------------------------------------
    # Private: response parsing (mirrors GeminiReasoningEngine logic)
    # -------------------------------------------------------------------------

    def _parse_reasoning_response(
        self,
        response: str,
        metadata: RawMetadata,
    ) -> ReasoningResult:
        """Parse AI response content into ReasoningResult."""
        # Strip markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self.logger.warning(
                f"AIRouterEngine | JSON parse failed | error={exc} | "
                f"raw={response[:200]}"
            )
            return ReasoningResult(
                success=False,
                error_message=f"JSON parse error: {exc}",
                raw_response=response,
            )

        title = data.get("title")
        x_label = data.get("x_axis_label")
        y_label = data.get("y_axis_label")
        corrections = data.get("corrections", [])

        # Parse series
        series: List[DataSeries] = []
        for s_data in data.get("series", []):
            points = [
                DataPoint(
                    label=str(p.get("label", "")),
                    value=float(p.get("value", 0)),
                    confidence=0.9,
                )
                for p in s_data.get("points", [])
            ]
            color: Optional[Color] = None
            color_rgb = s_data.get("color_rgb")
            if color_rgb and isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
                color = Color(r=int(color_rgb[0]), g=int(color_rgb[1]), b=int(color_rgb[2]))

            series.append(
                DataSeries(
                    name=s_data.get("name", "Unknown"),
                    color=color,
                    points=points,
                )
            )

        return ReasoningResult(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            series=series,
            description=data.get("description", ""),
            corrections=corrections,
            confidence=0.85,
            raw_response=response,
            success=True,
        )

    def _parse_ocr_correction(
        self,
        response: str,
        original_texts: List[OCRText],
    ) -> tuple[List[OCRText], List[Dict[str, Any]]]:
        """Parse OCR correction response."""
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response.strip()

        try:
            data = json.loads(json_str)
            # Handle both array format and {"corrections": [...]} format
            if isinstance(data, list):
                corrections = data
            else:
                corrections = data.get("corrections", [])
        except json.JSONDecodeError:
            return original_texts, []

        correction_map = {c["original"]: c["corrected"] for c in corrections if "original" in c}

        corrected_texts: List[OCRText] = []
        for text in original_texts:
            if text.text in correction_map:
                corrected_texts.append(
                    OCRText(
                        text=correction_map[text.text],
                        bbox=text.bbox,
                        confidence=text.confidence,
                        role=text.role,
                    )
                )
            else:
                corrected_texts.append(text)

        return corrected_texts, corrections

    @staticmethod
    def _fallback_description(
        chart_type: ChartType,
        title: Optional[str],
        series: List[DataSeries],
        x_label: Optional[str],
        y_label: Optional[str],
    ) -> str:
        """Minimal description when AI is unavailable."""
        parts = [f"A {chart_type.value} chart"]
        if title:
            parts[0] += f' titled "{title}"'
        total = sum(len(s.points) for s in series)
        if total:
            parts.append(f"with {total} data points")
        if x_label:
            parts.append(f"x-axis: {x_label}")
        if y_label:
            parts.append(f"y-axis: {y_label}")
        return ". ".join(parts) + "."
