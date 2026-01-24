"""
Chart Classifier Module

Classifies chart type from extracted features.

Uses a hybrid approach combining:
1. Structural features (bars, lines, circles)
2. Spatial layout patterns
3. Text clues from OCR

Reference: docs/architecture/PIPELINE_FLOW.md
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from ...schemas.enums import ChartType
from ...schemas.extraction import (
    BarRectangle,
    DataMarker,
    PieSlice,
    Polyline,
)
from ...schemas.stage_outputs import OCRText

logger = logging.getLogger(__name__)


class ClassifierConfig(BaseModel):
    """Configuration for chart classification."""
    
    # Feature thresholds
    min_bars_for_bar_chart: int = Field(default=2, ge=1)
    min_points_for_line_chart: int = Field(default=3, ge=2)
    min_points_for_scatter: int = Field(default=5, ge=2)
    circularity_threshold: float = Field(default=0.7, gt=0, le=1)
    
    # Confidence thresholds
    min_confidence: float = Field(default=0.5, ge=0, le=1)


@dataclass
class ClassificationResult:
    """Result of chart classification."""
    
    chart_type: ChartType
    confidence: float
    features: Dict[str, float]
    reasoning: str


class ChartClassifier:
    """
    Classifies chart type from structural features.
    
    Uses a rule-based approach with feature scoring to determine
    the most likely chart type.
    
    Example:
        classifier = ChartClassifier(ClassifierConfig())
        result = classifier.classify(bars, polylines, markers, texts)
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize classifier.
        
        Args:
            config: Classification configuration (uses defaults if None)
        """
        self.config = config or ClassifierConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def classify(
        self,
        bars: List[BarRectangle],
        polylines: List[Polyline],
        markers: List[DataMarker],
        slices: List[PieSlice],
        texts: List[OCRText],
        image_shape: tuple,
        chart_id: str = "unknown",
    ) -> ClassificationResult:
        """
        Classify chart type from features.
        
        Args:
            bars: Detected bar rectangles
            polylines: Extracted polylines
            markers: Detected data markers
            slices: Detected pie slices
            texts: OCR text results
            image_shape: (height, width) of image
            chart_id: Chart identifier for logging
        
        Returns:
            ClassificationResult with type and confidence
        """
        self.logger.debug(f"Classification started | chart_id={chart_id}")
        
        # Compute feature scores
        features = self._compute_features(
            bars, polylines, markers, slices, texts, image_shape
        )
        
        # Score each chart type
        scores = {
            ChartType.BAR: self._score_bar(features),
            ChartType.LINE: self._score_line(features),
            ChartType.PIE: self._score_pie(features),
            ChartType.SCATTER: self._score_scatter(features),
        }
        
        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Check confidence threshold
        if best_score < self.config.min_confidence:
            best_type = ChartType.UNKNOWN
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, scores, best_type)
        
        self.logger.info(
            f"Classification complete | chart_id={chart_id} | "
            f"type={best_type.value} | confidence={best_score:.2f}"
        )
        
        return ClassificationResult(
            chart_type=best_type,
            confidence=best_score,
            features=features,
            reasoning=reasoning,
        )
    
    def _compute_features(
        self,
        bars: List[BarRectangle],
        polylines: List[Polyline],
        markers: List[DataMarker],
        slices: List[PieSlice],
        texts: List[OCRText],
        image_shape: tuple,
    ) -> Dict[str, float]:
        """Compute feature scores for classification."""
        h, w = image_shape[:2]
        total_area = h * w
        
        features = {
            # Count features
            "num_bars": float(len(bars)),
            "num_polylines": float(len(polylines)),
            "num_markers": float(len(markers)),
            "num_slices": float(len(slices)),
            
            # Coverage features
            "bar_coverage": self._compute_bar_coverage(bars, total_area),
            "line_coverage": self._compute_line_length(polylines, w),
            
            # Spatial features
            "bars_aligned": self._check_bar_alignment(bars),
            "markers_clustered": self._check_marker_clustering(markers),
            "has_circular_structure": self._check_circular_structure(markers, slices),
            
            # Text features
            "has_percentage_labels": self._check_percentage_labels(texts),
            "has_axis_labels": self._check_axis_labels(texts),
        }
        
        return features
    
    def _score_bar(self, features: Dict[str, float]) -> float:
        """Score likelihood of bar chart."""
        score = 0.0
        
        # Strong indicator: multiple bars
        if features["num_bars"] >= self.config.min_bars_for_bar_chart:
            score += 0.4
        
        # Bars are aligned (either vertically or horizontally)
        score += 0.3 * features["bars_aligned"]
        
        # Has axis labels
        score += 0.2 * features["has_axis_labels"]
        
        # Penalty for line/scatter indicators
        if features["num_polylines"] > 0 and features["num_bars"] == 0:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_line(self, features: Dict[str, float]) -> float:
        """Score likelihood of line chart."""
        score = 0.0
        
        # Strong indicator: polylines with multiple vertices
        if features["num_polylines"] > 0:
            score += 0.4
        
        # Good line coverage
        score += 0.3 * features["line_coverage"]
        
        # Has axis labels
        score += 0.2 * features["has_axis_labels"]
        
        # Penalty for bar indicators
        if features["num_bars"] > 3:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_pie(self, features: Dict[str, float]) -> float:
        """Score likelihood of pie chart."""
        score = 0.0
        
        # Strong indicator: slices or circular structure
        if features["num_slices"] > 0:
            score += 0.5
        
        score += 0.3 * features["has_circular_structure"]
        
        # Percentage labels common in pie charts
        score += 0.2 * features["has_percentage_labels"]
        
        # No axis labels in pie charts
        if features["has_axis_labels"] > 0.5:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_scatter(self, features: Dict[str, float]) -> float:
        """Score likelihood of scatter plot."""
        score = 0.0
        
        # Strong indicator: many markers without connecting lines
        if features["num_markers"] >= self.config.min_points_for_scatter:
            score += 0.4
        
        # Markers not well-clustered (spread across plot)
        if features["num_markers"] > 0 and features["markers_clustered"] < 0.5:
            score += 0.3
        
        # Has axis labels
        score += 0.2 * features["has_axis_labels"]
        
        # Penalty if markers are connected by polylines
        if features["num_polylines"] > 0 and features["num_markers"] > 0:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _compute_bar_coverage(
        self,
        bars: List[BarRectangle],
        total_area: float,
    ) -> float:
        """Compute fraction of image covered by bars."""
        if not bars or total_area <= 0:
            return 0.0
        
        bar_area = sum(bar.area for bar in bars)
        return min(1.0, bar_area / total_area)
    
    def _compute_line_length(
        self,
        polylines: List[Polyline],
        image_width: float,
    ) -> float:
        """Compute normalized total line length."""
        if not polylines or image_width <= 0:
            return 0.0
        
        total_length = sum(pl.total_length for pl in polylines)
        
        # Normalize by image width
        normalized = total_length / (image_width * len(polylines))
        return min(1.0, normalized)
    
    def _check_bar_alignment(self, bars: List[BarRectangle]) -> float:
        """Check if bars are aligned (typical for bar charts)."""
        if len(bars) < 2:
            return 0.0
        
        # Check vertical alignment (bottom edges aligned)
        bottoms = [bar.y_max for bar in bars]
        bottom_std = np.std(bottoms) if len(bottoms) > 1 else 0
        
        # Check horizontal alignment (left edges evenly spaced)
        lefts = sorted([bar.x_min for bar in bars])
        if len(lefts) > 2:
            spacings = [lefts[i+1] - lefts[i] for i in range(len(lefts)-1)]
            spacing_std = np.std(spacings) / (np.mean(spacings) + 1e-10)
        else:
            spacing_std = 0
        
        # Good alignment = low standard deviation
        alignment_score = 1.0 - min(1.0, (bottom_std + spacing_std * 100) / 100)
        return max(0.0, alignment_score)
    
    def _check_marker_clustering(self, markers: List[DataMarker]) -> float:
        """Check if markers are clustered (typical for legend) vs spread."""
        if len(markers) < 3:
            return 0.5  # Neutral
        
        # Compute pairwise distances
        centers = [(m.center.x, m.center.y) for m in markers]
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                distances.append((dx**2 + dy**2) ** 0.5)
        
        if not distances:
            return 0.5
        
        # High variance in distances = scattered
        cv = np.std(distances) / (np.mean(distances) + 1e-10)
        
        # cv > 1 means scattered (good for scatter plot)
        return min(1.0, cv)
    
    def _check_circular_structure(
        self,
        markers: List[DataMarker],
        slices: List[PieSlice],
    ) -> float:
        """Check for circular/radial structure."""
        if slices:
            return 1.0
        
        if len(markers) < 4:
            return 0.0
        
        # Check if markers form a circle
        centers = [(m.center.x, m.center.y) for m in markers]
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        
        # Distances from centroid
        distances = [((x - cx)**2 + (y - cy)**2)**0.5 for x, y in centers]
        
        # Circular if distances are similar
        cv = np.std(distances) / (np.mean(distances) + 1e-10)
        
        return max(0.0, 1.0 - cv)
    
    def _check_percentage_labels(self, texts: List[OCRText]) -> float:
        """Check for percentage labels (common in pie charts)."""
        percent_count = sum(1 for t in texts if "%" in t.text)
        
        if not texts:
            return 0.0
        
        return min(1.0, percent_count / len(texts) * 2)
    
    def _check_axis_labels(self, texts: List[OCRText]) -> float:
        """Check for axis labels (common in bar/line/scatter)."""
        axis_roles = {"x_tick", "y_tick", "x_axis_label", "y_axis_label"}
        axis_count = sum(1 for t in texts if t.role in axis_roles)
        
        if not texts:
            return 0.0
        
        return min(1.0, axis_count / max(4, len(texts)))
    
    def _generate_reasoning(
        self,
        features: Dict[str, float],
        scores: Dict[ChartType, float],
        best_type: ChartType,
    ) -> str:
        """Generate human-readable reasoning."""
        reasons = []
        
        if best_type == ChartType.BAR:
            reasons.append(f"Detected {int(features['num_bars'])} bar elements")
            if features["bars_aligned"] > 0.5:
                reasons.append("Bars are well-aligned")
        
        elif best_type == ChartType.LINE:
            reasons.append(f"Detected {int(features['num_polylines'])} polyline(s)")
            if features["has_axis_labels"] > 0.5:
                reasons.append("Has axis labels")
        
        elif best_type == ChartType.PIE:
            if features["num_slices"] > 0:
                reasons.append(f"Detected {int(features['num_slices'])} pie slices")
            if features["has_percentage_labels"] > 0.5:
                reasons.append("Has percentage labels")
        
        elif best_type == ChartType.SCATTER:
            reasons.append(f"Detected {int(features['num_markers'])} data markers")
            if features["markers_clustered"] < 0.5:
                reasons.append("Markers are scattered across plot")
        
        else:
            reasons.append("Could not determine chart type with confidence")
        
        return "; ".join(reasons)
