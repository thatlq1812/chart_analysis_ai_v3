"""
ML-based Chart Classifier.

Uses a trained Random Forest model for chart type classification.
Falls back to rule-based approach if model not available.

Author: That Le
Date: 2025-01-21
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ...schemas.enums import ChartType
from .simple_classifier import SimpleChartClassifier, SimpleClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class MLClassificationResult:
    """Result of ML-based classification."""
    
    chart_type: ChartType
    confidence: float
    features: Dict[str, float]
    probabilities: Dict[str, float]
    reasoning: str


class MLChartClassifier:
    """
    ML-based chart classifier using trained Random Forest.
    
    Extracts image features using SimpleChartClassifier, then
    uses a trained Random Forest model for classification.
    
    Example:
        classifier = MLChartClassifier()
        result = classifier.classify(image)
    """
    
    # Mapping from model output labels to ChartType enum
    # Model outputs: bar, line, pie, other
    LABEL_TO_CHARTTYPE = {
        "bar": ChartType.BAR,
        "line": ChartType.LINE,
        "pie": ChartType.PIE,
        "other": ChartType.UNKNOWN,
    }
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[SimpleClassifierConfig] = None,
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model file
            config: Configuration for feature extraction
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_extractor = SimpleChartClassifier(config or SimpleClassifierConfig())
        
        # Load model
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
        if model_path is None:
            # Default path
            model_path = Path(__file__).parent.parent.parent.parent.parent / "models/weights/chart_classifier_rf.pkl"
        
        if model_path.exists():
            self._load_model(model_path)
        else:
            self.logger.warning(
                f"Model not found at {model_path}. "
                "Will use rule-based fallback."
            )
    
    def _load_model(self, model_path: Path):
        """Load trained model."""
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            
            self.model = data["model"]
            self.label_encoder = data["label_encoder"]
            self.feature_names = data["feature_names"]
            
            self.logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def classify(
        self,
        image: np.ndarray,
        chart_id: str = "unknown",
    ) -> MLClassificationResult:
        """
        Classify chart type from image.
        
        Args:
            image: BGR color image
            chart_id: Chart identifier for logging
        
        Returns:
            MLClassificationResult with type and confidence
        """
        self.logger.debug(f"ML classification started | chart_id={chart_id}")
        
        # Extract features
        simple_result = self.feature_extractor.classify(image, chart_id=chart_id)
        features = simple_result.features
        
        # Use model if available
        if self.model is not None:
            return self._classify_with_model(features, chart_id)
        else:
            # Fallback to rule-based
            chart_type = self.LABEL_TO_CHARTTYPE.get(
                simple_result.chart_type.value,
                ChartType.UNKNOWN
            )
            return MLClassificationResult(
                chart_type=chart_type,
                confidence=simple_result.confidence,
                features=features,
                probabilities={simple_result.chart_type.value: simple_result.confidence},
                reasoning=simple_result.reasoning + " (rule-based fallback)",
            )
    
    def _classify_with_model(
        self,
        features: Dict[str, float],
        chart_id: str,
    ) -> MLClassificationResult:
        """Classify using trained model."""
        # Build feature vector in correct order
        feature_vector = [features[name] for name in self.feature_names]
        X = np.array([feature_vector])
        
        # Predict
        y_pred = self.model.predict(X)[0]
        y_proba = self.model.predict_proba(X)[0]
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform([y_pred])[0]
        
        # Get probabilities for all classes
        probabilities = {
            label: prob
            for label, prob in zip(self.label_encoder.classes_, y_proba)
        }
        
        # Map to ChartType
        chart_type = self.LABEL_TO_CHARTTYPE.get(predicted_label, ChartType.UNKNOWN)
        confidence = y_proba[y_pred]
        
        # Generate reasoning
        top_features = sorted(
            [(name, features[name]) for name in self.feature_names],
            key=lambda x: -self.model.feature_importances_[
                self.feature_names.index(x[0])
            ]
        )[:3]
        
        reasoning = (
            f"ML classified as {predicted_label} (confidence={confidence:.2f}). "
            f"Top features: {', '.join(f'{n}={v:.2f}' for n, v in top_features)}"
        )
        
        self.logger.info(
            f"ML classification complete | chart_id={chart_id} | "
            f"type={chart_type.value} | confidence={confidence:.2f}"
        )
        
        return MLClassificationResult(
            chart_type=chart_type,
            confidence=confidence,
            features=features,
            probabilities=probabilities,
            reasoning=reasoning,
        )
