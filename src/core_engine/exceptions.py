"""
Custom Exceptions for Chart Analysis Pipeline

Hierarchical exception structure:
    ChartAnalysisError (base)
    ├── PipelineError
    │   ├── StageInputError
    │   └── StageProcessingError
    ├── ConfigurationError
    └── ModelError
"""

from typing import Optional


class ChartAnalysisError(Exception):
    """Base exception for all chart analysis errors."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class PipelineError(ChartAnalysisError):
    """
    Error during pipeline execution.
    
    Attributes:
        stage: Name of the stage where error occurred
        recoverable: Whether the pipeline can continue
        original_error: The underlying exception if any
    """
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        recoverable: bool = False,
        original_error: Optional[Exception] = None,
    ):
        self.stage = stage
        self.recoverable = recoverable
        self.original_error = original_error
        
        # Format message with stage info
        if stage:
            message = f"[{stage}] {message}"
        
        super().__init__(message)


class StageInputError(PipelineError):
    """Invalid input provided to a pipeline stage."""
    
    def __init__(
        self,
        message: str,
        stage: str,
        expected_type: Optional[str] = None,
        received_type: Optional[str] = None,
    ):
        self.expected_type = expected_type
        self.received_type = received_type
        
        if expected_type and received_type:
            message = f"{message} (expected: {expected_type}, received: {received_type})"
        
        super().__init__(message, stage=stage, recoverable=False)


class StageProcessingError(PipelineError):
    """Error during stage processing."""
    
    def __init__(
        self,
        message: str,
        stage: str,
        recoverable: bool = False,
        fallback_available: bool = False,
        original_error: Optional[Exception] = None,
    ):
        self.fallback_available = fallback_available
        super().__init__(
            message,
            stage=stage,
            recoverable=recoverable,
            original_error=original_error,
        )


class ConfigurationError(ChartAnalysisError):
    """Invalid or missing configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        
        if config_key:
            message = f"Configuration error for '{config_key}': {message}"
        
        super().__init__(message)


class ModelError(ChartAnalysisError):
    """Error related to AI model loading or inference."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        
        if model_name:
            message = f"Model '{model_name}': {message}"
        if model_path:
            message = f"{message} (path: {model_path})"
        
        super().__init__(message)


class ModelNotLoadedError(ModelError):
    """Model failed to load."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(message, model_name=model_name, model_path=model_path)


class OCRError(StageProcessingError):
    """OCR extraction failed."""
    
    def __init__(
        self,
        message: str,
        engine: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.engine = engine
        super().__init__(
            message,
            stage="extraction",
            recoverable=True,
            fallback_available=True,
            original_error=original_error,
        )


class GeometricCalculationError(StageProcessingError):
    """Geometric value mapping failed."""
    
    def __init__(
        self,
        message: str,
        chart_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.chart_id = chart_id
        super().__init__(
            message,
            stage="reasoning",
            recoverable=True,
            fallback_available=True,
            original_error=original_error,
        )


class SLMError(StageProcessingError):
    """SLM inference failed."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        timeout: bool = False,
        original_error: Optional[Exception] = None,
    ):
        self.model_name = model_name
        self.timeout = timeout
        super().__init__(
            message,
            stage="reasoning",
            recoverable=True,
            fallback_available=True,
            original_error=original_error,
        )
