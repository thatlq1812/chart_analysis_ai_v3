"""
Base Stage Class

Abstract base class that all pipeline stages must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseStage(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for pipeline stages.
    
    All stages must implement:
    - process(): Main processing logic
    - validate_input(): Input validation
    
    Attributes:
        config: Stage-specific configuration
        logger: Stage logger instance
        name: Stage name for logging
    
    Example:
        class Stage1Ingestion(BaseStage[Path, Stage1Output]):
            def process(self, input_data: Path) -> Stage1Output:
                # Implementation
                pass
    """
    
    def __init__(self, config: BaseModel | dict):
        """
        Initialize stage with configuration.
        
        Args:
            config: Stage configuration (Pydantic model or dict)
        """
        self.config = config
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"stage.{self.name}")
        
        self.logger.debug(f"Initialized {self.name}")
    
    @property
    def is_critical(self) -> bool:
        """
        Whether this stage is critical for pipeline completion.
        
        Override in subclass if stage is optional.
        """
        return True
    
    @abstractmethod
    def process(self, input_data: InputT) -> OutputT:
        """
        Process input data and return output.
        
        Args:
            input_data: Input data for this stage
            
        Returns:
            Processed output data
            
        Raises:
            StageProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: InputT) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid
            
        Raises:
            StageInputError: If input is invalid
        """
        pass
    
    def get_fallback_output(self, input_data: InputT) -> OutputT:
        """
        Get fallback output when processing fails.
        
        Override in subclass to provide graceful degradation.
        
        Args:
            input_data: Original input
            
        Returns:
            Fallback output (may be empty/partial)
        """
        raise NotImplementedError(f"{self.name} does not support fallback")
    
    def __call__(self, input_data: InputT) -> OutputT:
        """
        Run stage with input validation.
        
        Args:
            input_data: Input data
            
        Returns:
            Processed output
            
        Raises:
            StageInputError: If validation fails
            StageProcessingError: If processing fails
        """
        self.logger.info(f"Starting {self.name}")
        
        # Validate input
        if not self.validate_input(input_data):
            from ..exceptions import StageInputError
            raise StageInputError(
                message="Input validation failed",
                stage=self.name,
            )
        
        # Process
        try:
            output = self.process(input_data)
            self.logger.info(f"Completed {self.name}")
            return output
            
        except Exception as e:
            self.logger.error(f"Failed {self.name}: {e}")
            raise
