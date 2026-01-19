---
applyTo: 'src/**'
---

# CODING STANDARDS - Python Guidelines

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-19 | That Le | Python coding standards for core engine |

## 1. Code Style

### 1.1. Formatter and Linter

| Tool | Purpose | Config File |
| --- | --- | --- |
| Black | Code formatting | `pyproject.toml` |
| Ruff | Linting (replaces flake8, isort) | `pyproject.toml` |
| MyPy | Type checking | `pyproject.toml` |

**pyproject.toml configuration:**

```toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
]
ignore = ["E501"]  # line too long (handled by black)

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
```

### 1.2. Import Organization

```python
# 1. Standard library (alphabetical)
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 2. Third-party (alphabetical)
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

# 3. Local imports (relative)
from .schemas import BoundingBox, ChartType
from .stages import BaseStage
from ..utils import compute_hash
```

### 1.3. Naming Conventions

| Type | Convention | Example |
| --- | --- | --- |
| Module | snake_case | `ocr_engine.py` |
| Class | PascalCase | `ChartDetector` |
| Function | snake_case | `extract_text()` |
| Variable | snake_case | `image_path` |
| Constant | SCREAMING_SNAKE | `MAX_IMAGE_SIZE` |
| Private | _leading_underscore | `_load_model()` |
| Type Variable | Single capital | `T`, `InputT` |

## 2. Type Hints

### 2.1. Required Everywhere

```python
# Bad: No type hints
def process(data):
    return data.transform()

# Good: Full type hints
def process(data: RawChartData) -> ProcessedChartData:
    return data.transform()
```

### 2.2. Common Patterns

```python
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from collections.abc import Iterator, Sequence
from pathlib import Path

# Optional (can be None)
def load_model(path: Optional[Path] = None) -> Model:
    pass

# Union types (Python 3.10+)
def parse_input(data: str | bytes | Path) -> ParsedData:
    pass

# Callable
def apply_transform(
    image: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    pass

# Generic types
T = TypeVar("T")
def first_or_none(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Literal for fixed values
def set_mode(mode: Literal["train", "eval", "inference"]) -> None:
    pass
```

### 2.3. Type Aliases

```python
# Define at module level
from typing import TypeAlias

ImageArray: TypeAlias = np.ndarray  # Shape: (H, W, C)
BBoxCoords: TypeAlias = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
ConfigDict: TypeAlias = Dict[str, Any]
```

## 3. Documentation

### 3.1. Module Docstring

```python
"""
Chart detection module using YOLO.

This module provides the Stage2Detection class for detecting chart regions
in document images using a fine-tuned YOLO model.

Example:
    detector = Stage2Detection(config)
    results = detector.process(stage1_output)

Attributes:
    DEFAULT_CONFIDENCE: Default confidence threshold (0.5)
"""

DEFAULT_CONFIDENCE = 0.5
```

### 3.2. Class Docstring

```python
class ChartDetector:
    """
    Detect chart regions in images using YOLO.
    
    This class wraps the Ultralytics YOLO model for chart detection,
    providing preprocessing, inference, and postprocessing.
    
    Attributes:
        model: Loaded YOLO model instance
        config: Detection configuration
        device: Inference device (cpu/cuda/mps)
    
    Example:
        detector = ChartDetector(config)
        boxes = detector.detect("image.png")
        for box in boxes:
            print(f"Found chart at {box.coordinates}")
    """
```

### 3.3. Function Docstring (Google Style)

```python
def detect_charts(
    image_path: Path,
    confidence_threshold: float = 0.5,
    max_detections: int = 10,
) -> List[BoundingBox]:
    """
    Detect chart regions in an image.
    
    Runs YOLO inference on the input image and returns bounding boxes
    for all detected chart regions above the confidence threshold.
    
    Args:
        image_path: Path to the input image file. Supports PNG, JPG, JPEG.
        confidence_threshold: Minimum confidence score for detections.
            Defaults to 0.5. Range: [0.0, 1.0].
        max_detections: Maximum number of detections to return.
            Defaults to 10. Set to -1 for unlimited.
    
    Returns:
        List of BoundingBox objects, sorted by confidence (descending).
        Empty list if no charts detected.
    
    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If confidence_threshold not in [0.0, 1.0].
        ModelNotLoadedError: If YOLO model weights not initialized.
    
    Example:
        >>> boxes = detect_charts(Path("chart.png"), confidence_threshold=0.7)
        >>> print(f"Found {len(boxes)} charts")
        Found 2 charts
    
    Note:
        For multi-page documents, process each page separately.
        Consider using Stage2Detection for batch processing.
    """
```

## 4. Pydantic Models

### 4.1. Base Model Configuration

```python
from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    """Base class for all schema models."""
    
    model_config = ConfigDict(
        frozen=True,  # Immutable
        extra="forbid",  # No extra fields
        validate_assignment=True,
        use_enum_values=True,
    )
```

### 4.2. Field Definitions

```python
from pydantic import BaseModel, Field, field_validator
from typing import Annotated

class BoundingBox(BaseModel):
    """Bounding box with validation."""
    
    x_min: Annotated[int, Field(ge=0, description="Left edge pixel")]
    y_min: Annotated[int, Field(ge=0, description="Top edge pixel")]
    x_max: Annotated[int, Field(gt=0, description="Right edge pixel")]
    y_max: Annotated[int, Field(gt=0, description="Bottom edge pixel")]
    confidence: Annotated[float, Field(ge=0, le=1, description="Detection confidence")]
    
    @field_validator("x_max")
    @classmethod
    def x_max_greater_than_x_min(cls, v: int, info) -> int:
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v
    
    @property
    def area(self) -> int:
        """Compute bounding box area in pixels."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
```

### 4.3. Nested Models

```python
class ChartData(BaseModel):
    """Complete chart data structure."""
    
    chart_id: str = Field(..., min_length=1)
    chart_type: ChartType
    title: Optional[str] = None
    series: List[DataSeries] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def has_data(self) -> bool:
        """Check if chart has any data series."""
        return len(self.series) > 0
```

## 5. Error Handling

### 5.1. Custom Exceptions

```python
# src/core_engine/exceptions.py

class ChartAnalysisError(Exception):
    """Base exception for all chart analysis errors."""
    pass

class PipelineError(ChartAnalysisError):
    """Pipeline execution error."""
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        recoverable: bool = False,
    ):
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(f"[{stage}] {message}" if stage else message)

class ConfigurationError(ChartAnalysisError):
    """Invalid configuration."""
    pass

class ModelError(ChartAnalysisError):
    """Model loading or inference error."""
    pass
```

### 5.2. Error Handling Patterns

```python
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Pattern 1: Specific exception catching
def load_image(path: Path) -> np.ndarray:
    try:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV failed to load image: {path}")
        return image
    except FileNotFoundError:
        logger.error(f"Image file not found: {path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid image format: {e}")
        raise

# Pattern 2: Context manager for cleanup
@contextmanager
def temporary_file(suffix: str = ".png"):
    """Create and clean up temporary file."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        yield Path(path)
    finally:
        os.close(fd)
        Path(path).unlink(missing_ok=True)

# Pattern 3: Retry decorator
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_external_api(url: str) -> dict:
    """Call API with automatic retry."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

### 5.3. Logging

```python
import logging
from pathlib import Path

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """Configure logging for the application."""
    
    handlers: List[logging.Handler] = [
        logging.StreamHandler()  # Console output
    ]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

# Usage in modules
logger = logging.getLogger(__name__)

def process_image(path: Path) -> Result:
    logger.info(f"Processing image: {path}")
    try:
        result = _do_processing(path)
        logger.debug(f"Processing result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Failed to process {path}")
        raise
```

## 6. Testing

### 6.1. Test File Structure

```python
# tests/test_detection.py

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from core_engine.stages import Stage2Detection
from core_engine.schemas import Stage1Output, BoundingBox

# Fixtures
@pytest.fixture
def mock_yolo_model():
    """Create mock YOLO model."""
    mock = Mock()
    mock.predict.return_value = [Mock(boxes=[
        Mock(xyxy=[[10, 10, 100, 100]], conf=[0.9])
    ])]
    return mock

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create sample test image."""
    import numpy as np
    from PIL import Image
    
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    path = tmp_path / "test.png"
    img.save(path)
    return path

# Tests
class TestStage2Detection:
    """Tests for chart detection stage."""
    
    def test_detect_single_chart(self, mock_yolo_model, sample_image):
        """Should detect single chart in image."""
        with patch.object(Stage2Detection, '_load_model', return_value=mock_yolo_model):
            detector = Stage2Detection(config={})
            # ... test logic
    
    def test_handle_no_detections(self, mock_yolo_model, sample_image):
        """Should return empty list when no charts found."""
        mock_yolo_model.predict.return_value = [Mock(boxes=[])]
        # ... test logic
    
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_threshold(self, confidence: float):
        """Should filter by confidence threshold."""
        # ... test logic
```

### 6.2. Test Markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring external resources",
    "gpu: marks tests requiring GPU",
]

# Usage
@pytest.mark.slow
def test_full_pipeline():
    """Full pipeline test (takes ~30s)."""
    pass

@pytest.mark.gpu
def test_yolo_inference():
    """Test requiring CUDA."""
    pass
```

## 7. Performance

### 7.1. Memory Management

```python
import gc
from contextlib import contextmanager

@contextmanager
def memory_efficient_processing():
    """Context manager for memory-intensive operations."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage
with memory_efficient_processing():
    process_large_batch(images)
```

### 7.2. Batch Processing

```python
from typing import Iterator
from itertools import islice

def batch_iterator(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Yield successive batches from items."""
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch

# Usage
for batch in batch_iterator(images, batch_size=16):
    results.extend(process_batch(batch))
```

### 7.3. Caching

```python
from functools import lru_cache
from pathlib import Path
import hashlib

@lru_cache(maxsize=100)
def load_model(model_path: str) -> Model:
    """Load model with caching."""
    return Model.from_pretrained(model_path)

def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash for caching."""
    return hashlib.md5(path.read_bytes()).hexdigest()
```

## 8. Shared Vocabulary Rule (The Rosetta Stone)

### 8.1. Problem Statement

When multiple AI agents (or sessions) work on different modules, they may use inconsistent naming:
- Agent A defines `chart_type = "bar_chart"` (snake_case)
- Agent B writes `if chart.type == "BarChart"` (PascalCase)
- Result: Silent bugs that pass linting but fail at runtime

### 8.2. The Solution: Single Source of Truth

All Enums, Constants, and shared values MUST be defined in ONE location:

**Python Source:** `src/core_engine/schemas/enums.py`
**Reference Doc:** `.github/instructions/references/shared_vocabulary.md`

### 8.3. Enforcement Rules

**[CRITICAL] NEVER hard-code string literals in processing logic:**

```python
# [FORBIDDEN] Hard-coded string literal
if chart.chart_type == "bar":
    process_bar_chart(chart)

# [CORRECT] Import from enums
from core_engine.schemas.enums import ChartType

if chart.chart_type == ChartType.BAR:
    process_bar_chart(chart)
```

**[CRITICAL] Always import from enums.py:**

```python
# [FORBIDDEN] Redefining enum locally
class ChartType(str, Enum):
    BAR = "bar"  # Duplicating definition

# [CORRECT] Import the shared definition
from core_engine.schemas.enums import ChartType
```

### 8.4. What Belongs in enums.py

| Category | Examples |
| --- | --- |
| Chart Types | `ChartType.BAR`, `ChartType.LINE` |
| Pipeline Status | `StageStatus.PENDING`, `PipelineStatus.COMPLETED` |
| Text Roles | `TextRole.TITLE`, `TextRole.LEGEND` |
| Element Types | `ElementType.BAR`, `ElementType.POINT` |
| Error Codes | `ErrorCode.S1_FILE_NOT_FOUND` |
| Thresholds | `ConfidenceThreshold.DETECTION_MIN` |

### 8.5. Cross-Language Reference

When working with Frontend (TypeScript/JavaScript), agents MUST:

1. Read `.github/instructions/references/shared_vocabulary.md` first
2. Use exact same string values as Python enums
3. Create TypeScript types that mirror Python enums:

```typescript
// Must match Python ChartType enum values exactly
type ChartType = "bar" | "line" | "pie" | "scatter" | "area" | "unknown";
```

### 8.6. Pre-Code Checklist

Before writing any code that uses shared values:

1. Check if the value exists in `enums.py`
2. If not, add it to `enums.py` first
3. Update `shared_vocabulary.md` if adding new categories
4. Import from enums, never hard-code
