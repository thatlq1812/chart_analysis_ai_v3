# Development Guide

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-25 | That Le | Development setup and contribution guide |

## 1. Development Environment Setup

### 1.1. Prerequisites

| Tool | Version | Purpose |
| --- | --- | --- |
| Python | 3.11+ | Runtime |
| Git | Latest | Version control |
| VS Code | Latest | Recommended IDE |
| uv | Latest | Fast package manager |

### 1.2. IDE Setup (VS Code)

**Recommended extensions:**
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter
- Ruff
- GitLens

**settings.json:**
```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

### 1.3. Install Development Dependencies

```bash
# Clone and enter project
git clone <repo_url>
cd chart_analysis_ai_v3

# Create virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## 2. Project Structure

```
chart_analysis_ai_v3/
|
+-- src/core_engine/         # Main library (EDIT HERE)
|   +-- __init__.py          # Public exports
|   +-- pipeline.py          # Pipeline orchestrator
|   +-- exceptions.py        # Custom exceptions
|   +-- schemas/             # Pydantic models
|   +-- stages/              # Stage implementations
|   +-- validators/          # Input validators
|
+-- tests/                   # Test suite (MUST UPDATE)
|   +-- conftest.py          # Shared fixtures
|   +-- test_schemas.py      # Schema tests
|   +-- test_s3_extraction/  # Stage 3 tests
|
+-- scripts/                 # Utility scripts
+-- notebooks/               # Jupyter notebooks
+-- config/                  # Configuration files
```

## 3. Coding Standards

### 3.1. Style Guide

All code must follow standards defined in `.github/instructions/coding-standards.instructions.md`:

- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff
- **Type Checker**: MyPy (strict mode)

### 3.2. Type Hints (Required)

```python
# Bad
def process(data):
    return data.transform()

# Good
def process(data: RawChartData) -> ProcessedChartData:
    return data.transform()
```

### 3.3. Docstrings (Google Style)

```python
def detect_charts(
    image_path: Path,
    confidence_threshold: float = 0.5,
) -> List[DetectedChart]:
    """
    Detect chart regions in an image.

    Args:
        image_path: Path to input image file
        confidence_threshold: Minimum confidence for detection

    Returns:
        List of detected charts with bounding boxes

    Raises:
        FileNotFoundError: If image does not exist
        ModelError: If detection model fails
    """
```

### 3.4. Import Organization

```python
# 1. Standard library
import logging
from pathlib import Path
from typing import List, Optional

# 2. Third-party
import numpy as np
from pydantic import BaseModel

# 3. Local imports
from .schemas import ChartResult
from .stages import BaseStage
```

## 4. Testing

### 4.1. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schemas.py -v

# Run with coverage
pytest tests/ --cov=src/core_engine --cov-report=html

# Run specific test
pytest tests/test_s3_extraction/test_preprocessor.py::test_negative_transform -v
```

### 4.2. Writing Tests

```python
# tests/test_example.py
import pytest
from core_engine.stages import Stage3Extraction

@pytest.fixture
def stage3():
    """Create Stage3 instance for testing."""
    return Stage3Extraction()

def test_process_bar_chart(stage3, sample_bar_chart):
    """Test processing of bar chart image."""
    result = stage3.process_single_image(sample_bar_chart)
    
    assert result is not None
    assert result.chart_type == "bar"
    assert len(result.elements) > 0
```

### 4.3. Test Fixtures

Shared fixtures are in `tests/conftest.py`:

```python
@pytest.fixture
def sample_chart_image(tmp_path):
    """Create a sample chart image for testing."""
    # Create test image
    image = create_test_chart()
    path = tmp_path / "test_chart.png"
    cv2.imwrite(str(path), image)
    return path
```

## 5. Making Changes

### 5.1. Workflow

1. **Create branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Edit code, add tests
3. **Run tests**: `pytest tests/ -v`
4. **Format code**: `black src/ tests/`
5. **Lint**: `ruff check src/ tests/`
6. **Commit**: `git commit -m "feat(stage3): add new feature"`
7. **Push**: `git push origin feature/your-feature`

### 5.2. Commit Message Format

```
<type>(<scope>): <description>

[optional body]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Build, dependencies

**Examples:**
```
feat(stage3): add bar chart detection
fix(ocr): handle empty text results
docs(readme): update installation steps
refactor(pipeline): extract common validation
```

## 6. Adding a New Stage

### 6.1. Create Stage File

```python
# src/core_engine/stages/s4_reasoning.py
from typing import List
from .base import BaseStage
from ..schemas import Stage3Output, Stage4Output

class Stage4Reasoning(BaseStage[Stage3Output, Stage4Output]):
    """
    Stage 4: Apply SLM reasoning to refine extracted data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._load_models()
    
    def process(self, input_data: Stage3Output) -> Stage4Output:
        """Process Stage 3 output through SLM reasoning."""
        refined_charts = []
        for metadata in input_data.metadata:
            refined = self._reason_chart(metadata)
            refined_charts.append(refined)
        
        return Stage4Output(
            session=input_data.session,
            charts=refined_charts
        )
    
    def validate_input(self, input_data: Stage3Output) -> bool:
        """Validate Stage 3 output before processing."""
        return input_data is not None and len(input_data.metadata) > 0
```

### 6.2. Add Tests

```python
# tests/test_stages/test_s4_reasoning.py
import pytest
from core_engine.stages import Stage4Reasoning

class TestStage4Reasoning:
    def test_process_valid_input(self, stage3_output):
        stage4 = Stage4Reasoning()
        result = stage4.process(stage3_output)
        
        assert result is not None
        assert len(result.charts) == len(stage3_output.metadata)
```

### 6.3. Export from Package

```python
# src/core_engine/stages/__init__.py
from .s4_reasoning import Stage4Reasoning

__all__ = [
    # ... existing exports
    "Stage4Reasoning",
]
```

## 7. Documentation

### 7.1. Updating Docs

When making changes:
1. Update relevant architecture doc in `docs/architecture/`
2. Update `docs/MASTER_CONTEXT.md` if status changes
3. Add entry to `CHANGELOG.md`

### 7.2. Documentation Standards

See `.github/instructions/docs.instructions.md`:
- Every file needs version header table
- Use mermaid for diagrams
- No emojis - use `[DONE]`, `[TODO]`, `[IN PROGRESS]`
- Code blocks must specify language

## 8. Common Tasks

### 8.1. Adding a New Dependency

```bash
# Add to pyproject.toml
# [project.dependencies] section

# Install
pip install -e .
```

### 8.2. Running Scripts

```bash
# Generate reports
python scripts/generate_stage3_report.py

# Test on academic dataset
python scripts/test_stage3_academic_dataset.py

# Benchmark classifier
python scripts/benchmark_classifier.py
```

### 8.3. Working with Notebooks

```bash
# Start Jupyter
jupyter lab

# Convert notebook to script
jupyter nbconvert --to python notebooks/01_exploration.ipynb
```

## 9. Troubleshooting

### 9.1. Import Errors

```bash
# Reinstall in editable mode
pip install -e ".[dev]"

# Check Python path
python -c "import sys; print(sys.path)"
```

### 9.2. Test Failures

```bash
# Run with verbose output
pytest tests/ -v --tb=long

# Run specific failing test
pytest tests/test_file.py::test_name -v -s
```

### 9.3. Type Errors

```bash
# Run mypy
mypy src/core_engine/

# Ignore specific error (if justified)
# Add: # type: ignore[error-code]
```

## 10. Resources

- **Coding Standards**: `.github/instructions/coding-standards.instructions.md`
- **Pipeline Design**: `.github/instructions/pipeline.instructions.md`
- **Architecture**: `docs/architecture/`
- **API Reference**: Docstrings in source code
