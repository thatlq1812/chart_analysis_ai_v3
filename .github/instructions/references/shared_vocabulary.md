# SHARED VOCABULARY & CONSTANTS

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-01-19 | That Le | Cross-language reference for all agents |

## Purpose

This file serves as the **Single Source of Truth** for naming conventions and allowed values across all agents (Backend, Frontend, Testing). When working on this project, ALL agents MUST refer to this document to ensure consistency.

**Python Implementation:** `src/core_engine/schemas/enums.py`

## 1. Chart Types

| Concept | Variable Name | Allowed Values | Notes |
| --- | --- | --- | --- |
| Chart Type | `chart_type` | `bar`, `line`, `pie`, `scatter`, `area`, `histogram`, `stacked_bar`, `grouped_bar`, `donut`, `unknown` | Use lowercase, underscore for multi-word |

```python
# Python
from core_engine.schemas.enums import ChartType
chart_type = ChartType.BAR  # "bar"
```

```typescript
// TypeScript
type ChartType = "bar" | "line" | "pie" | "scatter" | "area" | "histogram" | "stacked_bar" | "grouped_bar" | "donut" | "unknown";
```

## 2. Pipeline Status

| Concept | Variable Name | Allowed Values | Description |
| --- | --- | --- | --- |
| Stage Status | `status` | `pending`, `processing`, `completed`, `failed`, `skipped` | Individual stage status |
| Pipeline Status | `pipeline_status` | `idle`, `running`, `completed`, `partial`, `failed`, `cancelled` | Overall pipeline status |

## 3. Text Roles (OCR Classification)

| Role | Value | Location in Chart |
| --- | --- | --- |
| Title | `title` | Top center |
| X Axis Label | `x_axis_label` | Below X axis |
| Y Axis Label | `y_axis_label` | Left of Y axis |
| X Tick Labels | `x_tick` | On X axis |
| Y Tick Labels | `y_tick` | On Y axis |
| Legend | `legend` | Usually top-right or bottom |
| Data Label | `data_label` | On/near data points |
| Annotation | `annotation` | Anywhere |

## 4. Element Types

| Element | Value | Used In |
| --- | --- | --- |
| Bar | `bar` | Bar charts |
| Line | `line` | Line charts |
| Point | `point` | Scatter, line charts |
| Slice | `slice` | Pie, donut charts |
| Area | `area` | Area charts |

## 5. Insight Types

| Insight | Value | Example |
| --- | --- | --- |
| Trend | `trend` | "Values show increasing trend from Q1 to Q4" |
| Comparison | `comparison` | "Product A leads with 45% market share" |
| Anomaly | `anomaly` | "Q3 shows unusual spike of 150% vs average" |
| Summary | `summary` | "Chart displays quarterly sales data" |
| Correlation | `correlation` | "Strong positive correlation between X and Y" |

## 6. File Types

### Input Formats
| Format | Value | MIME Type |
| --- | --- | --- |
| PDF | `pdf` | application/pdf |
| DOCX | `docx` | application/vnd.openxmlformats... |
| PNG | `png` | image/png |
| JPG/JPEG | `jpg`, `jpeg` | image/jpeg |

### Output Formats
| Format | Value | Use Case |
| --- | --- | --- |
| JSON | `json` | API response, structured data |
| Markdown | `markdown` | Human-readable reports |
| HTML | `html` | Web display |
| CSV | `csv` | Data export |

## 7. Confidence Thresholds

| Threshold | Variable | Value | Usage |
| --- | --- | --- | --- |
| Detection Min | `DETECTION_MIN` | 0.5 | Minimum to accept chart detection |
| Detection High | `DETECTION_HIGH` | 0.8 | High confidence detection |
| OCR Min | `OCR_MIN` | 0.6 | Minimum to accept OCR text |
| OCR High | `OCR_HIGH` | 0.9 | High confidence OCR |
| Value Extraction | `VALUE_EXTRACTION` | 0.7 | Value mapping confidence |

## 8. Quality Levels

| Level | Value | Confidence Range |
| --- | --- | --- |
| High | `high` | >= 0.8 |
| Medium | `medium` | 0.5 - 0.8 |
| Low | `low` | < 0.5 |
| Invalid | `invalid` | Failed validation |

## 9. Error Codes

Format: `STAGE_ERROR_TYPE` (e.g., `s1_file_not_found`)

| Stage | Error Code | Description |
| --- | --- | --- |
| S1 | `s1_file_not_found` | Input file does not exist |
| S1 | `s1_unsupported_format` | File type not supported |
| S1 | `s1_corrupted_file` | File is corrupted |
| S1 | `s1_low_quality` | Image quality too low |
| S2 | `s2_model_not_loaded` | YOLO model failed to load |
| S2 | `s2_no_detections` | No charts detected |
| S2 | `s2_inference_failed` | Model inference error |
| S3 | `s3_ocr_failed` | OCR extraction failed |
| S3 | `s3_classification_failed` | Chart type classification failed |
| S4 | `s4_slm_not_loaded` | SLM model failed to load |
| S4 | `s4_reasoning_timeout` | SLM inference timeout |
| S5 | `s5_validation_failed` | Output schema validation failed |

## 10. Bounding Box Format

| Format | Structure | Example |
| --- | --- | --- |
| XYXY | `[x_min, y_min, x_max, y_max]` | `[100, 50, 300, 200]` |
| XYWH | `[x, y, width, height]` | `[100, 50, 200, 150]` |

**Standard:** Use XYXY format internally. Convert to XYWH only when required by external libraries.

## 10.1. Color Format Standard

| Library | Native Format | Example |
| --- | --- | --- |
| OpenCV | BGR | `(255, 0, 0)` = Blue |
| PIL/Pillow | RGB | `(255, 0, 0)` = Red |
| Web/CSS | Hex | `#FF0000` = Red |
| Matplotlib | RGB (0-1) | `(1.0, 0, 0)` = Red |

**[CRITICAL] Internal Standard: RGB (0-255)**

```python
# Core Engine uses RGB internally
from core_engine.schemas import Color
color = Color(r=255, g=0, b=0)  # Red

# Convert when interfacing with OpenCV
cv2_color = (color.b, color.g, color.r)  # BGR for OpenCV

# Convert for web
hex_color = color.hex  # "#ff0000"
```

**Conversion Rules:**
- Store colors as `Color(r, g, b)` in all schemas
- Convert to BGR only at OpenCV function calls
- Convert to Hex only at API response/web rendering

## 11. Naming Conventions

| Context | Convention | Example |
| --- | --- | --- |
| Variable names | snake_case | `chart_type`, `bbox_confidence` |
| Class names | PascalCase | `ChartType`, `BoundingBox` |
| Constants | UPPER_SNAKE | `DETECTION_MIN`, `MAX_IMAGE_SIZE` |
| Enum values | lowercase | `"bar"`, `"pending"` |
| File names | snake_case | `stage_outputs.py` |
| Config keys | snake_case | `confidence_threshold` |

## 12. API Response Structure

```json
{
  "session_id": "string",
  "status": "completed | partial | failed",
  "charts": [
    {
      "chart_id": "string",
      "chart_type": "bar | line | pie | ...",
      "data": { ... },
      "insights": [ ... ]
    }
  ],
  "errors": [
    {
      "code": "s2_no_detections",
      "message": "string",
      "stage": "detection"
    }
  ],
  "processing_time_seconds": 0.0
}
```

## Enforcement Rules

1. **Python Agents:** MUST import from `core_engine.schemas.enums`
2. **TypeScript Agents:** MUST reference this document for type definitions
3. **Test Agents:** MUST use enums for assertions, not string literals
4. **NO HARDCODING:** Never write `"bar"` directly in logic code, always use `ChartType.BAR`
