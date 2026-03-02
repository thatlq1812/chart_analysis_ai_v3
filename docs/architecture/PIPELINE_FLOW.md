# Pipeline Flow

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 3.0.0 | 2026-03-02 | That Le | Full rewrite: all 5 stages complete, AI Router, actual schemas |
| 2.0.0 | 2026-02-04 | That Le | Updated pipeline status |

## Implementation Status

| Stage | Status | Key Class | Notes |
| --- | --- | --- | --- |
| Stage 1 | **Complete** | `Stage1Ingestion` | PDF/Image ingestion |
| Stage 2 | **Complete** | `Stage2Detection` | YOLO 93.5% mAP@50 |
| Stage 3 | **Complete** | `Stage3Extraction` (11 submodules) | OCR + Geometry + Classification |
| Stage 4 | **Complete** | `Stage4Reasoning` (6 submodules) | AI Router + 3 adapters |
| Stage 5 | **Complete** | `Stage5Reporting` | Insights, validation, reports |
| AI Router | **Complete** | `AIRouter` (8 files, 55 tests) | Multi-provider fallback |
| Pipeline | **Complete** | `ChartAnalysisPipeline` | All 5 stages wired |

---

## 1. Pipeline Overview

The Geo-SLM Chart Analysis pipeline processes input documents through 5 sequential stages, orchestrated by `ChartAnalysisPipeline` in `src/core_engine/pipeline.py`:

```mermaid
flowchart LR
    A[Input\nPDF/Image] --> S1[Stage 1\nIngestion]
    S1 --> S2[Stage 2\nDetection]
    S2 --> S3[Stage 3\nExtraction]
    S3 --> S4[Stage 4\nReasoning]
    S4 --> S5[Stage 5\nReporting]
    S5 --> B[Output\nJSON + Report]

    style S1 fill:#e3f2fd
    style S2 fill:#e8f5e9
    style S3 fill:#fff3e0
    style S4 fill:#fce4ec
    style S5 fill:#f3e5f5
```

**Orchestration:**
```python
from core_engine import ChartAnalysisPipeline

pipeline = ChartAnalysisPipeline.from_config()  # Loads base.yaml + models.yaml + pipeline.yaml
result = pipeline.run("report.pdf")              # Returns PipelineResult
```

Each stage is a `BaseStage[InputT, OutputT]` subclass with typed I/O schemas enforced by Pydantic v2.

---

## 2. Stage 1: Ingestion & Sanitation

### 2.1. Purpose

Transform diverse input formats into normalized images ready for processing.

### 2.2. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Input Files"]
        A1[PDF Document]
        A2[DOCX Document]
        A3[PNG/JPG Image]
    end

    subgraph Detection["File Type Detection"]
        B[Detect File Type]
    end

    subgraph Processing["Processing Branch"]
        C1[PDF Processor\nPyMuPDF]
        C2[DOCX Processor\npython-docx]
        C3[Image Loader\nPillow]
    end

    subgraph Conversion["Page Conversion"]
        D[Convert Pages\nto Images]
    end

    subgraph Validation["Quality Validation"]
        E1[Check Resolution\nmin 72 DPI]
        E2[Check Blur\nLaplacian variance]
        E3[Check Size\nmax 4096px]
    end

    subgraph Normalization["Normalization"]
        F1[Resize if needed]
        F2[Enhance Contrast]
        F3[Generate Session ID]
    end

    subgraph Output["Stage 1 Output"]
        G[Stage1Output\nList of CleanImage]
    end

    A1 --> B
    A2 --> B
    A3 --> B

    B -->|PDF| C1
    B -->|DOCX| C2
    B -->|Image| C3

    C1 --> D
    C2 --> D
    C3 --> D

    D --> E1 --> E2 --> E3

    E3 -->|Pass| F1 --> F2 --> F3
    E3 -->|Fail| H[Skip with Warning]

    F3 --> G
    H --> G
```

### 2.3. Input/Output Schema

```python
# Input
input_path: Path  # Path to PDF, DOCX, or image file

# Output (src/core_engine/schemas/stage_outputs.py)
class Stage1Output(BaseModel):
    session: SessionInfo
    images: List[CleanImage]
    warnings: List[str]

class CleanImage(BaseModel):
    image_path: Path
    original_path: Path
    page_number: int
    width: int
    height: int
    is_grayscale: bool
```

---

## 3. Stage 2: Detection & Localization

### 3.1. Purpose

Detect and crop chart regions from document images using YOLO.

### 3.2. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 1 Output"]
        A[Stage1Output\nList of CleanImage]
    end

    subgraph ModelLoad["Model Loading"]
        B[Load YOLO Model\nSingleton Pattern]
    end

    subgraph Processing["Per-Image Processing"]
        C[Load Image]
        D[YOLO Inference]
        E[NMS Filtering]
        F[Confidence Filtering\nthreshold=0.5]
    end

    subgraph Cropping["Chart Cropping"]
        G{Multiple\nCharts?}
        H1[Crop Single Chart]
        H2[Crop Each Chart\nwith Unique ID]
    end

    subgraph Saving["Save Results"]
        I[Save Cropped Images]
        J[Record BBox Info]
    end

    subgraph Output["Stage 2 Output"]
        K[Stage2Output\nList of DetectedChart]
    end

    A --> B --> C --> D --> E --> F

    F --> G
    G -->|No| H1
    G -->|Yes| H2

    H1 --> I
    H2 --> I

    I --> J --> K
```

### 3.3. Multi-Chart Handling

```mermaid
flowchart LR
    subgraph OriginalImage["Original Image (1 page)"]
        A[Page with\n3 Charts]
    end

    subgraph Detection["YOLO Detection"]
        B1[Chart 1\nBBox]
        B2[Chart 2\nBBox]
        B3[Chart 3\nBBox]
    end

    subgraph CroppedCharts["Cropped Results"]
        C1[chart_001.png]
        C2[chart_002.png]
        C3[chart_003.png]
    end

    A --> B1 --> C1
    A --> B2 --> C2
    A --> B3 --> C3
```

### 3.4. Input/Output Schema

```python
# Output (src/core_engine/schemas/stage_outputs.py)
class Stage2Output(BaseModel):
    session: SessionInfo
    charts: List[DetectedChart]
    total_detected: int
    skipped_low_confidence: int

class DetectedChart(BaseModel):
    chart_id: str
    source_image: Path
    cropped_path: Path
    bbox: BoundingBox
    page_number: int
```

---

## 4. Stage 3: Structural Analysis (Hybrid)

### 4.1. Purpose

Extract raw metadata through parallel OCR, element detection, classification, and geometric analysis. This is the most complex stage with 11 submodules.

### 4.2. Submodule Architecture

| Submodule | File | Purpose |
| --- | --- | --- |
| `Stage3Extraction` | `s3_extraction.py` | Orchestrator |
| `ResNetClassifier` | `resnet_classifier.py` | Chart type classification (94.14% acc) |
| `MLClassifier` | `ml_classifier.py` | Fallback ML classifier |
| `SimpleClassifier` | `simple_classifier.py` | Rule-based fallback |
| `ElementDetector` | `element_detector.py` | Bar/point/slice/line detection |
| `GeometricMapper` | `geometric_mapper.py` | Axis scale + value mapping |
| `OCREngine` | `ocr_engine.py` | PaddleOCR text extraction |
| `Preprocessor` | `preprocessor.py` | Image preprocessing |
| `Skeletonizer` | `skeletonizer.py` | Morphological analysis |
| `Vectorizer` | `vectorizer.py` | Contour vectorization |
| `Classifier` | `classifier.py` | Classifier dispatcher |

### 4.3. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 2 Output"]
        A[DetectedChart\nCropped Image]
    end

    subgraph Parallel["Parallel Processing"]
        subgraph Classification["Chart Classification"]
            B1[ResNet-18\n94.14% accuracy]
            B2[Fallback: ML or Rule-based]
        end

        subgraph OCR["Text Extraction"]
            C1[PaddleOCR]
            C2[Text Detection]
            C3[Role Assignment\ntitle/label/legend/value/tick]
        end

        subgraph Elements["Element Detection"]
            D1[Color Analysis]
            D2[Contour Detection]
            D3[Element Classification\nbar/point/slice/line]
        end
    end

    subgraph Geometric["Geometric Analysis"]
        E1[Detect Axes]
        E2[Extract Scale]
        E3[Map Coordinates\nto Values]
    end

    subgraph Assembly["Data Assembly"]
        F[Combine into RawMetadata\nwith ExtractionConfidence]
    end

    subgraph Output["Stage 3 Output"]
        G[Stage3Output\nList of RawMetadata]
    end

    A --> B1
    B1 -->|low conf| B2
    A --> C1 --> C2 --> C3
    A --> D1 --> D2 --> D3

    B1 --> E1
    B2 --> E1
    C3 --> E1
    D3 --> E1

    E1 --> E2 --> E3 --> F --> G
```

### 4.4. OCR Text Role Detection

Roles are defined in the `TextRole` enum:

```mermaid
flowchart TB
    subgraph Chart["Chart Image Regions"]
        A[Top: TITLE / SUBTITLE]
        B[Left: Y_AXIS_LABEL / Y_TICK]
        C[Bottom: X_AXIS_LABEL / X_TICK]
        D[Corner: LEGEND / LEGEND_ITEM]
        E[On elements: DATA_LABEL]
        F[Other: ANNOTATION]
    end

    subgraph Roles["TextRole Enum"]
        R1[TITLE]
        R2[SUBTITLE]
        R3[X_AXIS_LABEL]
        R4[Y_AXIS_LABEL]
        R5[X_TICK]
        R6[Y_TICK]
        R7[LEGEND]
        R8[DATA_LABEL]
        R9[ANNOTATION]
        R10[UNKNOWN]
    end
```

### 4.5. Input/Output Schema

```python
# Output (src/core_engine/schemas/stage_outputs.py)
class Stage3Output(BaseModel):
    session: SessionInfo
    metadata: List[RawMetadata]

class RawMetadata(BaseModel):
    chart_id: str
    chart_type: ChartType        # 12 types in enum
    texts: List[OCRText]
    elements: List[ChartElement]
    axis_info: Optional[AxisInfo]
    confidence: ExtractionConfidence

class OCRText(BaseModel):
    text: str
    bbox: BoundingBox
    confidence: float
    role: Optional[str]          # TextRole value

class ChartElement(BaseModel):
    element_type: str            # ElementType value
    bbox: BoundingBox
    center: Point
    color: Optional[Color]
    area_pixels: Optional[int]
```

---

## 5. Stage 4: Semantic Reasoning (AI Router)

### 5.1. Purpose

Apply AI reasoning to correct OCR errors, map geometric values, and generate descriptions. Uses multi-provider routing with automatic fallback.

### 5.2. Submodule Architecture

| Component | File | Purpose |
| --- | --- | --- |
| `Stage4Reasoning` | `s4_reasoning.py` (479 lines) | Orchestrator |
| `GeometricValueMapper` | `value_mapper.py` (764 lines) | Pixel-to-value mapping |
| `GeminiPromptBuilder` | `prompt_builder.py` (833 lines) | Structured prompt construction |
| `ReasoningEngine` | `reasoning_engine.py` (185 lines) | Abstract base for engines |
| `GeminiReasoningEngine` | `gemini_engine.py` (626 lines) | Direct Gemini API engine |
| `AIRouterEngine` | `router_engine.py` (410 lines) | Multi-provider via AIRouter |
| Prompt templates | `prompts/*.txt, *.md` | 5 template files |

### 5.3. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 3 Output"]
        A[RawMetadata\nOCR + Elements + Geometry]
    end

    subgraph GeometricMapping["Geometric Value Mapping"]
        B1[Calculate Axis Scale\nlinear/log]
        B2[Map Pixel to Value\ninverted Y-axis]
        B3[Initial Value Estimates\nwith confidence scores]
    end

    subgraph PromptConstruction["Prompt Building"]
        C1[GeminiPromptBuilder]
        C2[Build CanonicalContext\nchart_type + OCR texts + values]
        C3[Select prompt template\nreasoning/ocr/description/value]
    end

    subgraph EngineRouting["Engine Selection"]
        D0{ReasoningConfig\n.engine}
        D1[GeminiReasoningEngine]
        D2[AIRouterEngine\nMulti-provider fallback]
        D3[Rule-based fallback]
    end

    subgraph AIRouter["AI Router (router engine)"]
        E1[Resolve provider\nfor TaskType]
        E2[Walk fallback chain\nlocal_slm -> gemini -> openai]
        E3[Confidence check\nthreshold=0.7]
    end

    subgraph PostProcess["Post-Processing"]
        F1[Apply OCR corrections\nloo -> 100]
        F2[Refine values]
        F3[Map legend to colors]
        F4[Generate description]
    end

    subgraph Output["Stage 4 Output"]
        G[Stage4Output\nList of RefinedChartData]
    end

    A --> B1 --> B2 --> B3
    B3 --> C1 --> C2 --> C3

    C3 --> D0
    D0 -->|"gemini"| D1
    D0 -->|"router"| D2
    D0 -->|"rule_based"| D3

    D2 --> E1 --> E2 --> E3

    D1 --> F1
    E3 --> F1
    D3 --> F1

    F1 --> F2 --> F3 --> F4 --> G
```

### 5.4. AI Router Task Types and Fallback Chains

```python
class TaskType(str, Enum):
    CHART_REASONING = "chart_reasoning"     # Full analysis: OCR + values + description
    OCR_CORRECTION = "ocr_correction"       # Fix OCR misreads
    DESCRIPTION_GEN = "description_gen"     # Academic-style description
    DATA_VALIDATION = "data_validation"     # Validate extracted data
```

| Task Type | Default Fallback Chain |
| --- | --- |
| `CHART_REASONING` | local_slm -> gemini -> openai |
| `OCR_CORRECTION` | local_slm -> gemini |
| `DESCRIPTION_GEN` | local_slm -> gemini -> openai |
| `DATA_VALIDATION` | gemini -> openai |

### 5.5. Adapter Architecture

```mermaid
flowchart TB
    subgraph Router["AIRouter"]
        R[resolve task_type\nwalk fallback chain]
    end

    subgraph Adapters["Concrete Adapters"]
        A1[LocalSLMAdapter\nQwen/Llama LoRA]
        A2[GeminiAdapter\ngemini-2.0-flash]
        A3[OpenAIAdapter\ngpt-4o-mini]
    end

    subgraph Base["BaseAIAdapter ABC"]
        B1["reason(system_prompt, user_prompt, ...) -> AIResponse"]
        B2["health_check() -> bool"]
        B3["get_default_model() -> str"]
    end

    Router --> A1
    Router --> A2
    Router --> A3

    A1 -.-> Base
    A2 -.-> Base
    A3 -.-> Base
```

### 5.6. Input/Output Schema

```python
# Output (src/core_engine/schemas/stage_outputs.py)
class Stage4Output(BaseModel):
    session: SessionInfo
    charts: List[RefinedChartData]

class RefinedChartData(BaseModel):
    chart_id: str
    chart_type: ChartType
    title: Optional[str]
    x_axis_label: Optional[str]
    y_axis_label: Optional[str]
    series: List[DataSeries]
    description: str
    correction_log: List[str]

class DataSeries(BaseModel):
    name: str
    color: Optional[Color]
    points: List[DataPoint]

class DataPoint(BaseModel):
    label: str
    value: float
    unit: Optional[str]
    confidence: float
```

---

## 6. Stage 5: Insight & Reporting

### 6.1. Purpose

Validate refined data, generate insights (trend, comparison, anomaly, summary), and produce final structured output in multiple formats.

### 6.2. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 4 Output"]
        A[RefinedChartData]
    end

    subgraph Validation["Schema Validation"]
        B1[Check empty chart_id]
        B2[Check missing description]
        B3[Check empty series]
        B4[Check NaN/Inf values]
        B5[Flag low-confidence points]
    end

    subgraph Insights["Insight Generation (4 types)"]
        C1[Summary\nChart description or auto-generated]
        C2[Trend\nLinear regression slope per series]
        C3[Comparison\nSeries maxima + dominance ratio]
        C4[Anomaly\nZ-score flagging]
    end

    subgraph Formatting["Output Formatting"]
        D1[Build PipelineResult]
        D2[Add processing time + model versions]
        D3[Generate text summary]
    end

    subgraph Output["Final Output"]
        E1[PipelineResult JSON]
        E2[Text Report]
        E3[Markdown Report]
    end

    A --> B1 --> B2 --> B3 --> B4 --> B5
    B5 --> C1 --> C2 --> C3 --> C4
    C4 --> D1 --> D2 --> D3
    D3 --> E1
    D3 --> E2
    D3 --> E3
```

### 6.3. Insight Types

| Type | Detection Method | Example |
| --- | --- | --- |
| `summary` | Chart description or auto-generated from metadata | "Bar chart showing quarterly revenue for 2025" |
| `trend` | Linear regression slope per data series | "Values show increasing trend from Q1 to Q4" |
| `comparison` | Compare series maxima, find dominant series + ratio | "Product A leads with 45% market share" |
| `anomaly` | Z-score method, flag |z| > threshold | "Q3 shows unusual spike of 150% vs average" |

```python
class InsightType(str, Enum):
    TREND = "trend"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"
    SUMMARY = "summary"
    CORRELATION = "correlation"  # Future use
```

### 6.4. Final Output Schema

```python
# Output (src/core_engine/schemas/stage_outputs.py)
class PipelineResult(BaseModel):
    session: SessionInfo
    charts: List[FinalChartResult]
    summary: str
    processing_time_seconds: float
    model_versions: Dict[str, str]
    warnings: List[str]

class FinalChartResult(BaseModel):
    chart_id: str
    chart_type: ChartType
    title: Optional[str]
    data: RefinedChartData
    insights: List[ChartInsight]
    source_info: Dict[str, Any]

class ChartInsight(BaseModel):
    insight_type: str      # InsightType value
    text: str
    confidence: float
```

---

## 7. Full Pipeline Sequence

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant S1 as Stage 1
    participant S2 as Stage 2
    participant S3 as Stage 3
    participant S4 as Stage 4
    participant S5 as Stage 5
    participant YOLO
    participant OCR
    participant Router as AI Router

    User->>Pipeline: run("report.pdf")

    Note over Pipeline,S1: Stage 1: Ingestion
    Pipeline->>S1: process(path)
    S1->>S1: Load PDF (PyMuPDF)
    S1->>S1: Convert to images
    S1->>S1: Validate quality
    S1-->>Pipeline: Stage1Output (N images)

    Note over Pipeline,S2: Stage 2: Detection
    Pipeline->>S2: process(Stage1Output)
    loop For each image
        S2->>YOLO: detect(image)
        YOLO-->>S2: bboxes (conf > 0.5)
        S2->>S2: Crop charts
    end
    S2-->>Pipeline: Stage2Output (M charts)

    Note over Pipeline,S3: Stage 3: Extraction
    Pipeline->>S3: process(Stage2Output)
    loop For each chart
        par Parallel Analysis
            S3->>S3: ResNet-18 classify (94.14%)
            S3->>OCR: PaddleOCR extract
            OCR-->>S3: texts + roles
            S3->>S3: Element detection
        end
        S3->>S3: Geometric value mapping
    end
    S3-->>Pipeline: Stage3Output

    Note over Pipeline,S4: Stage 4: Reasoning
    Pipeline->>S4: process(Stage3Output)
    loop For each metadata
        S4->>S4: GeometricValueMapper
        S4->>S4: GeminiPromptBuilder
        S4->>Router: resolve(CHART_REASONING)
        Router->>Router: Walk chain: local_slm -> gemini -> openai
        Router-->>S4: AIResponse
        S4->>S4: Apply corrections
    end
    S4-->>Pipeline: Stage4Output

    Note over Pipeline,S5: Stage 5: Reporting
    Pipeline->>S5: process(Stage4Output)
    S5->>S5: Validate schemas
    S5->>S5: Generate insights (trend/comparison/anomaly/summary)
    S5->>S5: Format output (JSON + text + markdown)
    S5-->>Pipeline: PipelineResult

    Pipeline-->>User: PipelineResult (JSON + Report)
```

---

## 8. Error Recovery

### 8.1. Exception Hierarchy

```
ChartAnalysisError (base)
    PipelineError (stage, recoverable, original_error)
        StageInputError (expected_type, received_type)
        StageProcessingError (fallback_available)
    ConfigurationError (config_key)
    ModelError

AIProviderError (AI-specific base)
    AIRateLimitError (retry_after)
    AIAuthenticationError
    AITimeoutError (timeout_seconds)
    AIInvalidResponseError (raw_response)
AIProviderExhaustedError (all providers failed)
```

### 8.2. Recovery Flow

```mermaid
flowchart TD
    A[Stage Error] --> B{Recoverable?}

    B -->|Yes| C[Log Warning]
    C --> D[Get Fallback Output]
    D --> E[Continue Pipeline]

    B -->|No| F[Log Error]
    F --> G[Raise PipelineError]
    G --> H[Return Partial Result]
```

### 8.3. Error Types by Stage

| Stage | Error Type | Recovery Strategy |
| --- | --- | --- |
| S1 | File not found | Abort |
| S1 | Low quality image | Skip with warning |
| S2 | No detections | Return empty list |
| S2 | Model load failure | Abort |
| S3 | OCR failure | Use empty text |
| S3 | Classification uncertain | Fallback to SimpleClassifier |
| S4 | Primary AI provider fails | Auto-fallback via AIRouter chain |
| S4 | All providers exhausted | AIProviderExhaustedError -> rule-based fallback |
| S4 | SLM timeout | Try next provider in chain |
| S5 | Validation failure | Return without insights |

---

## 9. Key Enums

All enums live in `src/core_engine/schemas/enums.py` (single source of truth):

| Enum | Values |
| --- | --- |
| `ChartType` | BAR, LINE, PIE, SCATTER, AREA, HISTOGRAM, HEATMAP, BOX, STACKED_BAR, GROUPED_BAR, DONUT, UNKNOWN |
| `InsightType` | TREND, COMPARISON, ANOMALY, SUMMARY, CORRELATION |
| `TextRole` | TITLE, SUBTITLE, X_AXIS_LABEL, Y_AXIS_LABEL, X_TICK, Y_TICK, LEGEND, DATA_LABEL, ANNOTATION, UNKNOWN |
| `ElementType` | BAR, LINE, POINT, SLICE, AREA, GRID_LINE, AXIS, LEGEND_ITEM |
| `StageStatus` | PENDING, PROCESSING, COMPLETED, FAILED, SKIPPED |
| `PipelineStatus` | IDLE, RUNNING, COMPLETED, PARTIAL, FAILED, CANCELLED |
| `ErrorCode` | 15 codes across S1-S5 + general |
| `ConfidenceThreshold` | DETECTION_MIN=0.5, OCR_MIN=0.6, VALUE_EXTRACTION=0.7 |
