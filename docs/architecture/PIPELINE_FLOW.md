# Pipeline Flow

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2.0.0 | 2026-02-04 | That Le | Updated pipeline status |

## Implementation Status

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1 | ✅ Complete | PDF/Image ingestion |
| Stage 2 | ✅ Complete | YOLO 93.5% mAP@50 |
| Stage 3 | ✅ Complete | 139/140 tests pass |
| Stage 4 | 🔄 In Progress | SLM integration |
| Stage 5 | ⏳ Planned | Reporting |

## 1. Pipeline Overview

The Geo-SLM Chart Analysis pipeline processes input documents through 5 sequential stages:

```mermaid
flowchart LR
    A[Input] --> S1[Stage 1\nIngestion]
    S1 --> S2[Stage 2\nDetection]
    S2 --> S3[Stage 3\nExtraction]
    S3 --> S4[Stage 4\nReasoning]
    S4 --> S5[Stage 5\nReporting]
    S5 --> B[Output]
    
    style S1 fill:#e3f2fd
    style S2 fill:#e8f5e9
    style S3 fill:#fff3e0
    style S4 fill:#fce4ec
    style S5 fill:#f3e5f5
```

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

# Output
class Stage1Output(BaseModel):
    session: SessionInfo
    images: List[CleanImage]
    warnings: List[str]

class CleanImage(BaseModel):
    image_path: Path      # Path to normalized image
    original_path: Path   # Source file reference
    page_number: int      # Page number (1 for images)
    width: int
    height: int
    is_grayscale: bool
```

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
        F[Confidence Filtering]
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
# Input
Stage1Output

# Output
class Stage2Output(BaseModel):
    session: SessionInfo
    charts: List[DetectedChart]
    total_detected: int
    skipped_low_confidence: int

class DetectedChart(BaseModel):
    chart_id: str           # Unique identifier
    source_image: Path      # Original image path
    cropped_path: Path      # Cropped chart path
    bbox: BoundingBox       # Detection coordinates
    page_number: int        # Source page
```

## 4. Stage 3: Structural Analysis (Hybrid)

### 4.1. Purpose

Extract raw metadata through OCR, element detection, and geometric analysis.

### 4.2. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 2 Output"]
        A[DetectedChart\nCropped Image]
    end
    
    subgraph Parallel["Parallel Processing"]
        subgraph Classification["Chart Classification"]
            B1[Load Classifier]
            B2[Classify Type\nbar/line/pie/scatter]
        end
        
        subgraph OCR["Text Extraction"]
            C1[PaddleOCR Init]
            C2[Extract All Text]
            C3[Identify Roles\ntitle/label/legend/value]
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
        E3[Map Coordinates]
    end
    
    subgraph Assembly["Data Assembly"]
        F[Combine Results\ninto RawMetadata]
    end
    
    subgraph Output["Stage 3 Output"]
        G[Stage3Output\nList of RawMetadata]
    end
    
    A --> B1 --> B2
    A --> C1 --> C2 --> C3
    A --> D1 --> D2 --> D3
    
    B2 --> E1
    C3 --> E1
    D3 --> E1
    
    E1 --> E2 --> E3 --> F --> G
```

### 4.3. OCR Text Role Detection

```mermaid
flowchart TB
    subgraph Chart["Chart Image"]
        A[Title Area\nTop region]
        B[Y-Axis Labels\nLeft region]
        C[X-Axis Labels\nBottom region]
        D[Legend\nTop-right or bottom]
        E[Value Labels\nOn/near elements]
    end
    
    subgraph Roles["Detected Roles"]
        R1[role: title]
        R2[role: ylabel]
        R3[role: xlabel]
        R4[role: legend]
        R5[role: value]
    end
    
    A --> R1
    B --> R2
    C --> R3
    D --> R4
    E --> R5
```

### 4.4. Input/Output Schema

```python
# Input
Stage2Output

# Output
class Stage3Output(BaseModel):
    session: SessionInfo
    metadata: List[RawMetadata]

class RawMetadata(BaseModel):
    chart_id: str
    chart_type: ChartType
    texts: List[OCRText]
    elements: List[ChartElement]
    axis_info: Optional[AxisInfo]

class OCRText(BaseModel):
    text: str
    bbox: BoundingBox
    confidence: float
    role: Optional[str]  # title, xlabel, ylabel, legend, value

class ChartElement(BaseModel):
    element_type: str    # bar, point, slice, line
    bbox: BoundingBox
    center: Point
    color: Optional[Color]
    area_pixels: Optional[int]
```

## 5. Stage 4: Semantic Reasoning (SLM)

### 5.1. Purpose

Apply SLM to correct OCR errors, map values, and generate descriptions.

### 5.2. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 3 Output"]
        A[RawMetadata\nOCR + Elements + Geometry]
    end
    
    subgraph GeometricMapping["Geometric Value Mapping"]
        B1[Calculate Axis Scale]
        B2[Map Pixel to Value]
        B3[Initial Value Estimates]
    end
    
    subgraph SLMProcessing["SLM Processing"]
        C1[Build Context Prompt]
        C2[Include:\n- Chart type\n- OCR text\n- Initial values]
        C3[SLM Inference]
        C4[Parse Response]
    end
    
    subgraph Correction["Error Correction"]
        D1[OCR Error Fixes\nloo -> 100]
        D2[Value Refinement]
        D3[Legend-Color Mapping]
    end
    
    subgraph Description["Description Generation"]
        E1[Generate Summary]
        E2[Academic Style]
    end
    
    subgraph Output["Stage 4 Output"]
        F[Stage4Output\nList of RefinedChartData]
    end
    
    A --> B1 --> B2 --> B3
    
    B3 --> C1 --> C2 --> C3 --> C4
    
    C4 --> D1 --> D2 --> D3
    
    D3 --> E1 --> E2 --> F
```

### 5.3. SLM Prompt Structure

```mermaid
flowchart LR
    subgraph Prompt["SLM Prompt"]
        A["System: You are a chart analysis expert..."]
        B["Context:\n- Chart Type: bar\n- OCR Texts: [...]"]
        C["Task:\n1. Fix OCR errors\n2. Map legend to colors\n3. Generate description"]
    end
    
    subgraph Response["Expected Response"]
        D["Corrections: {...}"]
        E["Legend Mapping: {...}"]
        F["Description: '...'"]
    end
    
    A --> B --> C --> D --> E --> F
```

### 5.4. Input/Output Schema

```python
# Input
Stage3Output

# Output
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

## 6. Stage 5: Insight & Reporting

### 6.1. Purpose

Generate final output with insights and formatted report.

### 6.2. Flow Diagram

```mermaid
flowchart TD
    subgraph Input["Stage 4 Output"]
        A[RefinedChartData]
    end
    
    subgraph Validation["Schema Validation"]
        B1[Validate Required Fields]
        B2[Check Value Ranges]
        B3[Verify Relationships]
    end
    
    subgraph Insights["Insight Generation"]
        C1[Trend Analysis\nIncreasing/Decreasing]
        C2[Comparison\nMax/Min/Average]
        C3[Anomaly Detection\nOutliers]
    end
    
    subgraph Formatting["Output Formatting"]
        D1[Build JSON Structure]
        D2[Add Traceability Info]
        D3[Generate Text Summary]
    end
    
    subgraph Output["Final Output"]
        E1[PipelineResult JSON]
        E2[Human-readable Report]
    end
    
    A --> B1 --> B2 --> B3
    B3 --> C1 --> C2 --> C3
    C3 --> D1 --> D2 --> D3
    D3 --> E1
    D3 --> E2
```

### 6.3. Insight Types

```mermaid
flowchart TB
    subgraph TrendInsight["Trend Insight"]
        T1["insight_type: trend"]
        T2["text: Values show increasing trend\nfrom Q1 to Q4"]
    end
    
    subgraph ComparisonInsight["Comparison Insight"]
        C1["insight_type: comparison"]
        C2["text: Product A leads with 45%\nmarket share"]
    end
    
    subgraph AnomalyInsight["Anomaly Insight"]
        A1["insight_type: anomaly"]
        A2["text: Q3 shows unusual spike\nof 150% vs average"]
    end
    
    subgraph SummaryInsight["Summary Insight"]
        S1["insight_type: summary"]
        S2["text: Chart displays quarterly\nsales data for 2025"]
    end
```

### 6.4. Final Output Schema

```python
class PipelineResult(BaseModel):
    session: SessionInfo
    charts: List[FinalChartResult]
    summary: str
    processing_time_seconds: float
    model_versions: Dict[str, str]

class FinalChartResult(BaseModel):
    chart_id: str
    chart_type: ChartType
    title: Optional[str]
    data: RefinedChartData
    insights: List[ChartInsight]
    source_info: Dict[str, Any]

class ChartInsight(BaseModel):
    insight_type: str  # trend, comparison, anomaly, summary
    text: str
    confidence: float
```

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
    participant SLM
    
    User->>Pipeline: run("report.pdf")
    
    Note over Pipeline,S1: Stage 1: Ingestion
    Pipeline->>S1: process(path)
    S1->>S1: Load PDF
    S1->>S1: Convert to images
    S1->>S1: Validate quality
    S1-->>Pipeline: Stage1Output (3 images)
    
    Note over Pipeline,S2: Stage 2: Detection
    Pipeline->>S2: process(Stage1Output)
    loop For each image
        S2->>YOLO: detect(image)
        YOLO-->>S2: bboxes
        S2->>S2: Crop charts
    end
    S2-->>Pipeline: Stage2Output (5 charts)
    
    Note over Pipeline,S3: Stage 3: Extraction
    Pipeline->>S3: process(Stage2Output)
    loop For each chart
        par Parallel
            S3->>S3: Classify type
            S3->>OCR: extract(image)
            OCR-->>S3: texts
            S3->>S3: Detect elements
        end
        S3->>S3: Geometric analysis
    end
    S3-->>Pipeline: Stage3Output
    
    Note over Pipeline,S4: Stage 4: Reasoning
    Pipeline->>S4: process(Stage3Output)
    loop For each metadata
        S4->>S4: Map pixel values
        S4->>SLM: reason(context)
        SLM-->>S4: corrections + description
        S4->>S4: Apply corrections
    end
    S4-->>Pipeline: Stage4Output
    
    Note over Pipeline,S5: Stage 5: Reporting
    Pipeline->>S5: process(Stage4Output)
    S5->>S5: Validate schema
    S5->>S5: Generate insights
    S5->>S5: Format output
    S5-->>Pipeline: PipelineResult
    
    Pipeline-->>User: JSON + Report
```

## 8. Error Recovery Flows

### 8.1. Recoverable Errors

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

### 8.2. Error Types by Stage

| Stage | Error Type | Recovery Strategy |
| --- | --- | --- |
| S1 | File not found | Abort |
| S1 | Low quality image | Skip with warning |
| S2 | No detections | Return empty list |
| S2 | Model load failure | Abort |
| S3 | OCR failure | Use empty text |
| S3 | Classification uncertain | Default to "unknown" |
| S4 | SLM timeout | Use rule-based fallback |
| S5 | Validation failure | Return without insights |
