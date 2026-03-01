# Stage 4: Semantic Reasoning

> **[TODO - SUPPLEMENT AFTER SLM TRAINING]**
> - Add SLM inference results (accuracy, latency, JSON validity)
> - Add local_slm_adapter.py implementation details once QLoRA training completes
> - Add comparison: Gemini API vs Local SLM on same test set

## 1. Architecture

### 1.1. Responsibility
Refine raw extracted metadata from Stage 3 using AI reasoning: correct OCR errors, map pixel coordinates to actual values, associate legend entries with data series, and generate natural language descriptions.

### 1.2. Position in Pipeline
```
Stage3Output(List[RawMetadata]) --> [Stage 4: Reasoning] --> Stage4Output(List[RefinedChartData])
                                                                   |
                                                                   v
                                                             Stage 5: Reporting
```

### 1.3. Class Hierarchy
```
BaseStage[Stage3Output, Stage4Output]
  +-- Stage4Reasoning (479 lines)
      |   Config: ReasoningConfig
      |
      +-- GeometricValueMapper (764 lines)
      +-- GeminiPromptBuilder (833 lines)
      +-- ReasoningEngine (ABC)
            +-- GeminiReasoningEngine (direct Gemini API)
            +-- AIRouterEngine (multi-provider via AIRouter)
```

### 1.4. Processing Pipeline (per chart)
```
RawMetadata
  |
  v
[1. GeometricValueMapper]
  |  Calibrate axes from AxisInfo
  |  Convert pixel -> value for all elements
  |  Handle linear/log scales, Y-inversion
  |
  v (MappingResult: mapped_points[], mapped_series[])
[2. GeminiPromptBuilder]
  |  Build CanonicalContext (structured sections)
  |  Apply anti-hallucination rules
  |  Format as JSON-expected prompt
  |
  v (system_prompt + user_prompt)
[3. ReasoningEngine (via AIRouter)]
  |  Route to: local_slm -> gemini -> openai (fallback chain)
  |  Parse structured JSON response
  |
  v (ReasoningResult)
[4. Post-processing]
  |  Merge OCR text + mapped values
  |  Apply OCR corrections
  |  Compute confidence scores
  |
  v
RefinedChartData {title, series[], description, source_info}
```

## 2. Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `engine` | `"router"` | Reasoning backend: router, gemini, local_slm, rule_based |
| `enable_ocr_correction` | True | Correct OCR errors via LLM |
| `enable_value_mapping` | True | Geometric pixel-to-value conversion |
| `enable_description` | True | Generate natural language description |
| `use_fallback_on_error` | True | Fall back to rules if LLM fails |
| `gemini.model_name` | `"gemini-2.0-flash"` | Gemini model for cloud inference |
| `local_slm_model` | `"Qwen/Qwen2.5-1.5B-Instruct"` | Local SLM model ID |

Source: `config/models.yaml` under `ai_routing:` + `config/pipeline.yaml` under `reasoning:`.

## 3. Algorithms

### 3.1. Geometric Value Mapping (764 lines)

Given calibrated `AxisInfo` from Stage 3:

**Linear mapping** (Y-axis inverted):
$$value_y = y_{max} - \frac{(pixel_y - pixel_{top})}{(pixel_{bottom} - pixel_{top})} \cdot (y_{max} - y_{min})$$

$$value_x = x_{min} + \frac{(pixel_x - pixel_{left})}{(pixel_{right} - pixel_{left})} \cdot (x_{max} - x_{min})$$

**Log scale mapping**:
$$value = 10^{value_{linear}}$$
where $value_{linear}$ is computed as above using log-transformed axis range.

**Confidence filtering**: Only applies mapping if calibration confidence $\geq 0.3$.

**Value clamping**: Optionally restricts mapped values to detected axis range to prevent extrapolation errors.

### 3.2. Prompt Engineering (833 lines)

**Canonical Format**: Structured prompt with explicit sections:
1. **CHART_METADATA**: type, dimensions, classification confidence
2. **OCR_TEXTS**: all extracted text with roles and bboxes
3. **ELEMENTS**: detected bars/points/slices with positions
4. **AXIS_INFO**: calibrated axis ranges and scale factors
5. **MAPPED_VALUES**: geometric value mapping results
6. **INSTRUCTIONS**: task-specific rules + anti-hallucination constraints

**Anti-hallucination Rules**:
- "Only report values that appear in OCR_TEXTS or MAPPED_VALUES"
- "If uncertain, report null rather than guess"
- "Do not invent data series not visible in the chart"
- Output must be valid JSON with specific schema

**Task Types**:
| PromptTask | Purpose |
| --- | --- |
| `FULL_REASONING` | Complete chart data extraction |
| `OCR_CORRECTION` | Fix OCR text errors |
| `VALUE_EXTRACTION` | Extract numerical values |
| `DESCRIPTION_ONLY` | Generate chart description |
| `LEGEND_MAPPING` | Map legend entries to series |
| `TREND_ANALYSIS` | Identify trends |

### 3.3. AI Routing (via AIRouter)

Stage 4 uses `AIRouterEngine` which delegates to the AI Routing Layer:

```
AIRouterEngine.reason(metadata)
  |
  v
AIRouter.route(TaskType.CHART_REASONING, system_prompt, user_prompt)
  |
  v
Fallback chain: local_slm -> gemini -> openai
  |
  v (AIResponse: content, confidence, provider)
Parse JSON -> ReasoningResult
```

**Confidence-based routing**: If provider returns confidence < 0.7, the next provider in chain is attempted.

### 3.4. Rule-Based Fallback

When all AI providers fail, a pure-rules engine applies:
- Common OCR corrections (pattern matching)
- Direct use of geometric mapping values (no LLM refinement)
- Template-based descriptions
- No semantic reasoning (just structural pass-through)

## 4. Output Schema: RefinedChartData

```python
class RefinedChartData:
    chart_id: str
    chart_type: ChartType
    title: Optional[str]
    x_label: Optional[str]
    y_label: Optional[str]
    series: List[DataSeries]       # Named data series with points
    description: Optional[str]     # Natural language summary
    source_info: Dict[str, Any]    # Traceability metadata
    confidence: float              # Overall reasoning confidence
```

```python
class DataSeries:
    name: str
    data_points: List[DataPoint]   # (x, y) with labels
    color: Optional[Color]

class DataPoint:
    x: Optional[float]
    y: Optional[float]
    label: Optional[str]
```

## 5. Results

| Metric | Value | Notes |
| --- | --- | --- |
| Gemini API integration | Working | Used for rapid prototyping |
| Value mapper tests | 16/16 passing | Linear/log/inverted scales |
| Prompt builder tests | 20/20 passing | All task types |
| Engine selection | Router (default) | Multi-provider fallback |
| Local SLM | Training in progress | Qwen-2.5-1.5B + LoRA |

## 6. Lessons Learned

1. **Canonical Format prompts** dramatically reduce hallucination compared to free-form prompts.
2. **Geometric pre-computation** before LLM reduces token cost and improves accuracy -- the LLM validates/refines rather than computes from scratch.
3. **Anti-hallucination constraints** in system prompt are essential: explicit grounding rules reduce fabricated data series by >80%.
4. **Router pattern** decouples reasoning logic from provider -- switching from Gemini to local SLM requires zero pipeline code changes.
5. **Rule-based fallback** ensures the pipeline never fails completely, even without network/GPU.

## 7. Limitations

- Local SLM not yet operational (training in progress with 268,799 samples)
- Gemini API dependency for cloud inference (cost, latency, rate limits)
- Complex multi-series charts with overlapping legends remain challenging
- No vision input to LLM (text-only prompts from Stage 3 metadata)
- Description quality depends on OCR quality from Stage 3
