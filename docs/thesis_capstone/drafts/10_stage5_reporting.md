# Stage 5: Insight & Reporting

## 1. Architecture

### 1.1. Responsibility
Final pipeline stage. Generates human-readable insights from structured chart data, validates output schema, and writes JSON + text reports with full traceability.

### 1.2. Position in Pipeline
```
Stage4Output(List[RefinedChartData]) --> [Stage 5: Reporting] --> PipelineResult
                                                                    |
                                                                    v
                                                            JSON + Markdown Report
```

### 1.3. Class Hierarchy
```
BaseStage[Stage4Output, PipelineResult]
  +-- Stage5Reporting (635 lines)
      Config: ReportingConfig
```

## 2. Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `enable_insights` | True | Generate trend/comparison/anomaly insights |
| `max_insights_per_chart` | 5 | Maximum insights per chart |
| `min_series_for_comparison` | 2 | Minimum series for comparison insights |
| `anomaly_z_score_threshold` | 2.0 | Z-score threshold for anomaly detection |
| `save_json` | True | Write output JSON to disk |
| `save_report` | True | Write text report to disk |
| `output_dir` | `data/output` | Output directory |
| `require_description` | True | Fail if description is empty |

Source: `config/pipeline.yaml` under `reporting:`.

## 3. Algorithms

### 3.1. Insight Generation (4 types)

**Trend Detection**:
- For time-series data: compute slope via linear regression
- Classify as: increasing, decreasing, stable, volatile
- Confidence = $R^2$ of linear fit

**Comparison Insights**:
- For multi-series charts: compare max/min/mean across series
- Identify dominant series, largest gap, crossover points
- Requires $\geq 2$ data series

**Anomaly Detection**:
- Z-score method: flag points where $|z| > 2.0$
- $z = (x - \mu) / \sigma$
- Applied per-series

**Summary Generation**:
- Template-based: combines chart type, axis labels, data range, and trend info
- Example: "Bar chart showing X vs Y, with values ranging from A to B, exhibiting an increasing trend."

### 3.2. Schema Validation
- Each `RefinedChartData` is wrapped in `FinalChartResult` with insights and source_info
- Failed charts are included with error insights rather than dropped
- Session summary aggregates statistics across all charts

### 3.3. Output Generation

**JSON Export**: Full `PipelineResult` serialized with:
- Session metadata (ID, timestamp, config hash)
- All chart results with data, insights, and source traceability
- Processing time and model versions

**Text Report**: Markdown-formatted session summary with per-chart sections.

## 4. Output Schema

```python
class FinalChartResult:
    chart_id: str
    chart_type: ChartType
    title: Optional[str]
    data: RefinedChartData
    insights: List[ChartInsight]
    source_info: Dict[str, Any]   # Traceability: page, bbox, session

class PipelineResult:
    session: SessionInfo
    charts: List[FinalChartResult]
    summary: str
    processing_time_seconds: float
    model_versions: Dict[str, str]
```

## 5. Results

| Metric | Value |
| --- | --- |
| Insight types | 4 (trend, comparison, anomaly, summary) |
| Output formats | JSON (P0), Markdown (P0) |
| Error handling | Include failed charts with error insights |

## 6. Lessons Learned

1. **Include-not-drop** policy for failed charts preserves pipeline completeness.
2. **Z-score anomaly detection** is simple but effective for most chart data distributions.
3. **Traceability metadata** (page_number, bbox, session_id) is essential for debugging and validation.

## 7. Limitations

- Insight generation is rule-based (no LLM-powered reasoning)
- CSV, LaTeX, Excel export not yet implemented
- No interactive visualization (planned for Streamlit demo)
- Summary quality depends entirely on Stage 4 output quality
- Parallel insight generation using ThreadPoolExecutor is planned but not implemented
