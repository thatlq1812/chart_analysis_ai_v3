# Stage 1: Ingestion & Sanitation

## 1. Architecture

### 1.1. Responsibility
Transform diverse input formats (PDF, PNG, JPG) into normalized `CleanImage` objects ready for chart detection.

### 1.2. Position in Pipeline
```
Input (PDF/Image) --> [Stage 1: Ingestion] --> Stage1Output(List[CleanImage])
                                                    |
                                                    v
                                              Stage 2: Detection
```

### 1.3. Class Hierarchy
```
BaseStage[Path, Stage1Output]
  +-- Stage1Ingestion (511 lines)
      Config: IngestionConfig (Pydantic BaseModel)
```

### 1.4. Subcomponents
| Component | Function |
| --- | --- |
| PDF Renderer | PyMuPDF (fitz) - renders pages at configurable DPI |
| Image Loader | OpenCV + Pillow - direct image loading |
| Quality Validator | Blur detection (Laplacian variance), size checks |
| Normalizer | Resize, format conversion |
| Session Generator | UUID-based session IDs, config hashing |

## 2. Configuration Parameters

| Parameter | Default | Range | Description |
| --- | --- | --- | --- |
| `pdf_dpi` | 150 | 72-300 | DPI for PDF page rendering |
| `max_image_size` | 4096 | - | Maximum dimension in pixels |
| `min_image_size` | 100 | - | Minimum dimension in pixels |
| `min_blur_threshold` | 100.0 | - | Laplacian variance threshold |
| `output_format` | PNG | - | Output image format |
| `preserve_color` | True | - | Keep original color channels |

Source: `config/pipeline.yaml` under `ingestion:` key.

## 3. Algorithms

### 3.1. Blur Detection
Uses **Laplacian variance** method:
1. Convert image to grayscale
2. Compute Laplacian: $L = \nabla^2 I$
3. Calculate variance: $\sigma^2(L)$
4. If $\sigma^2 < \text{threshold}$ (100.0), image is blurry

Low-quality images are **warned but not rejected** -- downstream stages decide.

### 3.2. PDF Rendering
- PyMuPDF renders each page with transformation matrix: `Matrix(DPI/72, DPI/72)`
- RGB output from fitz is converted to BGR (OpenCV convention)
- Each page is independently validated

### 3.3. Image Normalization
- Images exceeding `max_image_size` are downscaled with `INTER_AREA` interpolation
- Images below `min_image_size` are rejected
- Color space preserved unless `preserve_color=False`

## 4. Implementation Details

### 4.1. Supported Formats
`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

File type detection uses magic bytes (not extension), with fallback to extension matching.

### 4.2. Session Management
Each pipeline run generates:
- `session_id`: `sess_YYYYMMDD_HHMMSS_{uuid8}`
- `config_hash`: MD5 of serialized config (first 16 chars) for reproducibility

### 4.3. Output Structure
```
data/processed/{session_id}/
    page_0001.png
    page_0002.png
    ...
```

### 4.4. Error Handling
- Stage 1 is **critical** (`is_critical = True`) -- pipeline cannot proceed without input
- `StageInputError` for invalid files
- `StageProcessingError` for rendering failures
- Fallback output: empty `Stage1Output` with warning messages

## 5. Results / Performance

| Metric | Value |
| --- | --- |
| Supported formats | 6 (PDF, PNG, JPG, JPEG, TIFF, BMP) |
| PDF rendering speed | ~0.5s per page at 150 DPI |
| Memory (per page) | ~10-50 MB depending on DPI |

## 6. Limitations

- DOCX support is planned but not yet implemented
- No OCR-based text extraction from PDFs (deferred to Stage 3)
- No multi-threaded page rendering (sequential processing)
- Blur threshold is fixed -- no adaptive quality assessment
- No automatic DPI detection from PDF metadata
