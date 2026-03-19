# System Architecture -- Mermaid Diagrams

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 1.0.0 | 2026-03-16 | Copilot (Claude Opus 4.6) | Full system architecture diagrams |

Tai lieu nay mo ta toan bo kien truc he thong Geo-SLM Chart Analysis v3 bang cac Mermaid diagrams. Moi diagram tap trung vao mot khia canh cua he thong.

---

## 1. Tong quan Kien truc He thong

```mermaid
flowchart TB
    subgraph INTERFACE["INTERFACE LAYER"]
        direction LR
        CLI["CLI Tool<br/><small>interface/cli.py</small>"]
        GRADIO["Gradio Demo<br/><small>interface/demo_app.py</small>"]
        API["FastAPI Server<br/><small>src/api/main.py</small>"]
    end

    subgraph CONFIG["CONFIGURATION"]
        direction LR
        BASE_YAML["base.yaml<br/><small>Logging, paths</small>"]
        PIPE_YAML["pipeline.yaml<br/><small>Stage toggles</small>"]
        MODEL_YAML["models.yaml<br/><small>Weights, thresholds</small>"]
        TRAIN_YAML["training.yaml<br/><small>LoRA, hyperparams</small>"]
    end

    subgraph CORE["CORE ENGINE  --  src/core_engine/"]
        direction TB

        PIPELINE["ChartAnalysisPipeline<br/><small>pipeline.py</small>"]
        BUILDER["PipelineBuilder<br/><small>builder.py</small>"]
        REGISTRY["AdapterRegistry<br/><small>registry.py</small>"]

        subgraph STAGES["PIPELINE STAGES"]
            direction LR
            S1["S1: Ingestion"]
            S2["S2: Detection"]
            S3["S3: Extraction"]
            S4["S4: Reasoning"]
            S5["S5: Reporting"]
            S1 --> S2 --> S3 --> S4 --> S5
        end

        subgraph AI_LAYER["AI ROUTING LAYER  --  ai/"]
            direction LR
            ROUTER["AIRouter<br/><small>router.py</small>"]
            ADAPT_LOCAL["LocalSLMAdapter<br/><small>Qwen-2.5-1.5B</small>"]
            ADAPT_GEMINI["GeminiAdapter<br/><small>gemini-2.0-flash</small>"]
            ADAPT_OPENAI["OpenAIAdapter<br/><small>gpt-4o-mini</small>"]
            ADAPT_PADDLE["PaddleVLAdapter<br/><small>PaddleOCR-VL</small>"]
        end

        subgraph SCHEMAS["SCHEMAS  --  schemas/"]
            direction LR
            COMMON["common.py<br/><small>BoundingBox, ChartType</small>"]
            OUTPUTS["stage_outputs.py<br/><small>Stage1-5 Output</small>"]
            ENUMS["enums.py<br/><small>Shared vocabulary</small>"]
        end

        PIPELINE --> BUILDER
        BUILDER --> REGISTRY
        BUILDER --> STAGES
        S4 --> ROUTER
        ROUTER --> ADAPT_LOCAL
        ROUTER --> ADAPT_GEMINI
        ROUTER --> ADAPT_OPENAI
        ROUTER --> ADAPT_PADDLE
    end

    subgraph SERVICES["EXTERNAL SERVICES  --  Docker"]
        direction LR
        OCR_SVC["PaddleOCR Service<br/><small>services/ocr/<br/>Port 8001</small>"]
        REDIS["Redis<br/><small>Cache + Queue</small>"]
    end

    subgraph TRAINING["TRAINING INFRA"]
        direction LR
        RUN_MGR["RunManager<br/><small>src/training/</small>"]
        TRACKER["ExperimentTracker<br/><small>WandB/TB/JSON</small>"]
        SCRIPTS_T["Training Scripts<br/><small>scripts/training/</small>"]
    end

    subgraph EVAL["EVALUATION"]
        direction LR
        BENCH_RUN["BenchmarkRunner<br/><small>benchmarks/runner.py</small>"]
        BENCH_REG["BenchmarkRegistry<br/><small>7 suites</small>"]
        METRICS["Metrics<br/><small>ANLS, CER, F1...</small>"]
    end

    subgraph DATA["DATA LAYER"]
        direction LR
        RAW["data/raw_pdfs/"]
        PROCESSED["data/processed/"]
        CACHE["data/cache/"]
        BENCHMARK["data/benchmark/"]
        SLM_DATA["data/slm_training_v3/"]
        MODELS["models/weights/"]
    end

    INTERFACE --> PIPELINE
    CONFIG --> PIPELINE
    CONFIG --> BUILDER
    ADAPT_PADDLE --> OCR_SVC
    TRAINING --> DATA
    EVAL --> DATA
    CORE --> DATA
    CORE --> MODELS

    style INTERFACE fill:#e1f5fe,stroke:#0288d1
    style CORE fill:#fff3e0,stroke:#ef6c00
    style SERVICES fill:#fce4ec,stroke:#c62828
    style TRAINING fill:#e8f5e9,stroke:#2e7d32
    style EVAL fill:#f3e5f5,stroke:#7b1fa2
    style DATA fill:#eceff1,stroke:#546e6a
    style CONFIG fill:#fffde7,stroke:#f9a825
```

---

## 2. Pipeline Data Flow Chi tiet (S1 -> S5)

```mermaid
flowchart TD
    INPUT["User Input<br/>PDF / DOCX / MD / PNG / JPG"]

    subgraph S1["STAGE 1: INGESTION & SANITATION"]
        direction TB
        S1_DETECT["Detect File Type"]
        S1_PARSE["Parser Selection"]

        subgraph PARSERS["Parsers"]
            direction LR
            PDF_P["PDFParser<br/><small>PyMuPDF, 150 DPI</small>"]
            DOCX_P["DocxParser<br/><small>python-docx</small>"]
            IMG_P["ImageParser<br/><small>Pillow direct load</small>"]
            MD_P["MarkdownParser<br/><small>Extract local images</small>"]
        end

        S1_VALIDATE["Quality Validation<br/><small>blur, min_size, format</small>"]
        S1_NORMALIZE["Normalize<br/><small>resize, contrast, RGB</small>"]
        S1_OUT["Stage1Output<br/><small>SessionInfo + CleanImage[]</small>"]

        S1_DETECT --> S1_PARSE
        S1_PARSE --> PARSERS
        PARSERS --> S1_VALIDATE
        S1_VALIDATE --> S1_NORMALIZE
        S1_NORMALIZE --> S1_OUT
    end

    subgraph S2["STAGE 2: DETECTION & LOCALIZATION"]
        direction TB
        S2_LOAD["Load Detection Model"]

        subgraph DETECTORS["Detection Adapters"]
            direction LR
            YOLO8["YOLOv8Adapter<br/><small>chart_detector.pt</small>"]
            YOLO11["YOLOv11Adapter<br/><small>extends v8</small>"]
            MOCK_D["MockAdapter<br/><small>testing</small>"]
        end

        S2_INFER["Run Inference<br/><small>conf >= 0.5</small>"]
        S2_NMS["NMS Filter<br/><small>IoU 0.45</small>"]
        S2_CROP["Crop + Pad<br/><small>10px padding</small>"]
        S2_OUT["Stage2Output<br/><small>DetectedChart[]<br/>bbox + cropped images</small>"]

        S2_LOAD --> DETECTORS
        DETECTORS --> S2_INFER
        S2_INFER --> S2_NMS
        S2_NMS --> S2_CROP
        S2_CROP --> S2_OUT
    end

    subgraph S3["STAGE 3: EXTRACTION & CLASSIFICATION"]
        direction TB

        subgraph CLASSIFY["Chart Classification"]
            direction LR
            EFFNET["EfficientNet-B0<br/><small>97.54% acc<br/>3-class: bar/line/pie</small>"]
            ML_CLS["MLClassifier<br/><small>fallback</small>"]
        end

        subgraph VLM["VLM Extraction Backends"]
            direction LR
            DEPLOT["DeplotExtractor<br/><small>google/deplot<br/>DEFAULT</small>"]
            MATCHA["MatchaExtractor<br/><small>google/matcha-base</small>"]
            PIX2S["Pix2StructExtractor<br/><small>google/pix2struct-base</small>"]
            SVLM["SVLMExtractor<br/><small>Qwen2-VL-2B</small>"]
        end

        S3_TABLE["Pix2StructResult<br/><small>headers, rows, records<br/>raw_html, confidence</small>"]
        S3_META["RawMetadata<br/><small>chart_type + pix2struct_table<br/>+ OCR texts + elements</small>"]
        S3_OUT["Stage3Output<br/><small>List[RawMetadata]</small>"]

        CLASSIFY --> VLM
        VLM --> S3_TABLE
        S3_TABLE --> S3_META
        S3_META --> S3_OUT
    end

    subgraph S4["STAGE 4: SEMANTIC REASONING"]
        direction TB

        subgraph ENGINES["Reasoning Engines"]
            direction LR
            ROUTER_E["AIRouterEngine<br/><small>RECOMMENDED</small>"]
            GEMINI_E["GeminiEngine<br/><small>legacy</small>"]
        end

        S4_PROMPT["PromptBuilder<br/><small>Build structured prompt<br/>from RawMetadata</small>"]
        S4_VMAP["ValueMapper<br/><small>pix2struct_table -> DataSeries</small>"]
        S4_REASON["AI Reasoning<br/><small>OCR correction<br/>value refinement<br/>description gen</small>"]
        S4_OUT["Stage4Output<br/><small>RefinedChartData[]<br/>title, series, description</small>"]

        S4_PROMPT --> ENGINES
        ENGINES --> S4_REASON
        S4_REASON --> S4_VMAP
        S4_VMAP --> S4_OUT
    end

    subgraph S5["STAGE 5: REPORTING & INSIGHTS"]
        direction TB
        S5_VALIDATE["Schema Validation<br/><small>FinalChartResult</small>"]
        S5_INSIGHT["Insight Generation<br/><small>trend, comparison,<br/>anomaly (z-score > 2.0)</small>"]
        S5_FORMAT["Format Output<br/><small>JSON + Markdown + CSV</small>"]
        S5_OUT["PipelineResult<br/><small>FinalChartResult[]<br/>summary, timing, versions</small>"]

        S5_VALIDATE --> S5_INSIGHT
        S5_INSIGHT --> S5_FORMAT
        S5_FORMAT --> S5_OUT
    end

    INPUT --> S1
    S1 -->|"CleanImage[]"| S2
    S2 -->|"DetectedChart[]"| S3
    S3 -->|"RawMetadata[]"| S4
    S4 -->|"RefinedChartData[]"| S5
    S5 --> FINAL["Final JSON + Report"]

    style S1 fill:#e3f2fd,stroke:#1565c0
    style S2 fill:#e8f5e9,stroke:#2e7d32
    style S3 fill:#fff3e0,stroke:#ef6c00
    style S4 fill:#fce4ec,stroke:#c62828
    style S5 fill:#f3e5f5,stroke:#7b1fa2
    style INPUT fill:#fff9c4,stroke:#f57f17
    style FINAL fill:#c8e6c9,stroke:#388e3c
```

---

## 3. AI Router -- Fallback Chains & Provider Selection

```mermaid
flowchart TD
    STAGE4["Stage 4: Reasoning<br/><small>s4_reasoning.py</small>"]

    STAGE4 --> ROUTER_ENGINE["AIRouterEngine<br/><small>router_engine.py</small>"]

    ROUTER_ENGINE --> ROUTER["AIRouter<br/><small>ai/router.py</small>"]

    ROUTER --> TASK_RESOLVE{"Resolve TaskType"}

    TASK_RESOLVE --> |"CHART_REASONING"| CHAIN_CR["Fallback Chain:<br/>1. local_slm<br/>2. gemini<br/>3. openai"]
    TASK_RESOLVE --> |"OCR_CORRECTION"| CHAIN_OCR["Fallback Chain:<br/>1. local_slm<br/>2. gemini"]
    TASK_RESOLVE --> |"DESCRIPTION_GEN"| CHAIN_DESC["Fallback Chain:<br/>1. local_slm<br/>2. gemini<br/>3. openai"]
    TASK_RESOLVE --> |"DATA_VALIDATION"| CHAIN_VAL["Fallback Chain:<br/>1. gemini<br/>2. openai"]
    TASK_RESOLVE --> |"DATA_EXTRACTION"| CHAIN_EXT["Fallback Chain:<br/>1. paddlevl<br/>2. gemini"]

    CHAIN_CR --> TRY_PROVIDER
    CHAIN_OCR --> TRY_PROVIDER
    CHAIN_DESC --> TRY_PROVIDER
    CHAIN_VAL --> TRY_PROVIDER
    CHAIN_EXT --> TRY_PROVIDER

    TRY_PROVIDER{"Try Provider[i]"}

    TRY_PROVIDER --> HEALTH{"health_check()"}
    HEALTH --> |"unhealthy"| NEXT_P["Next in chain"]
    HEALTH --> |"healthy"| CALL["adapter.reason()"]

    CALL --> CHECK_CONF{"confidence >= 0.7?"}
    CHECK_CONF --> |"yes"| RETURN["Return AIResponse"]
    CHECK_CONF --> |"no"| NEXT_P

    NEXT_P --> |"more providers"| TRY_PROVIDER
    NEXT_P --> |"chain exhausted"| EXHAUST["AIProviderExhaustedError"]

    subgraph ADAPTERS["Provider Adapters"]
        direction TB
        LOCAL["LocalSLMAdapter<br/><small>Qwen-2.5-1.5B / Llama-3.2-1B<br/>Local GPU/CPU, cost = $0<br/>enabled: false (until trained)</small>"]
        GEMINI["GeminiAdapter<br/><small>gemini-2.0-flash<br/>Google AI Studio<br/>Vision capable</small>"]
        OPENAI["OpenAIAdapter<br/><small>gpt-4o-mini<br/>OpenAI API<br/>Vision capable</small>"]
        PADDLE["PaddleVLAdapter<br/><small>PaddleOCR-VL<br/>Docker microservice<br/>Port 8001</small>"]
    end

    CALL --> ADAPTERS

    subgraph RESPONSE["AIResponse"]
        direction LR
        R_CONTENT["content: str"]
        R_MODEL["model_used: str"]
        R_PROVIDER["provider: str"]
        R_CONF["confidence: float"]
        R_USAGE["usage: Dict"]
        R_SUCCESS["success: bool"]
    end

    RETURN --> RESPONSE

    style ROUTER fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style ADAPTERS fill:#e8f5e9,stroke:#2e7d32
    style RESPONSE fill:#e3f2fd,stroke:#1565c0
    style EXHAUST fill:#ffcdd2,stroke:#c62828
```

---

## 4. Stage 3 -- VLM Extraction Architecture

```mermaid
flowchart TD
    CHART_IMG["Cropped Chart Image<br/><small>from Stage 2</small>"]

    CHART_IMG --> CLASSIFY["EfficientNet-B0 Classifier<br/><small>predict_with_confidence()</small>"]

    CLASSIFY --> |"chart_type + confidence"| CONF_CHECK{"confidence >= 0.55?"}
    CONF_CHECK --> |"yes"| USE_TYPE["Use predicted type"]
    CONF_CHECK --> |"no"| UNKNOWN["ChartType.UNKNOWN"]

    CHART_IMG --> FACTORY["create_extractor(backend)<br/><small>extractors.py factory</small>"]

    FACTORY --> |"backend=deplot"| DEPLOT["DeplotExtractor<br/><small>google/deplot<br/>~1.1 GB<br/>Linearized table parser</small>"]
    FACTORY --> |"backend=matcha"| MATCHA["MatchaExtractor<br/><small>google/matcha-base<br/>~1.1 GB<br/>Enhanced math</small>"]
    FACTORY --> |"backend=pix2struct"| PIX2S["Pix2StructBaselineExtractor<br/><small>google/pix2struct-base<br/>Ablation baseline</small>"]
    FACTORY --> |"backend=svlm"| SVLM["SVLMExtractor<br/><small>Qwen2-VL-2B-Instruct<br/>~6 GB, zero-shot<br/>JSON-first parsing</small>"]

    DEPLOT --> PARSE_LIN["Linearized Table Parser<br/><small>TITLE | ... newline<br/>col0 | col1 | col2 newline<br/>val0 | val1 | val2</small>"]
    MATCHA --> PARSE_LIN
    PIX2S --> PARSE_LIN

    SVLM --> PARSE_JSON["JSON-first Parser<br/><small>Fallback: linearized</small>"]

    PARSE_LIN --> PIX2RESULT["Pix2StructResult<br/><small>headers: List[str]<br/>rows: List[List[str]]<br/>records: List[Dict]<br/>raw_html: str<br/>extraction_confidence: float</small>"]

    PARSE_JSON --> PIX2RESULT

    USE_TYPE --> RAW_META["RawMetadata"]
    UNKNOWN --> RAW_META
    PIX2RESULT --> RAW_META

    RAW_META --> S3_OUT["Stage3Output<br/><small>session + List[RawMetadata]</small>"]

    subgraph LEGACY["LEGACY (not used, historical reference)"]
        direction LR
        OCR_E["OCREngine<br/><small>PaddleOCR</small>"]
        PREPROC["Preprocessor<br/><small>negative, threshold</small>"]
        SKEL["Skeletonizer<br/><small>Lee-Zhang thinning</small>"]
        VECTOR["Vectorizer<br/><small>RDP simplification</small>"]
        ELEM_DET["ElementDetector<br/><small>bar/point/slice</small>"]
        GEO_MAP["GeometricMapper<br/><small>RANSAC calibration</small>"]
    end

    style CHART_IMG fill:#fff9c4,stroke:#f57f17
    style PIX2RESULT fill:#c8e6c9,stroke:#388e3c
    style S3_OUT fill:#c8e6c9,stroke:#388e3c
    style LEGACY fill:#f5f5f5,stroke:#bdbdbd,stroke-dasharray: 5 5
    style DEPLOT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

---

## 5. Serving Layer -- API, Jobs, Docker

```mermaid
flowchart TD
    USER["User / Client"]

    USER --> |"POST /api/v1/documents/analyze<br/>(multipart file upload)"| API["FastAPI Server<br/><small>src/api/main.py<br/>Port 8000</small>"]

    USER --> |"GET /api/v1/jobs/{id}/status"| API
    USER --> |"GET /api/v1/jobs/{id}/result"| API
    USER --> |"GET /api/v1/health"| API

    API --> CORS["CORS Middleware"]
    CORS --> ROUTES{"Route Handler"}

    ROUTES --> |"/documents/analyze"| DOC_ROUTE["documents.py<br/><small>Upload file<br/>Create job<br/>Launch background task</small>"]
    ROUTES --> |"/jobs/*"| JOB_ROUTE["jobs.py<br/><small>Query job status/result</small>"]
    ROUTES --> |"/health"| HEALTH_ROUTE["health.py<br/><small>Liveness + readiness</small>"]

    DOC_ROUTE --> JOB_STORE["InMemoryJobStore<br/><small>job_store.py<br/>(Redis in production)</small>"]
    DOC_ROUTE --> BG_TASK["BackgroundTask<br/><small>or Celery worker</small>"]

    BG_TASK --> PIPELINE["ChartAnalysisPipeline<br/><small>pipeline.py</small>"]

    PIPELINE --> S1_S5["S1 -> S2 -> S3 -> S4 -> S5"]
    S1_S5 --> RESULT["PipelineResult"]
    RESULT --> JOB_STORE

    JOB_ROUTE --> JOB_STORE

    subgraph JOB_STATES["Job State Machine"]
        direction LR
        PENDING["PENDING"] --> PROCESSING["PROCESSING"]
        PROCESSING --> COMPLETED["COMPLETED"]
        PROCESSING --> FAILED["FAILED"]
    end

    JOB_STORE --> JOB_STATES

    subgraph DOCKER["Docker Compose  --  docker-compose.yml"]
        direction TB
        OCR_CONTAINER["ocr-service<br/><small>services/ocr/<br/>Port 8001<br/>PaddleOCR-VL<br/>Python 3.10</small>"]
        TRAINER_CONTAINER["trainer<br/><small>services/trainer/<br/>GPU access<br/>SLM fine-tuning</small>"]
    end

    S1_S5 --> |"PaddleVL calls"| OCR_CONTAINER

    subgraph CONFIG_API["API Configuration"]
        direction LR
        SETTINGS["Settings(BaseSettings)<br/><small>src/api/config.py<br/>host, port, debug<br/>redis_url, celery_enabled</small>"]
        ENV[".env<br/><small>GOOGLE_API_KEY<br/>OPENAI_API_KEY<br/>DATABASE_URL</small>"]
    end

    ENV --> SETTINGS
    SETTINGS --> API

    style USER fill:#fff9c4,stroke:#f57f17
    style API fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style DOCKER fill:#fce4ec,stroke:#c62828
    style JOB_STATES fill:#f3e5f5,stroke:#7b1fa2
    style RESULT fill:#c8e6c9,stroke:#388e3c
```

---

## 6. Pipeline Orchestration -- Builder, Registry, Config

```mermaid
flowchart TD
    subgraph ENTRY["Entry Points"]
        direction LR
        CLI_E["CLI<br/><small>interface/cli.py</small>"]
        API_E["API Route<br/><small>documents.py</small>"]
        DEMO_E["Gradio Demo<br/><small>demo_app.py</small>"]
        NB_E["Notebook<br/><small>00_full_pipeline_test</small>"]
    end

    ENTRY --> FROM_CFG["ChartAnalysisPipeline.from_config()<br/><small>config_dir='config/', overrides={}</small>"]

    subgraph CONFIG_LOAD["Config Resolution"]
        direction TB
        BASE["base.yaml<br/><small>Logging, data dirs</small>"]
        PIPE["pipeline.yaml<br/><small>Stage settings</small>"]
        MOD["models.yaml<br/><small>Model paths, AI routing</small>"]
        MERGE["OmegaConf.merge()<br/><small>Hierarchical merge</small>"]

        BASE --> MERGE
        PIPE --> MERGE
        MOD --> MERGE
    end

    FROM_CFG --> CONFIG_LOAD
    CONFIG_LOAD --> BUILDER["PipelineBuilder<br/><small>builder.py</small>"]

    BUILDER --> BUILD_S1["_build_s1(config.ingestion)<br/><small>-> Stage1Ingestion</small>"]
    BUILDER --> BUILD_S2["_build_s2(config.detection)<br/><small>-> Stage2Detection</small>"]
    BUILDER --> BUILD_S3["_build_s3(config.extraction)<br/><small>-> Stage3Extraction</small>"]
    BUILDER --> BUILD_S4["_build_s4(config.reasoning)<br/><small>-> Stage4Reasoning</small>"]
    BUILDER --> BUILD_S5["_build_s5(config.reporting)<br/><small>-> Stage5Reporting</small>"]

    BUILD_S2 --> REGISTRY["AdapterRegistry.resolve()<br/><small>'s2_detection', 'yolov8'</small>"]
    REGISTRY --> |"YOLOv8Adapter"| BUILD_S2

    BUILD_S1 --> STAGES_LIST["stages: List[BaseStage]"]
    BUILD_S2 --> STAGES_LIST
    BUILD_S3 --> STAGES_LIST
    BUILD_S4 --> STAGES_LIST
    BUILD_S5 --> STAGES_LIST

    STAGES_LIST --> PIPELINE_OBJ["ChartAnalysisPipeline"]

    PIPELINE_OBJ --> RUN["pipeline.run(input_path)"]

    RUN --> SESSION["_create_session(path)<br/><small>session_id = UUID<br/>config_hash = MD5</small>"]

    SESSION --> LOOP{"For each stage"}
    LOOP --> |"stage.validate_input()"| VALIDATE["Input Validation"]
    VALIDATE --> |"valid"| PROCESS["stage.process(data)"]
    VALIDATE --> |"invalid"| ERROR["StageInputError"]
    PROCESS --> |"next stage"| LOOP
    PROCESS --> |"all done"| METRICS["PipelineMetrics<br/><small>timing per stage</small>"]

    METRICS --> FINAL_RESULT["PipelineResult"]

    style ENTRY fill:#fff9c4,stroke:#f57f17
    style PIPELINE_OBJ fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style FINAL_RESULT fill:#c8e6c9,stroke:#388e3c
    style ERROR fill:#ffcdd2,stroke:#c62828
```

---

## 7. Error Handling -- Fail Gracefully Strategy

```mermaid
flowchart TD
    S1["Stage 1: Ingestion"]
    S2["Stage 2: Detection"]
    S3["Stage 3: Extraction"]
    S4["Stage 4: Reasoning"]
    S5["Stage 5: Reporting"]

    S1 --> |"Success"| S2
    S1 --> |"FileNotFoundError<br/>CorruptedFile"| ABORT["ABORT Pipeline<br/><small>PipelineResult.failed()</small>"]
    S1 --> |"Low quality image"| WARN1["WARNING: skip image<br/>continue with others"]
    WARN1 --> S2

    S2 --> |"Success"| S3
    S2 --> |"ModelNotLoadedError"| ABORT
    S2 --> |"No charts detected"| EMPTY["EMPTY Result<br/><small>PipelineResult.empty()<br/>status='empty'</small>"]
    S2 --> |"Low confidence"| WARN2["WARNING: include<br/>with flag"]
    WARN2 --> S3

    S3 --> |"Success"| S4
    S3 --> |"VLM extraction failed"| S3_FALL["Fallback: empty table<br/><small>confidence=0.0</small>"]
    S3_FALL --> S4
    S3 --> |"Classification uncertain"| S3_UNK["Use ChartType.UNKNOWN"]
    S3_UNK --> S4

    S4 --> |"Success"| S5
    S4 --> |"All providers exhausted"| S4_RULE["Rule-based fallback<br/><small>GeometricValueMapper only</small>"]
    S4_RULE --> S5
    S4 --> |"Timeout"| S4_PARTIAL["Partial result<br/><small>raw values, low confidence</small>"]
    S4_PARTIAL --> S5

    S5 --> |"Success"| DONE["PipelineResult<br/><small>status='completed'</small>"]
    S5 --> |"With warnings"| PARTIAL["PipelineResult<br/><small>status='partial'</small>"]

    subgraph STATUS_CODES["Result Status"]
        direction LR
        COMPLETED["completed<br/><small>All stages passed</small>"]
        PARTIAL_S["partial<br/><small>Some fallbacks used</small>"]
        EMPTY_S["empty<br/><small>No charts found</small>"]
        FAILED_S["failed<br/><small>Critical error</small>"]
    end

    style ABORT fill:#ffcdd2,stroke:#c62828
    style EMPTY fill:#fff9c4,stroke:#f57f17
    style DONE fill:#c8e6c9,stroke:#388e3c
    style PARTIAL fill:#fff3e0,stroke:#ef6c00
```

---

## 8. Training & Evaluation Infrastructure

```mermaid
flowchart TD
    subgraph DATA_PREP["Data Preparation"]
        direction TB
        QA_PAIRS["Chart QA Pairs<br/><small>data/academic_dataset/<br/>chart_qa_v2/generated/<br/>32,364 pairs</small>"]
        S3_FEATURES["Stage 3 Features<br/><small>data/academic_dataset/<br/>stage3_features/<br/>32,364 files</small>"]
        OCR_CACHE["OCR Cache<br/><small>data/cache/ocr_cache.json<br/>46,910 entries, 589 MB</small>"]

        QA_PAIRS --> PREP_SCRIPT["prepare_slm_training_v3.py"]
        S3_FEATURES --> PREP_SCRIPT
        OCR_CACHE --> PREP_SCRIPT

        PREP_SCRIPT --> TRAIN_DATA["data/slm_training_v3/<br/><small>268,799 samples<br/>ChatML format<br/>8 chart types</small>"]

        TRAIN_DATA --> SPLIT["train.json / val.json / test.json<br/><small>Split by chart_id (no leakage)</small>"]
    end

    subgraph TRAINING["Training Pipeline"]
        direction TB
        SPLIT --> SLM_TRAIN["train_slm_lora.py<br/><small>QLoRA fine-tuning</small>"]

        subgraph MODELS_SELECT["Model Selection"]
            direction LR
            QWEN15["Qwen2.5-1.5B<br/><small>PRIMARY</small>"]
            QWEN05["Qwen2.5-0.5B"]
            LLAMA1["Llama-3.2-1B"]
            LLAMA3["Llama-3.2-3B"]
        end

        SLM_TRAIN --> MODELS_SELECT

        subgraph LORA_CFG["LoRA Config"]
            direction LR
            RANK["rank: 16"]
            ALPHA["alpha: 32"]
            DROPOUT["dropout: 0.05"]
            TARGETS["q/k/v/o/gate/up/down_proj"]
        end

        MODELS_SELECT --> LORA_CFG

        LORA_CFG --> RUN_MGR["RunManager<br/><small>Isolated run directory<br/>Config freezing<br/>Run registry</small>"]

        RUN_MGR --> TRACKER["ExperimentTracker<br/><small>WandB / TensorBoard / JSON</small>"]

        RUN_MGR --> RUN_DIR["runs/{model}_{timestamp}/<br/><small>resolved_config.yaml<br/>run_metadata.json<br/>checkpoints/<br/>logs/<br/>final/</small>"]
    end

    subgraph OTHER_TRAINING["Other Training"]
        direction TB
        YOLO_TRAIN["train_yolo_chart_detector.py<br/><small>YOLOv8 fine-tuning<br/>data/yolo_chart_detection/</small>"]
        CLS_TRAIN["train_chart_classifier.py<br/><small>EfficientNet-B0<br/>3/4/8 class modes<br/>7 backbone options</small>"]
        ABLATION["run_mini_ablation.py<br/>run_medium_ablation.py<br/><small>Component comparison</small>"]
    end

    subgraph EVAL["Evaluation Framework"]
        direction TB
        BENCH_RUNNER["run_benchmarks.py<br/><small>CLI entry point</small>"]

        subgraph SUITES["7 Benchmark Suites"]
            direction TB
            SUITE_VLM["vlm_extraction<br/><small>VLM table accuracy</small>"]
            SUITE_OCR["ocr_quality<br/><small>OCR CER/WER</small>"]
            SUITE_CLS["classifier<br/><small>Chart type F1</small>"]
            SUITE_SLM["slm_reasoning<br/><small>SLM quality</small>"]
            SUITE_BASE["baseline_vlm<br/><small>Gemini/GPT-4o zero-shot</small>"]
            SUITE_E2E["e2e_pipeline<br/><small>S1->S5 full pipeline</small>"]
            SUITE_ABL["ablation<br/><small>Component removal study</small>"]
        end

        subgraph METRICS_LIB["Metrics Library"]
            direction LR
            ANLS_M["ANLS"]
            EXACT_M["exact_match"]
            NUM_M["numeric_accuracy"]
            CER_M["CER"]
            OVERLAP_M["text_overlap"]
            RECALL_M["table_value_recall"]
        end

        BENCH_RUNNER --> SUITES
        SUITES --> METRICS_LIB
        METRICS_LIB --> RESULTS["data/benchmark/results/runs/<br/><small>{suite}_{timestamp}/<br/>results.json</small>"]
    end

    subgraph THESIS["Thesis Auto-generation"]
        direction TB
        RESULTS --> TABLE_GEN["generate_thesis_tables.py<br/><small>15 tables (12 static + 3 dynamic)</small>"]
        RESULTS --> FIG_GEN["generate_thesis_figures.py<br/><small>10 figures (7 static + 3 dynamic)</small>"]
        TABLE_GEN --> LATEX_OUT["docs/thesis_capstone/<br/><small>figures/tables/*.tex<br/>figures/*.pdf</small>"]
        FIG_GEN --> LATEX_OUT
    end

    subgraph GCP["GCP Automation"]
        direction LR
        GCP_JOB["config/gcp/training_job.yaml<br/><small>Vertex AI custom job<br/>T4 GPU, 4h timeout</small>"]
        GCP_MAKE["Makefile targets:<br/><small>gcp-train<br/>gcp-sync-up<br/>gcp-pull-model</small>"]
    end

    style DATA_PREP fill:#e3f2fd,stroke:#1565c0
    style TRAINING fill:#e8f5e9,stroke:#2e7d32
    style EVAL fill:#f3e5f5,stroke:#7b1fa2
    style THESIS fill:#fff3e0,stroke:#ef6c00
    style GCP fill:#fce4ec,stroke:#c62828
```

---

## 9. Data Directory & File Flow Map

```mermaid
flowchart LR
    subgraph INPUT["Input Sources"]
        direction TB
        PDF_IN["PDF Documents"]
        IMG_IN["Chart Images"]
        DOCX_IN["DOCX Files"]
    end

    subgraph DATA_DIRS["data/"]
        direction TB
        RAW["raw_pdfs/<br/><small>READ-ONLY after ingestion</small>"]
        DETECTED["detected_charts/<br/><small>Stage 2 cropped images</small>"]
        PROCESSED["processed/<br/><small>Stage 3-5 JSON outputs</small>"]
        CACHE["cache/<br/><small>OCR cache (589 MB)<br/>VLM cache</small>"]
        SAMPLES["samples/<br/><small>Demo images for testing</small>"]
        UPLOADS["uploads/<br/><small>API file uploads</small>"]

        subgraph ACADEMIC["academic_dataset/"]
            direction TB
            CHART_QA["chart_qa_v2/<br/><small>32,364 QA pairs</small>"]
            S3_FEAT["stage3_features/<br/><small>32,364 feature JSONs</small>"]
        end

        subgraph TRAINING_DATA["Training Datasets"]
            direction TB
            SLM_V3["slm_training_v3/<br/><small>268,799 samples<br/>train/val/test.json</small>"]
            SLM_MINI["slm_training_mini/<br/><small>Test subset</small>"]
            YOLO_DATA["yolo_chart_detection/<br/><small>YOLO format labels</small>"]
        end

        subgraph BENCH_DATA["benchmark/"]
            direction TB
            ANNOTATIONS["annotations/<br/><small>50 annotated charts</small>"]
            BENCH_RESULTS["results/runs/<br/><small>{suite}_{timestamp}/</small>"]
        end
    end

    subgraph MODEL_DIR["models/"]
        direction TB
        WEIGHTS["weights/<br/><small>chart_detector.pt<br/>efficientnet_b0_3class_v1_best.pt<br/>resnet18_chart_classifier_v2_best.pt</small>"]
        SLM_MODELS["slm/<br/><small>LoRA adapters<br/>Merged weights</small>"]
        HF_CACHE["~/.cache/huggingface/<br/><small>DePlot (~1.1 GB)<br/>MatCha (~1.1 GB)<br/>Qwen2-VL (~6 GB)</small>"]
    end

    subgraph OUTPUT["output/"]
        direction TB
        MANIFEST["data_manifest.json<br/><small>Auto-generated audit</small>"]
        FINAL_JSON["pipeline_results/<br/><small>Final chart analysis</small>"]
    end

    subgraph RUNS["runs/  (gitignored)"]
        direction TB
        RUN_DIRS["{model}_{timestamp}/<br/><small>resolved_config.yaml<br/>checkpoints/<br/>logs/<br/>final/</small>"]
        RUN_REG["run_registry.json<br/><small>Index of all runs</small>"]
    end

    INPUT --> RAW
    INPUT --> UPLOADS
    RAW --> |"S1"| DETECTED
    DETECTED --> |"S2"| PROCESSED
    PROCESSED --> |"S3-S5"| FINAL_JSON
    ACADEMIC --> SLM_V3
    WEIGHTS --> |"S2"| DETECTED
    WEIGHTS --> |"S3"| PROCESSED
    HF_CACHE --> |"S3"| PROCESSED
    SLM_MODELS --> |"S4"| PROCESSED
    SLM_V3 --> |"training"| RUN_DIRS
    ANNOTATIONS --> BENCH_RESULTS

    style INPUT fill:#fff9c4,stroke:#f57f17
    style DATA_DIRS fill:#e3f2fd,stroke:#1565c0
    style MODEL_DIR fill:#e8f5e9,stroke:#2e7d32
    style OUTPUT fill:#c8e6c9,stroke:#388e3c
    style RUNS fill:#f5f5f5,stroke:#9e9e9e
```

---

## 10. Schema Hierarchy -- Pydantic Models

```mermaid
classDiagram
    class SessionInfo {
        +str session_id
        +datetime created_at
        +Path source_file
        +int total_pages
        +str config_hash
    }

    class CleanImage {
        +Path image_path
        +Path original_path
        +int page_number
        +int width
        +int height
        +str source_type
        +Optional~str~ caption
    }

    class Stage1Output {
        +SessionInfo session
        +List~CleanImage~ images
        +List~str~ warnings
    }

    class BoundingBox {
        +int x_min
        +int y_min
        +int x_max
        +int y_max
        +float confidence
        +width() int
        +height() int
        +center() Tuple
    }

    class DetectedChart {
        +str chart_id
        +Path source_image
        +Path cropped_path
        +BoundingBox bbox
        +int page_number
    }

    class Stage2Output {
        +SessionInfo session
        +List~DetectedChart~ charts
        +int total_detected
        +int skipped_low_confidence
    }

    class Pix2StructResult {
        +List~str~ headers
        +List~List~str~~ rows
        +List~Dict~ records
        +str raw_html
        +str model_name
        +float extraction_confidence
    }

    class RawMetadata {
        +str chart_id
        +ChartType chart_type
        +List~OCRText~ texts
        +List~ChartElement~ elements
        +Optional~Dict~ axis_info
        +Optional~Pix2StructResult~ pix2struct_table
    }

    class Stage3Output {
        +SessionInfo session
        +List~RawMetadata~ metadata
    }

    class DataPoint {
        +str label
        +float value
        +Optional~str~ unit
        +float confidence
    }

    class DataSeries {
        +str name
        +Optional~Color~ color
        +List~DataPoint~ points
    }

    class RefinedChartData {
        +str chart_id
        +ChartType chart_type
        +Optional~str~ title
        +Optional~str~ x_axis_label
        +Optional~str~ y_axis_label
        +List~DataSeries~ series
        +str description
        +List~str~ correction_log
    }

    class Stage4Output {
        +SessionInfo session
        +List~RefinedChartData~ charts
    }

    class ChartInsight {
        +str insight_type
        +str text
        +float confidence
    }

    class FinalChartResult {
        +str chart_id
        +ChartType chart_type
        +Optional~str~ title
        +RefinedChartData data
        +List~ChartInsight~ insights
        +Dict source_info
    }

    class PipelineResult {
        +SessionInfo session
        +List~FinalChartResult~ charts
        +str summary
        +float processing_time_seconds
        +Dict model_versions
        +str status
    }

    Stage1Output --> SessionInfo
    Stage1Output --> CleanImage
    Stage2Output --> SessionInfo
    Stage2Output --> DetectedChart
    DetectedChart --> BoundingBox
    Stage3Output --> SessionInfo
    Stage3Output --> RawMetadata
    RawMetadata --> Pix2StructResult
    Stage4Output --> SessionInfo
    Stage4Output --> RefinedChartData
    RefinedChartData --> DataSeries
    DataSeries --> DataPoint
    PipelineResult --> SessionInfo
    PipelineResult --> FinalChartResult
    FinalChartResult --> RefinedChartData
    FinalChartResult --> ChartInsight
```

---

## 11. CI/CD & Development Workflow

```mermaid
flowchart TD
    DEV["Developer Push"]

    DEV --> |"push / PR to main"| CI["GitHub Actions CI<br/><small>.github/workflows/ci.yml</small>"]

    subgraph CI_JOBS["CI Pipeline"]
        direction TB
        LINT["Lint Job"]
        TEST["Test Job"]

        subgraph LINT_STEPS["Lint"]
            direction TB
            RUFF["ruff check src/ tests/<br/><small>pycodestyle + pyflakes + isort</small>"]
            BLACK["black --check src/ tests/<br/><small>Code formatting</small>"]
        end

        subgraph TEST_STEPS["Test"]
            direction TB
            INSTALL["pip install -e '.[dev]'"]
            PYTEST["pytest tests/<br/><small>404 tests<br/>exclude: test_gpu/</small>"]
        end

        LINT --> LINT_STEPS
        TEST --> TEST_STEPS
        LINT_STEPS --> TEST_STEPS
    end

    CI --> CI_JOBS

    CI_JOBS --> |"All pass"| MERGE["Merge to main"]
    CI_JOBS --> |"Fail"| BLOCK["Block PR"]

    subgraph LOCAL_DEV["Local Development"]
        direction TB
        MAKE_LINT["make lint<br/><small>ruff + black + mypy</small>"]
        MAKE_TEST["make test<br/><small>pytest with coverage</small>"]
        MAKE_FORMAT["make format<br/><small>black + ruff --fix</small>"]
    end

    subgraph RELEASE["Release Process"]
        direction TB
        TAG["git tag v1.x.x"]
        DOCKER_BUILD["make docker-build<br/><small>Build images</small>"]
        DOCKER_PUSH["Push to registry"]
        TAG --> DOCKER_BUILD --> DOCKER_PUSH
    end

    MERGE --> |"Tagged release"| RELEASE

    style CI fill:#e3f2fd,stroke:#1565c0
    style CI_JOBS fill:#f3e5f5,stroke:#7b1fa2
    style LOCAL_DEV fill:#e8f5e9,stroke:#2e7d32
    style RELEASE fill:#fff3e0,stroke:#ef6c00
    style BLOCK fill:#ffcdd2,stroke:#c62828
```

---

## 12. Makefile Command Map

```mermaid
flowchart TD
    MAKE["Makefile<br/><small>18 targets</small>"]

    subgraph DEV_CMDS["Development"]
        direction TB
        INSTALL["make install<br/><small>pip install -e '.[dev]'</small>"]
        TEST_CMD["make test<br/><small>pytest --cov</small>"]
        LINT_CMD["make lint<br/><small>ruff + black + mypy</small>"]
        FORMAT_CMD["make format<br/><small>black + ruff --fix</small>"]
    end

    subgraph DOCKER_CMDS["Docker"]
        direction TB
        OCR_BUILD["make ocr-build<br/><small>Build OCR image</small>"]
        OCR_UP["make ocr-up<br/><small>Start OCR service</small>"]
        OCR_DOWN["make ocr-down<br/><small>Stop OCR service</small>"]
        OCR_LOGS["make ocr-logs<br/><small>View OCR logs</small>"]
    end

    subgraph SERVE_CMDS["Serving"]
        direction TB
        SERVE["make serve<br/><small>uvicorn --reload</small>"]
        DEMO_CMD["make demo<br/><small>Gradio app</small>"]
    end

    subgraph DATA_CMDS["Data"]
        direction TB
        AUDIT["make data-audit<br/><small>Scan data/ dirs</small>"]
    end

    subgraph GCP_CMDS["Cloud"]
        direction TB
        GCP_TRAIN["make gcp-train<br/><small>Vertex AI job</small>"]
        GCP_SYNC["make gcp-sync-up<br/><small>Upload runs to GCS</small>"]
        GCP_PULL["make gcp-pull-model<br/><small>Download model</small>"]
    end

    subgraph THESIS_CMDS["Thesis"]
        direction TB
        THESIS_UP["make thesis-update<br/><small>Generate tables + figures</small>"]
    end

    MAKE --> DEV_CMDS
    MAKE --> DOCKER_CMDS
    MAKE --> SERVE_CMDS
    MAKE --> DATA_CMDS
    MAKE --> GCP_CMDS
    MAKE --> THESIS_CMDS

    style MAKE fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style DEV_CMDS fill:#e3f2fd,stroke:#1565c0
    style DOCKER_CMDS fill:#fce4ec,stroke:#c62828
    style SERVE_CMDS fill:#e8f5e9,stroke:#2e7d32
    style DATA_CMDS fill:#f3e5f5,stroke:#7b1fa2
    style GCP_CMDS fill:#fff9c4,stroke:#f57f17
    style THESIS_CMDS fill:#eceff1,stroke:#546e6a
```

---

## 13. Complete End-to-End User Journey

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio / API
    participant P as Pipeline
    participant S1 as Stage 1
    participant S2 as Stage 2
    participant S3 as Stage 3
    participant S4 as Stage 4
    participant S5 as Stage 5
    participant YOLO as YOLO Model
    participant VLM as DePlot VLM
    participant CLS as EfficientNet
    participant AI as AI Router
    participant SLM as Local SLM
    participant GEM as Gemini API

    U->>G: Upload chart.pdf
    G->>P: pipeline.run("chart.pdf")
    P->>P: Create SessionInfo (UUID)

    Note over P,S1: Stage 1 -- Ingestion
    P->>S1: process(path)
    S1->>S1: Detect file type (PDF)
    S1->>S1: PyMuPDF render pages @150 DPI
    S1->>S1: Quality validation (blur, size)
    S1-->>P: Stage1Output (3 clean images)

    Note over P,S2: Stage 2 -- Detection
    P->>S2: process(stage1_output)
    S2->>YOLO: predict(image, conf=0.5)
    YOLO-->>S2: bounding boxes
    S2->>S2: NMS filter (IoU 0.45)
    S2->>S2: Crop + pad (10px)
    S2-->>P: Stage2Output (5 charts)

    Note over P,S3: Stage 3 -- Extraction
    P->>S3: process(stage2_output)
    loop For each chart
        S3->>CLS: classify(chart_image)
        CLS-->>S3: (bar, 0.97)
        S3->>VLM: extract(chart_image)
        VLM-->>S3: Pix2StructResult (table)
    end
    S3-->>P: Stage3Output (5 RawMetadata)

    Note over P,S4: Stage 4 -- Reasoning
    P->>S4: process(stage3_output)
    loop For each metadata
        S4->>S4: Build prompt from RawMetadata
        S4->>AI: route(CHART_REASONING, prompt)
        AI->>SLM: reason(prompt)
        alt SLM succeeds (conf >= 0.7)
            SLM-->>AI: AIResponse
        else SLM fails or low confidence
            AI->>GEM: reason(prompt)
            GEM-->>AI: AIResponse
        end
        AI-->>S4: AIResponse
        S4->>S4: ValueMapper: table -> DataSeries
    end
    S4-->>P: Stage4Output (5 RefinedChartData)

    Note over P,S5: Stage 5 -- Reporting
    P->>S5: process(stage4_output)
    S5->>S5: Validate schemas
    S5->>S5: Generate insights (trends, anomalies)
    S5->>S5: Format output (JSON + Markdown)
    S5-->>P: PipelineResult

    P-->>G: PipelineResult (JSON)
    G-->>U: Display: Table + Insights + JSON
```

---

*Tong cong 13 diagrams bao phu toan bo kien truc he thong, tu high-level architecture cho den chi tiet tung component, data flow, error handling, CI/CD, va user journey.*
