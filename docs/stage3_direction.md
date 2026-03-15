# Stage 3 Direction Note - 2026-03-13

## Context
Thesis target: Vietnamese Chart QA (not geometry-based extraction)

## Key Decisions

### 1. VLM Extraction (Stage 3)
- **Primary backend**: DePlot (google/deplot) - Pix2Struct fine-tuned on chart-to-table
- Rationale: Since we use a separate SLM for reasoning, DePlot table output is sufficient as context
- May ablation-compare pix2struct / matcha, but DePlot is the production choice
- Action: Run batch extraction on all 32,364 charts -> cache table outputs

### 2. OCR Engine - NEEDS CHANGE
- Current: EasyOCR -> weak, already removed from pipeline in v6.0.0
- OCR cache (data/cache/ocr_cache.json, ~600MB, 46,910 entries) is geometry-era, NOT used anymore
- Need research: PaddleOCR (already in stack via paddle_server.py) vs something better
- Use case: OCR is for title/legend/label extraction (supplementary to DePlot table)
- Candidates to research:
  - PaddleOCR v4 (already available via paddlevl_adapter)
  - Qwen2-VL (SVLM backend - can do OCR + extraction in one pass)
  - TrOCR (Microsoft) - transformer-based OCR
  - GOT-OCR2.0 - general OCR theory, SOTA 2024

### 3. Data Cache Strategy (NEW - v4 dataset)
Goal: Cache both OCR output AND VLM table extraction per chart_id

```
data/academic_dataset/
    stage3_vlm_cache/          # NEW - DePlot table outputs per chart
        bar/chart_id.json
        line/chart_id.json
        ...
    stage3_ocr_cache/          # NEW - Better OCR outputs per chart
        bar/chart_id.json
        ...
```

### 4. SLM Training Data v4
- Join: VLM table cache + Gemini QA pairs (268k pairs, chart_qa_v2/)
- Format: ChatML with [TABLE] + [QUESTION] -> answer
- Scale: ~32k high-quality samples (chart_id intersection)
- Target: SLM handles Vietnamese QA reasoning over extracted table data

## Next Steps (Ordered)
1. Research OCR options for chart text (title, legend, axis labels)
2. Run batch DePlot extraction on 32,364 charts -> stage3_vlm_cache/
3. Build prepare_slm_training_v4.py: join VLM cache + QA pairs
4. Update training.yaml for Qwen2.5-7B-Instruct
5. Run QLoRA training on v4 dataset

## Related Files
- scripts/pipeline/batch_stage3_parallel.py - batch extraction script
- scripts/training/prepare_slm_training_v3.py - v3 builder (geometry-era, needs v4 rewrite)
- src/core_engine/stages/s3_extraction/extractors.py - VLM backends
- src/core_engine/ai/adapters/paddlevl_adapter.py - PaddleOCR via microservice
- paddle_server.py - PaddleOCR microservice
