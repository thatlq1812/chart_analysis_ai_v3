# Confirmation: Stage 4 & Notebooks Update

| Date | Status | Author |
| --- | --- | --- |
| 2026-01-26 | PENDING REVIEW | AI Agent |

## 1. Notebooks Update Plan

### 1.1. Changes to be made

| Notebook | Status | Changes |
| --- | --- | --- |
| `00_quick_start.ipynb` | NEEDS UPDATE | Update to use new ResNet-18 classifier API |
| `01_stage1_ingestion.ipynb` | VERIFY | Check compatibility with current schemas |
| `02_stage2_detection.ipynb` | VERIFY | Check YOLO model path and config |
| `03_stage3_extraction.ipynb` | MAJOR UPDATE | Add ResNet-18 demo, update extraction flow |
| `04_stage4_reasoning.ipynb` | NEW | Create new notebook for Stage 4 demo |

### 1.2. Confirmation Questions

**Q1: Notebook 00 - Quick Start Scope**
- [x] Keep it as a brief overview (load image -> detect -> extract -> output)?
- [ ] Expand to include full pipeline demo?

**Answer:** Keep brief (selected default)

---

**Q2: Sample Images Location**
- [x] Generate synthetic test images in notebooks (current approach)?
- [ ] Use images from `data/academic_dataset/`?
- [ ] Both options available?

**Answer:** Default to synthetic + option to load real samples

---

## 2. Stage 4 Implementation Plan

### 2.1. Architecture Decision

**Q3: Gemini API Integration Approach**

Option A (RECOMMENDED):
```
s4_reasoning/
    __init__.py
    s4_reasoning.py        # Main orchestrator
    reasoning_engine.py    # Abstract interface
    gemini_engine.py       # Gemini Flash implementation
    slm_engine.py          # Local SLM (future)
    prompts/
        ocr_correction.txt
        value_mapping.txt
        description.txt
```

Option B (Simple):
```
s4_reasoning/
    __init__.py
    s4_reasoning.py        # All-in-one
```

**Confirmed:** Option A - Modular design for future SLM integration

---

**Q4: Gemini Model Selection**

| Model | Context | Speed | Cost |
| --- | --- | --- | --- |
| `gemini-2.0-flash-exp` | 1M tokens | Fast | Free tier |
| `gemini-1.5-flash` | 1M tokens | Fast | Cheap |
| `gemini-1.5-pro` | 2M tokens | Medium | Higher |

**Confirmed:** `gemini-2.0-flash-exp` for testing (free tier), configurable

---

**Q5: API Key Management**

- [x] Use `config/secrets/.env` or `GOOGLE_API_KEY` environment variable?
- [ ] Hardcode for testing (NOT RECOMMENDED)?

**Confirmed:** Environment variable with fallback to `config/secrets/.env`

---

### 2.2. Stage 4 Core Features

| Feature | Priority | Implementation |
| --- | --- | --- |
| OCR Error Correction | HIGH | Prompt-based, SLM corrects common errors |
| Value Mapping | HIGH | Geometric + SLM verification |
| Legend-Color Mapping | MEDIUM | Color proximity + SLM confirmation |
| Description Generation | MEDIUM | Prompt template with data |
| Confidence Scoring | LOW | Based on SLM uncertainty |

---

### 2.3. Prompt Templates (Draft)

**OCR Correction Prompt:**
```
You are analyzing a {chart_type} chart. The following text was extracted via OCR:

{ocr_texts}

Common OCR errors to fix:
- "loo" -> "100", "O" -> "0", "l" -> "1", "S" -> "5"
- "%"missing after percentage values

Return the corrected text in JSON format:
{
    "corrections": [
        {"original": "...", "corrected": "...", "reason": "..."}
    ],
    "title": "...",
    "x_axis_label": "...",
    "y_axis_label": "..."
}
```

---

## 3. Dependencies to Add

```toml
# pyproject.toml additions
google-generativeai = ">=0.3.0"  # Gemini API
```

---

## 4. Implementation Order

1. [x] Create this confirmation document
2. [ ] User confirms/modifies decisions
3. [ ] Update notebooks (00, 01, 02, 03)
4. [ ] Create Stage 4 folder structure
5. [ ] Implement Gemini engine
6. [ ] Create Stage 4 notebook
7. [ ] Write integration tests

---

## 5. User Confirmation Required

**Please review and confirm the following by editing this file:**

- [ ] **Notebook plan approved** (Section 1)
- [ ] **Stage 4 architecture (Option A) approved** (Section 2.1)
- [ ] **Gemini model selection approved** (Section 2.2)
- [ ] **Ready to proceed with implementation**

---

*If all looks good, you can simply reply "Confirmed" and I will proceed with implementation.*
