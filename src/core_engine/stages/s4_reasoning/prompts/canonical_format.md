# Canonical Format for Chart Analysis

This document defines the structured format for communicating chart data
to LLMs in a way that minimizes hallucination and maximizes accuracy.

## Format Overview

```
INPUT CONTEXT
├── Chart Information (type, id, confidence)
├── Axis Context (labels, ranges, units)
├── OCR Texts (with confidence scores)
├── Geometric Data (estimated values)
└── Quality Warnings

TASKS
├── OCR Correction
├── Value Refinement
├── Legend Mapping
└── Description Generation

OUTPUT FORMAT (JSON schema)

CONSTRAINTS (anti-hallucination rules)
```

## Key Principles

### 1. Grounding
Every piece of information must trace back to extracted data.
LLM should NOT invent values not present in context.

### 2. Confidence Propagation
Include confidence scores at every level:
- OCR text confidence (0-1)
- Calibration confidence (R²)
- Overall extraction confidence

### 3. Explicit Uncertainty
When confidence < 0.7, explicitly flag as uncertain.
Use phrases like "likely", "appears to be", "possibly".

### 4. Structured Output
Always request JSON output for machine parsing.
Include validation schema when possible.

## Example Context

```
### Chart Information
- ID: chart_001
- Type: bar
- Extraction Confidence: 85%

### Axis Context
- X-Axis: Label="Quarter", Range=[Q1, Q4]
- Y-Axis: Label="Revenue ($M)", Range=[0, 500]

### OCR Texts (with confidence)
1. "Quarterly Revenue" [title] (conf: 92%)
2. "Ql" [xlabel] (conf: 78%)  <- possible OCR error
3. "500" [ylabel] (conf: 95%)

### Estimated Data
- Series 1 (Blue): Q1:125, Q2:180, Q3:220, Q4:310

### Warnings
- Low confidence text at index 2
```

## Anti-Hallucination Constraints

```
CONSTRAINTS (CRITICAL):
- ONLY use data from INPUT CONTEXT
- Do NOT invent values not in the data
- If uncertain, set confidence < 0.7
- Flag suspicious values instead of correcting blindly
- Preserve original text if correction uncertain
```
