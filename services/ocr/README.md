# PaddleOCR-VL Extraction Server

Isolated microservice for chart data extraction using PaddleOCR-VL model.

## Why Isolated?

PaddleOCR-VL requires `transformers>=4.45`, which conflicts with the main project's
`transformers==4.44.2` (needed by Vintern). Running as a separate service avoids
dependency conflicts entirely.

## Quick Start

### Docker (Recommended)

```bash
# From project root:
docker compose up -d ocr-service

# Verify:
curl http://localhost:8001/health
```

### Local (Separate venv)

```bash
# Create isolated environment:
python -m venv .venv-paddle
.venv-paddle\Scripts\activate  # Windows
pip install -r requirements.txt

# Start server:
python server.py
```

## API

### GET /health

Returns model status and device info.

```json
{"status": "ok", "model": "PaddleOCR-VL", "device": "cpu", "version": "1.1.0"}
```

### POST /extract

Extract structured data from a chart image.

```bash
curl -X POST http://localhost:8001/extract \
  -F "image=@chart.png" \
  -F "max_new_tokens=512"
```

Response:
```json
{"extracted_data": "TITLE | Revenue Growth\nYear | Revenue\n2021 | 10M\n2022 | 15M"}
```

## Configuration

| Environment Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `models/paddleocr_vl` | Path to model weights |
| `PADDLE_DEVICE` | `cuda` (auto-detect) | Device: `cuda` or `cpu` |
| `MAX_NEW_TOKENS` | `512` | Max output tokens |
| `PORT` | `8001` | Server port |

## Model Weights

Model weights must be available at `MODEL_PATH`. They are volume-mounted
in Docker, not baked into the image.

Download: The model is a fine-tuned PaddleOCR-VL checkpoint stored at
`models/paddleocr_vl/` in the project root.
