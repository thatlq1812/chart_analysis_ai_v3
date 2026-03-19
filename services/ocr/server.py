"""
PaddleOCR-VL Extraction Server

Standalone FastAPI microservice for chart data extraction using PaddleOCR-VL.
Runs in an isolated environment (Docker or separate venv) due to
transformers>=4.45 dependency conflict with main project environment.

Endpoints:
    GET  /health   - Health check (model status, device info)
    POST /extract  - Extract structured data from chart image

Usage:
    # Via Docker (recommended):
    docker compose up -d ocr-service

    # Via uvicorn directly (requires separate venv with transformers>=4.45):
    uvicorn server:app --host 0.0.0.0 --port 8001

Model path: /app/models/paddleocr_vl (Docker) or models/paddleocr_vl (local)
"""

import io
import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
logger = logging.getLogger("ocr_service")

# ---------------------------------------------------------------------------
# Config (environment-aware)
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "models/paddleocr_vl")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
DEVICE = os.environ.get("PADDLE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.environ.get("PORT", "8001"))

# ---------------------------------------------------------------------------
# Application + model load
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PaddleOCR-VL Extraction Server",
    version="1.1.0",
    description="Isolated microservice for chart data extraction using PaddleOCR-VL",
)

logger.info(f"Loading PaddleOCR-VL | path={MODEL_PATH} | device={DEVICE}")
_load_start = time.time()

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

_load_time = time.time() - _load_start
logger.info(f"PaddleOCR-VL ready | device={DEVICE} | load_time={_load_time:.1f}s")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Health check -- returns model status and device."""
    return {
        "status": "ok",
        "model": "PaddleOCR-VL",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "version": "1.1.0",
    }


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    Extract structured data from a chart image.

    Args:
        image: Chart image file (PNG / JPG / WEBP, max ~10 MB)
        max_new_tokens: Maximum output tokens (default 512)

    Returns:
        {"extracted_data": "<structured table text>"}
    """
    start_time = time.time()
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Chart Recognition:"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": 560, "longest_edge": 1024 * 28 * 28}
            },
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        result = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] : -1],
            skip_special_tokens=True,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Extracted | image={image.filename} | chars={len(result)} | time={elapsed:.2f}s"
        )
        return {"extracted_data": result.strip()}

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(
            f"Extraction error | image={image.filename} | error={exc} | time={elapsed:.2f}s"
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "extracted_data": ""},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
