"""
PaddleOCR-VL Extraction Server

[DEPRECATED] This file has been migrated to services/ocr/server.py
for Docker-based isolation. This file is kept as a fallback for local
development without Docker.

For Docker usage:
    docker compose up -d ocr-service

For local usage (this file):
    python paddle_server.py

See: services/ocr/README.md for full documentation.
---

Standalone FastAPI microservice (port 8001) that loads PaddleOCR-VL and
exposes a single /extract endpoint.  Must run in a separate venv that has
transformers>=4.45 (incompatible with Vintern's transformers==4.44.2).

Usage:
    # Activate the paddle venv first (or create it with setup_paddle_env.sh)
    python paddle_server.py

    # Or via uvicorn explicitly:
    uvicorn paddle_server:app --port 8001 --reload

Model path: models/paddleocr_vl/
Prompt: "Chart Recognition:" — instructs the model to return a structured
        data table extracted from the chart image.
"""

import io
import logging

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/paddleocr_vl"
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Application + model load (at startup, blocking)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="PaddleOCR-VL Extraction Server", version="1.0.0")

logger.info(f"Loading PaddleOCR-VL from {MODEL_PATH} on {DEVICE} ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)
logger.info(f"PaddleOCR-VL ready on {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — returns model status and device."""
    return {"status": "ok", "model": "PaddleOCR-VL", "device": DEVICE}


@app.post("/extract")
async def extract(image: UploadFile = File(...), max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Extract structured data from a chart image.

    Args:
        image: Chart image (PNG / JPG / WEBP, max ~10 MB)
        max_new_tokens: Maximum output tokens (default 512)

    Returns:
        {"extracted_data": "<structured table text>"}
    """
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

        logger.info(
            f"Extracted | image={image.filename} | chars={len(result)}"
        )
        return {"extracted_data": result.strip()}

    except Exception as exc:
        logger.error(f"Extraction error | image={image.filename} | error={exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "extracted_data": ""},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
