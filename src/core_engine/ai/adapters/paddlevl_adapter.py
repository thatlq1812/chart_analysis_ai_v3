"""
PaddleOCR-VL Extraction Adapter

HTTP client that forwards chart images to the PaddleOCR-VL microservice
running on port 8001 (paddle_server.py).

Design decision: PaddleOCR-VL requires transformers>=4.45 which conflicts
with Vintern (transformers==4.44.2), so it runs as a separate process.
This adapter communicates with it via HTTP, keeping the main venv clean.

Provider ID: "paddlevl"
Task supported: TaskType.DATA_EXTRACTION

Health check: GET http://localhost:8001/health
Main call:    POST http://localhost:8001/extract  (multipart image)

If the server is not running, health_check() returns False and the AIRouter
falls back to the next provider in the chain (e.g. gemini with vision).
"""

import io
import logging
from pathlib import Path
from typing import Optional

from .base import AIResponse, BaseAIAdapter
from ..exceptions import AIProviderError

logger = logging.getLogger(__name__)

_PADDLE_URL = "http://localhost:8001"


class PaddleVLAdapter(BaseAIAdapter):
    """
    Adapter for PaddleOCR-VL running as HTTP microservice on port 8001.

    Usage by the router:
        response = await adapter.reason(
            system_prompt="",   # not used
            user_prompt="",     # not used
            image_path="/path/to/chart.png",
        )
        extracted_text = response.content

    The adapter ignores system_prompt / user_prompt and only uses image_path.
    For DATA_EXTRACTION tasks image_path is required.
    """

    provider_id = "paddlevl"

    def __init__(
        self,
        server_url: str = _PADDLE_URL,
        timeout: float = 300.0,
        max_new_tokens: int = 512,
    ) -> None:
        """
        Args:
            server_url: Base URL of paddle_server.py (default http://localhost:8001)
            timeout: HTTP timeout in seconds (PaddleVL can be slow on first request)
            max_new_tokens: Max tokens for PaddleOCR-VL generation
        """
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self._max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # BaseAIAdapter interface
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Ping paddle_server /health endpoint.
        Returns False (without raising) if the server is not running.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._server_url}/health")
                return resp.status_code == 200
        except Exception as exc:
            logger.debug(
                f"PaddleVLAdapter | health check failed | error={exc}"
            )
            return False

    async def reason(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """
        Send chart image to paddle_server and return extracted table text.

        Args:
            system_prompt: Ignored for this adapter (data extraction, not reasoning)
            user_prompt: Ignored for this adapter
            image_path: REQUIRED — path to the chart image file

        Returns:
            AIResponse with content = raw extracted text from PaddleOCR-VL

        Raises:
            AIProviderError: If image_path is missing, file not found, or HTTP error
        """
        if not image_path:
            raise AIProviderError(
                provider=self.provider_id,
                message="image_path is required for DATA_EXTRACTION tasks",
            )

        img_path = Path(image_path)
        if not img_path.exists():
            raise AIProviderError(
                provider=self.provider_id,
                message=f"Image not found: {image_path}",
            )

        try:
            import httpx
            from PIL import Image

            # Read and re-encode to ensure valid PNG
            img = Image.open(img_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._server_url}/extract",
                    files={"image": (img_path.name, buf, "image/png")},
                    params={"max_new_tokens": self._max_new_tokens},
                )
                resp.raise_for_status()
                data = resp.json()

        except ImportError as exc:
            raise AIProviderError(
                provider=self.provider_id,
                message=f"httpx not installed: {exc}",
            ) from exc
        except Exception as exc:
            raise AIProviderError(
                provider=self.provider_id,
                message=f"HTTP request to paddle_server failed: {exc}",
            ) from exc

        extracted = data.get("extracted_data", "")
        if not extracted:
            logger.warning(
                f"PaddleVLAdapter | empty extraction | image={img_path.name}"
            )

        # Confidence heuristic: longer and richer output = more confident
        confidence = min(0.95, 0.5 + len(extracted) / 2000)

        logger.info(
            f"PaddleVLAdapter | ok | image={img_path.name} | "
            f"chars={len(extracted)} | confidence={confidence:.2f}"
        )

        return AIResponse(
            content=extracted,
            model_used="PaddleOCR-VL",
            provider=self.provider_id,
            confidence=confidence,
            raw_response=data,
            success=True,
        )
