"""
Image Document Parser

Handles standalone image files: PNG, JPG/JPEG, WebP, TIFF, BMP.

This is the simplest parser -- a single image produces exactly one ParsedPage
with no surrounding text (there is no document context to extract).

Supported formats match OpenCV's imread capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from .base import BaseDocumentParser, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Map file extension to canonical format name
_FORMAT_MAP: dict[str, str] = {
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".webp": "webp",
    ".tiff": "tiff",
    ".tif": "tiff",
    ".bmp": "bmp",
}


class ImageParser(BaseDocumentParser):
    """
    Standalone image file parser.

    Reads the image directly via OpenCV and wraps it in a single ParsedPage.
    No text context is available for pure image inputs.

    Supported extensions: .png, .jpg, .jpeg, .webp, .tiff, .tif, .bmp

    Example:
        parser = ImageParser()
        doc = parser.parse(Path("chart.png"))
        assert len(doc.pages) == 1
    """

    def parse(self, path: Path, dpi: int = 150) -> ParsedDocument:
        """
        Load a single image file.

        Args:
            path: Absolute path to the image file.
            dpi:  Unused for raster images, kept for API compatibility.

        Returns:
            ParsedDocument with exactly one ParsedPage.

        Raises:
            ValueError: Image cannot be decoded by OpenCV.
            OSError:    File does not exist.
        """
        if not path.exists():
            raise OSError(f"Image file not found: {path}")

        img_array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError(
                f"OpenCV could not decode the image file: {path}. "
                "Verify the file is a valid image format."
            )

        source_format = _FORMAT_MAP.get(path.suffix.lower(), "image")
        document_title: Optional[str] = path.stem.replace("_", " ").replace("-", " ").title()

        logger.info(
            f"Parsing image | path={path.name} | "
            f"format={source_format} | "
            f"size={img_array.shape[1]}x{img_array.shape[0]}"
        )

        return ParsedDocument(
            source_path=path,
            document_title=document_title,
            pages=[
                ParsedPage(
                    page_number=1,
                    image_array=img_array,
                    source_format=source_format,
                    is_scanned=False,
                    surrounding_text=None,
                    figure_caption=None,
                    document_title=document_title,
                )
            ],
        )
