"""
PDF Document Parser

Handles both text-based and scanned (image-only) PDF files using PyMuPDF.

Key behaviours:
  - Text PDF  : extracts surrounding text from each page, detects figure captions.
  - Scanned PDF: detected when page text is below MIN_TEXT_CHARS threshold;
                 the full page is rasterized and is_scanned=True is set.
  - Mixed PDF : treated page-by-page; each page is assessed independently.
  - High DPI  : default 150 DPI (configurable) balances quality vs. file size.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseDocumentParser, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# A page with fewer than this many characters is treated as scanned.
MIN_TEXT_CHARS: int = 40

# Number of text "blocks" on either side of an image block to collect as context.
CONTEXT_BLOCKS_RADIUS: int = 3


class PDFParser(BaseDocumentParser):
    """
    PDF parser powered by PyMuPDF (fitz).

    Supports:
    - Multi-page text PDFs with figure captions and surrounding text
    - Scanned PDFs (no text layer) -- full-page rasterization
    - Mixed PDFs (some pages text, some scanned)

    Example:
        parser = PDFParser()
        doc = parser.parse(Path("report.pdf"), dpi=150)
        for page in doc.pages:
            print(page.page_number, page.is_scanned, page.figure_caption)
    """

    def parse(self, path: Path, dpi: int = 150) -> ParsedDocument:
        """
        Parse PDF file into pages with rasterized images and text context.

        Args:
            path: Absolute path to the PDF file.
            dpi:  Render resolution (72-300 DPI).

        Returns:
            ParsedDocument containing one ParsedPage per PDF page.

        Raises:
            ImportError: PyMuPDF is not installed.
            OSError:     File cannot be opened or is corrupt.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. "
                "Install with: pip install pymupdf"
            ) from exc

        doc = fitz.open(str(path))
        document_title = self._extract_title(doc, path)

        logger.info(
            f"Parsing PDF | path={path.name} | pages={len(doc)} | dpi={dpi}"
        )

        pages: List[ParsedPage] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_idx, page in enumerate(doc, start=1):
            try:
                parsed = self._parse_page(page, page_idx, matrix, document_title)
                pages.append(parsed)
                logger.debug(
                    f"PDF page processed | page={page_idx}/{len(doc)} | "
                    f"scanned={parsed.is_scanned} | "
                    f"has_caption={parsed.figure_caption is not None}"
                )
            except Exception as exc:
                logger.warning(
                    f"PDF page failed | page={page_idx} | path={path.name} | error={exc}"
                )

        doc.close()
        return ParsedDocument(
            source_path=path,
            document_title=document_title,
            pages=pages,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_page(
        self,
        page: "fitz.Page",  # type: ignore[name-defined]
        page_number: int,
        matrix: "fitz.Matrix",  # type: ignore[name-defined]
        document_title: Optional[str],
    ) -> ParsedPage:
        """Rasterize one page and extract its text context."""
        # ---- Rasterize ----
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.height, pix.width, pix.n)

        if pix.n == 3:  # PyMuPDF returns RGB, OpenCV expects BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        # ---- Scanned detection ----
        raw_text = page.get_text("text")
        is_scanned = len(raw_text.strip()) < MIN_TEXT_CHARS

        # ---- Context extraction (text PDFs only) ----
        surrounding_text: Optional[str] = None
        figure_caption: Optional[str] = None

        if not is_scanned:
            surrounding_text, figure_caption = self._extract_context(page, raw_text)

        return ParsedPage(
            page_number=page_number,
            image_array=img_array,
            source_format="pdf",
            is_scanned=is_scanned,
            surrounding_text=surrounding_text,
            figure_caption=figure_caption,
            document_title=document_title,
        )

    def _extract_context(
        self,
        page: "fitz.Page",  # type: ignore[name-defined]
        raw_text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract surrounding text and figure captions from a text-PDF page.

        Strategy:
         1. Use `page.get_text("blocks")` to get text blocks sorted by position.
         2. Identify blocks that look like figure captions.
         3. Take CONTEXT_BLOCKS_RADIUS blocks before/after each image slot.
         4. Concatenate and truncate to CONTEXT_WINDOW_CHARS.

        Returns:
            (surrounding_text, figure_caption)
        """
        try:
            blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,type)
        except Exception:
            blocks = []

        # type==1 is image block, type==0 is text block
        text_blocks: List[str] = []
        caption_candidates: List[str] = []

        for block in blocks:
            if len(block) < 7:
                continue
            block_type = block[6]
            block_text = block[4].strip() if block_type == 0 else ""

            if block_type == 0 and block_text:
                text_blocks.append(block_text)
                if self.is_caption(block_text):
                    caption_candidates.append(block_text)

        figure_caption: Optional[str] = None
        if caption_candidates:
            # Prefer the longest, most informative caption
            figure_caption = max(caption_candidates, key=len)

        # Build surrounding text: all text blocks joined, truncated
        if text_blocks:
            full_text = self.clean_text("\n".join(text_blocks))
            surrounding_text: Optional[str] = self.truncate_context(full_text)
        else:
            surrounding_text = None

        return surrounding_text, figure_caption

    def _extract_title(
        self,
        doc: "fitz.Document",  # type: ignore[name-defined]
        path: Path,
    ) -> Optional[str]:
        """
        Extract document title from PDF metadata or first page heading.

        Falls back to the filename stem if no title is found.

        Args:
            doc:  Open PyMuPDF document object.
            path: Source file path (used for filename fallback).

        Returns:
            Document title string or None.
        """
        # Try PDF metadata first
        meta = doc.metadata or {}
        if meta.get("title"):
            return meta["title"].strip()

        # Try extracting largest-font text from first page as title
        if len(doc) > 0:
            first_page = doc[0]
            try:
                spans = first_page.get_text("dict", flags=0).get("blocks", [])
                candidates: List[Tuple[float, str]] = []
                for block in spans:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            size = span.get("size", 0)
                            if text and len(text) > 5:
                                candidates.append((size, text))
                if candidates:
                    candidates.sort(reverse=True)
                    return candidates[0][1]
            except Exception:
                pass

        # Fallback to filename stem
        return path.stem.replace("_", " ").replace("-", " ").title()
