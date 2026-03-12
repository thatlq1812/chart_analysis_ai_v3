"""
Base Document Parser

Abstract interface for all document format parsers in Stage 1.

Every parser converts a source file into a list of ParsedPage objects,
each carrying:
  - the rasterized image (BGR numpy array)
  - extracted text context surrounding any figure locations
  - figure caption if found
  - document metadata (title, format, scan status)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Contracts
# =============================================================================


@dataclass
class ParsedPage:
    """
    Intermediate representation of a single page/frame from a document.

    Produced by the parser, consumed by Stage1Ingestion for validation,
    normalization, and CleanImage construction.
    """

    page_number: int
    """1-based page number within the source document."""

    image_array: np.ndarray
    """BGR numpy array (OpenCV convention). Shape: (H, W, 3) or (H, W)."""

    source_format: str
    """File format identifier: 'pdf', 'docx', 'md', 'png', 'jpg', 'webp', etc."""

    is_scanned: bool = False
    """True when the page contains no selectable text layer (scanned document)."""

    surrounding_text: Optional[str] = None
    """
    Text paragraphs extracted from the same page / nearby sections of the
    source document, used as contextual grounding for Stage 4 LLM reasoning.
    """

    figure_caption: Optional[str] = None
    """
    Caption extracted from the source document, e.g.
    'Figure 3. Monthly revenue by region, Q1-Q4 2024'.
    None when no caption-like text is detected near the image slot.
    """

    document_title: Optional[str] = None
    """
    Title of the source document (PDF metadata, DOCX core properties,
    first H1 heading in Markdown, or filename stem as fallback).
    """


@dataclass
class ParsedDocument:
    """
    All pages extracted from one source document.

    Returned by `BaseDocumentParser.parse()`.
    """

    source_path: Path
    document_title: Optional[str]
    pages: List[ParsedPage] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)


# =============================================================================
# Abstract Base
# =============================================================================


class BaseDocumentParser(ABC):
    """
    Abstract base class for document format parsers.

    Subclasses implement `parse()` for a specific file format.
    The parser is responsible ONLY for:
      1. Reading / rasterizing the source file
      2. Extracting text context and captions
      3. Returning ParsedDocument

    Image validation, normalization, and output persistence are handled
    by Stage1Ingestion (the consumer of this parser).

    Subclasses:
        PDFParser       - PyMuPDF, handles both text-PDF and scanned-PDF
        DocxParser      - python-docx, XML-based context extraction
        MarkdownParser  - regex-based image + context extraction
        ImageParser     - Direct rasterization of standalone image files
    """

    CONTEXT_WINDOW_CHARS: int = 800
    """
    Maximum characters of surrounding text to extract per page.
    Keeps prompt size reasonable for Stage 4 LLM inference.
    """

    CAPTION_PATTERNS: tuple[str, ...] = (
        "figure",
        "fig.",
        "chart",
        "graph",
        "diagram",
        "exhibit",
        "illustration",
        "bieu do",
        "hinh ",
    )
    """
    Lowercase trigger words for caption detection.
    Includes common English and Vietnamese terms.
    """

    @abstractmethod
    def parse(self, path: Path, dpi: int = 150) -> ParsedDocument:
        """
        Parse a document file and return all pages as ParsedDocument.

        Args:
            path: Absolute path to the source file.
            dpi:  Target resolution for rasterization (PDF / DOCX only).

        Returns:
            ParsedDocument with all extracted pages and metadata.

        Raises:
            ImportError:  Required third-party library is not installed.
            ValueError:   File is corrupt or unsupported variant.
            OSError:      File does not exist or cannot be read.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_caption(cls, text: str) -> bool:
        """
        Heuristic: does this line look like a figure/chart caption?

        Args:
            text: A single paragraph or line of text (lowercased internally).

        Returns:
            True if the text starts with or contains a known caption trigger.
        """
        lower = text.strip().lower()
        return any(lower.startswith(pat) or pat in lower for pat in cls.CAPTION_PATTERNS)

    @classmethod
    def clean_text(cls, raw: str) -> str:
        """
        Collapse whitespace and strip boilerplate from extracted text.

        Args:
            raw: Raw string from PDF/DOCX text extraction.

        Returns:
            Normalised, stripped string.
        """
        import re

        # Collapse multiple spaces/newlines
        text = re.sub(r"[ \t]+", " ", raw)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def truncate_context(cls, text: str) -> str:
        """Truncate surrounding_text to CONTEXT_WINDOW_CHARS."""
        if len(text) <= cls.CONTEXT_WINDOW_CHARS:
            return text
        return text[: cls.CONTEXT_WINDOW_CHARS] + "..."
