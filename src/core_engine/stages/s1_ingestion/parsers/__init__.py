"""
Stage 1 Parsers Package

Document format parsers for Stage 1 Ingestion.

Each parser converts a specific file format into a list of ParsedPage objects
carrying the rasterized image and extracted document context.
"""

from .base import BaseDocumentParser, ParsedDocument, ParsedPage
from .docx import DocxParser
from .image import ImageParser
from .markdown import MarkdownParser
from .pdf import PDFParser

__all__ = [
    "BaseDocumentParser",
    "ParsedDocument",
    "ParsedPage",
    "PDFParser",
    "DocxParser",
    "MarkdownParser",
    "ImageParser",
]
