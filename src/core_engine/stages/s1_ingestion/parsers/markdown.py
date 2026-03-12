"""
Markdown Document Parser

Handles Markdown (.md) files by:
  1. Locating all image references using `![alt](path)` or `![alt](url)` syntax.
  2. Loading each referenced local image as a BGR numpy array.
  3. Extracting the paragraphs immediately before and after each image
     reference as surrounding_text.
  4. Using the alt-text or the nearest caption-like paragraph as figure_caption.
  5. Treating the first H1 heading (# ...) as the document title.

URL-referenced images (http/https) are skipped.
Relative paths are resolved relative to the Markdown file's directory.

Dependencies:
  - No additional packages needed beyond NumPy and OpenCV (already required).
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

# Regex: matches ![alt text](image_path_or_url)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Regex: Markdown H1 heading
_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)

# Paragraphs (separated by blank lines) on each side to collect
CONTEXT_PARA_RADIUS: int = 3


class MarkdownParser(BaseDocumentParser):
    """
    Markdown document parser.

    Processes `.md` files and extracts embedded local images with their
    surrounding text context.

    If there are no local image references, a single 100x100 placeholder
    ParsedPage is returned carrying the full document text as surrounding_text
    (useful for downstream context enrichment without actual images).

    Example:
        parser = MarkdownParser()
        doc = parser.parse(Path("report.md"))
        for page in doc.pages:
            print(page.figure_caption, page.surrounding_text)
    """

    def parse(self, path: Path, dpi: int = 150) -> ParsedDocument:
        """
        Parse a Markdown file.

        Args:
            path: Absolute path to the .md file.
            dpi:  Unused for Markdown (no rasterization needed), kept for API compat.

        Returns:
            ParsedDocument with one ParsedPage per local image reference found.
        """
        content = path.read_text(encoding="utf-8", errors="replace")

        document_title = self._extract_title(content, path)
        paragraphs = self._split_paragraphs(content)
        image_refs = self._find_images(paragraphs)

        logger.info(
            f"Parsing Markdown | path={path.name} | images_found={len(image_refs)}"
        )

        pages: List[ParsedPage] = []
        image_page_number = 0

        for para_idx, alt_text, image_path_str in image_refs:
            # Skip external URLs
            if image_path_str.startswith(("http://", "https://")):
                logger.debug(f"Skipping remote image | url={image_path_str}")
                continue

            # Resolve relative path
            image_abs = (path.parent / image_path_str).resolve()
            if not image_abs.exists():
                logger.warning(
                    f"Image not found | path={image_abs} | source={path.name}"
                )
                continue

            img_array = cv2.imread(str(image_abs))
            if img_array is None:
                logger.warning(f"Cannot decode image | path={image_abs}")
                continue

            image_page_number += 1
            surrounding, caption = self._collect_context(
                paragraphs, para_idx, alt_text
            )

            logger.debug(
                f"Markdown image | idx={image_page_number} | "
                f"file={image_abs.name} | has_caption={caption is not None}"
            )

            pages.append(ParsedPage(
                page_number=image_page_number,
                image_array=img_array,
                source_format="md",
                is_scanned=False,
                surrounding_text=surrounding,
                figure_caption=caption,
                document_title=document_title,
            ))

        # No images -- return document text as single placeholder page
        if not pages:
            all_text = self.clean_text(content)
            surrounding_full = self.truncate_context(all_text) if all_text else None
            placeholder = np.full((100, 100, 3), 255, dtype=np.uint8)
            pages.append(ParsedPage(
                page_number=1,
                image_array=placeholder,
                source_format="md",
                is_scanned=False,
                surrounding_text=surrounding_full,
                figure_caption=None,
                document_title=document_title,
            ))

        return ParsedDocument(
            source_path=path,
            document_title=document_title,
            pages=pages,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(content: str) -> List[str]:
        """
        Split Markdown content into logical paragraphs at blank lines.

        Args:
            content: Full Markdown text.

        Returns:
            List of paragraph strings (may include headings, images, etc.).
        """
        return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

    def _find_images(
        self,
        paragraphs: List[str],
    ) -> List[Tuple[int, str, str]]:
        """
        Find all image references in the paragraph list.

        Returns:
            List of (paragraph_index, alt_text, image_path_or_url) tuples.
        """
        results: List[Tuple[int, str, str]] = []
        for idx, para in enumerate(paragraphs):
            for match in _IMAGE_RE.finditer(para):
                alt_text = match.group(1)
                image_path = match.group(2).split(" ")[0]  # strip optional title
                results.append((idx, alt_text, image_path))
        return results

    def _collect_context(
        self,
        paragraphs: List[str],
        image_para_idx: int,
        alt_text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Collect surrounding paragraphs and determine figure caption.

        Caption priority:
          1. Nearest caption-like paragraph (starts with 'Figure', 'Chart', etc.)
          2. Alt-text from the image markdown syntax (e.g., `![Figure 1. ...]`)
          3. None

        Args:
            paragraphs:      Full list of document paragraphs.
            image_para_idx:  Index of the paragraph containing the image ref.
            alt_text:        Alt text from the markdown image syntax.

        Returns:
            (surrounding_text, figure_caption)
        """
        start = max(0, image_para_idx - CONTEXT_PARA_RADIUS)
        end = min(len(paragraphs), image_para_idx + CONTEXT_PARA_RADIUS + 1)

        texts: List[str] = []
        caption: Optional[str] = None

        for idx in range(start, end):
            if idx == image_para_idx:
                continue
            # Strip the image markdown syntax itself from collected text
            para = _IMAGE_RE.sub("", paragraphs[idx]).strip()
            if not para:
                continue
            texts.append(para)
            if caption is None and self.is_caption(para):
                caption = para

        # Fallback to alt-text if no paragraph caption found
        if caption is None and alt_text and self.is_caption(alt_text):
            caption = alt_text

        surrounding = (
            self.truncate_context(self.clean_text("\n".join(texts))) if texts else None
        )
        return surrounding, caption

    @staticmethod
    def _extract_title(content: str, path: Path) -> Optional[str]:
        """
        Extract document title from first H1 heading or filename.

        Args:
            content: Full Markdown text.
            path:    Source file path (fallback).

        Returns:
            Title string.
        """
        match = _H1_RE.search(content)
        if match:
            return match.group(1).strip()
        return path.stem.replace("_", " ").replace("-", " ").title()
