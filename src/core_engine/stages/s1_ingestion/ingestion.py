"""
Stage 1: Ingestion & Sanitation (v2.0 - Production Ready)

Transforms diverse input files into normalized page images ready for
chart detection (Stage 2), while preserving rich document context
(surrounding text, figure captions) for Stage 4 Reasoning.

Supported input formats:
  PDF   - Multi-page rendering via PyMuPDF; scanned-PDF auto-detection
  DOCX  - Embedded image extraction via python-docx; LibreOffice layout fallback
  MD    - Local image extraction from Markdown with paragraph context
  PNG   - Direct load
  JPG / JPEG - Direct load
  WebP  - Direct load (OpenCV 4.4+)
  TIFF  - Direct load
  BMP   - Direct load

Output:
  Stage1Output: session metadata + list of CleanImage (image_path + context fields)

Architecture notes:
  - Ingestion is a PURE data transformation stage.
  - It has NO knowledge of downstream stages.
  - Each format is handled by a dedicated parser (Adapter pattern).
  - Image validation / normalization is parser-agnostic and runs centrally.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import cv2
import numpy as np

from ...exceptions import StageInputError, StageProcessingError
from ...schemas.common import SessionInfo
from ...schemas.stage_outputs import CleanImage, Stage1Output
from ..base import BaseStage
from .config import IngestionConfig
from .parsers import (
    BaseDocumentParser,
    DocxParser,
    ImageParser,
    MarkdownParser,
    PDFParser,
    ParsedDocument,
    ParsedPage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parser registry: extension -> parser class
# ---------------------------------------------------------------------------
_PARSER_REGISTRY: Dict[str, Type[BaseDocumentParser]] = {
    # Document formats
    ".pdf":  PDFParser,
    ".docx": DocxParser,
    ".doc":  DocxParser,  # python-docx can open older .doc via compatibility
    ".md":   MarkdownParser,
    # Raster image formats
    ".png":  ImageParser,
    ".jpg":  ImageParser,
    ".jpeg": ImageParser,
    ".webp": ImageParser,
    ".tiff": ImageParser,
    ".tif":  ImageParser,
    ".bmp":  ImageParser,
}


class Stage1Ingestion(BaseStage[Path, Stage1Output]):
    """
    Stage 1: Ingestion & Sanitation.

    Orchestrates document parsing, image validation, normalization,
    and persistent output for all supported input formats.

    Supported extensions: PDF, DOCX, DOC, MD, PNG, JPG, JPEG, WebP, TIFF, BMP.

    Each supported format has a dedicated parser that extracts:
      - Rasterized page images (BGR numpy arrays)
      - Document title
      - Surrounding text context
      - Figure captions

    After parsing, this class applies uniform quality validation and
    normalization (size limits, blur flagging) before persisting to disk
    and returning Stage1Output.

    Example:
        config = IngestionConfig(pdf_dpi=150, extract_context=True)
        stage = Stage1Ingestion(config)
        output = stage.process(Path("annual_report.pdf"))

        for img in output.images:
            print(img.image_path, img.figure_caption, img.surrounding_text[:100])
    """

    SUPPORTED_EXTENSIONS = frozenset(_PARSER_REGISTRY.keys())

    def __init__(self, config: IngestionConfig | dict | None = None) -> None:
        """
        Initialize ingestion stage.

        Args:
            config: Stage configuration. Uses defaults if None.
        """
        if config is None:
            config = IngestionConfig()
        elif isinstance(config, dict):
            config = IngestionConfig(**config)

        super().__init__(config)
        self.config: IngestionConfig = config

        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    @property
    def is_critical(self) -> bool:
        """Ingestion is critical -- pipeline cannot start without valid input."""
        return True

    def validate_input(self, input_path: Path) -> bool:
        """
        Validate input file exists and has a supported extension.

        Args:
            input_path: Path to input file.

        Returns:
            True if valid.

        Raises:
            StageInputError: If file is missing or format is unsupported.
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)

        if not input_path.exists():
            raise StageInputError(
                message=f"Input file not found: {input_path}",
                stage=self.name,
            )

        ext = input_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise StageInputError(
                message=(
                    f"Unsupported file format: '{ext}'. "
                    f"Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
                ),
                stage=self.name,
                expected_type=str(sorted(self.SUPPORTED_EXTENSIONS)),
                received_type=ext,
            )

        return True

    def process(self, input_path: Path) -> Stage1Output:
        """
        Process input file into a collection of normalized CleanImage objects.

        Steps:
          1. Create session metadata.
          2. Select parser from registry based on file extension.
          3. Parse document -> list of ParsedPage.
          4. Validate and normalize each page image.
          5. Persist normalized images to output directory.
          6. Build CleanImage objects with context fields.
          7. Return Stage1Output.

        Args:
            input_path: Path to input file.

        Returns:
            Stage1Output with session metadata and list of CleanImage.

        Raises:
            StageProcessingError: If processing fails critically.
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)

        session = self._create_session(input_path)

        logger.info(
            f"Stage 1 start | session={session.session_id} | "
            f"file={input_path.name} | format={input_path.suffix.lower()}"
        )

        warnings: List[str] = []

        try:
            parsed_doc = self._parse_document(input_path)
        except (ImportError, ValueError, OSError) as exc:
            raise StageProcessingError(
                message=f"Failed to parse '{input_path.name}': {exc}",
                stage=self.name,
                recoverable=False,
                original_error=exc,
            ) from exc

        output_dir = self._get_output_dir(session.session_id)
        images: List[CleanImage] = []

        for parsed_page in parsed_doc.pages:
            clean, page_warnings = self._process_parsed_page(
                parsed_page=parsed_page,
                source_path=input_path,
                output_dir=output_dir,
                session_id=session.session_id,
            )
            warnings.extend(page_warnings)

            if clean is not None:
                images.append(clean)

        # Update session total_pages
        session = SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            source_file=session.source_file,
            total_pages=len(images),
            config_hash=session.config_hash,
        )

        logger.info(
            f"Stage 1 complete | session={session.session_id} | "
            f"images={len(images)} | warnings={len(warnings)}"
        )

        return Stage1Output(
            session=session,
            images=images,
            warnings=warnings,
        )

    def get_fallback_output(self, input_path: Path) -> Stage1Output:
        """Minimal fallback output when critical error occurs."""
        session = self._create_session(input_path)
        return Stage1Output(
            session=session,
            images=[],
            warnings=[f"Ingestion failed for {input_path}"],
        )

    # ------------------------------------------------------------------
    # Parsing dispatch
    # ------------------------------------------------------------------

    def _parse_document(self, path: Path) -> ParsedDocument:
        """
        Select the correct parser for the file and run it.

        Args:
            path: Source file path.

        Returns:
            ParsedDocument from the appropriate parser.
        """
        ext = path.suffix.lower()
        parser_class = _PARSER_REGISTRY[ext]
        parser = parser_class()

        logger.debug(
            f"Dispatching parser | ext={ext} | parser={parser_class.__name__}"
        )

        parsed = parser.parse(path, dpi=self.config.pdf_dpi)

        # If context extraction is disabled, strip context fields
        if not self.config.extract_context:
            from .parsers.base import ParsedPage as PP
            parsed.pages = [
                PP(
                    page_number=p.page_number,
                    image_array=p.image_array,
                    source_format=p.source_format,
                    is_scanned=p.is_scanned,
                    surrounding_text=None,
                    figure_caption=None,
                    document_title=None,
                )
                for p in parsed.pages
            ]

        return parsed

    # ------------------------------------------------------------------
    # Per-page processing
    # ------------------------------------------------------------------

    def _process_parsed_page(
        self,
        parsed_page: ParsedPage,
        source_path: Path,
        output_dir: Path,
        session_id: str,
    ) -> Tuple[Optional[CleanImage], List[str]]:
        """
        Validate, normalize, and persist a single parsed page.

        Args:
            parsed_page: Page from the document parser.
            source_path: Original source file path (stored in CleanImage.original_path).
            output_dir:  Session output directory.
            session_id:  Current session ID (for logging).

        Returns:
            (CleanImage or None if rejected, list of warning strings)
        """
        warnings: List[str] = []
        page_num = parsed_page.page_number

        img = parsed_page.image_array
        if img is None or img.size == 0:
            warnings.append(f"Page {page_num}: empty image array, skipped")
            return None, warnings

        normalized, norm_warnings = self._validate_and_normalize(img, page_num)
        warnings.extend(norm_warnings)

        if normalized is None:
            return None, warnings

        # Save to disk
        ext = "png" if self.config.output_format.upper() == "PNG" else "jpg"
        output_path = output_dir / f"page_{page_num:04d}.{ext}"
        cv2.imwrite(str(output_path), normalized)

        height, width = normalized.shape[:2]
        is_gray = len(normalized.shape) == 2 or (
            len(normalized.shape) == 3 and normalized.shape[2] == 1
        )

        clean = CleanImage(
            image_path=output_path,
            original_path=source_path.absolute(),
            page_number=page_num,
            width=width,
            height=height,
            is_grayscale=is_gray,
            source_format=parsed_page.source_format,
            is_scanned=parsed_page.is_scanned,
            surrounding_text=parsed_page.surrounding_text,
            figure_caption=parsed_page.figure_caption,
            document_title=parsed_page.document_title,
        )

        logger.debug(
            f"Page saved | session={session_id} | page={page_num} | "
            f"size={width}x{height} | scanned={parsed_page.is_scanned}"
        )

        return clean, warnings

    def _validate_and_normalize(
        self,
        img: np.ndarray,
        page_num: int,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Validate image dimensions and quality, normalize if needed.

        Checks:
          - Minimum dimension (reject if too small)
          - Maximum dimension (resize if too large)
          - Blur score via Laplacian variance (warn if blurry)

        Args:
            img:      BGR numpy array.
            page_num: Page number for logging/warning messages.

        Returns:
            (normalized image or None if rejected, list of warnings)
        """
        warnings: List[str] = []
        height, width = img.shape[:2]

        # Reject too-small images
        if width < self.config.min_image_size or height < self.config.min_image_size:
            warnings.append(
                f"Page {page_num}: image too small ({width}x{height}), rejected."
            )
            return None, warnings

        # Downscale if too large
        if width > self.config.max_image_size or height > self.config.max_image_size:
            scale = self.config.max_image_size / max(width, height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            warnings.append(
                f"Page {page_num}: resized {width}x{height} -> {new_w}x{new_h}."
            )

        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.config.min_blur_threshold:
            warnings.append(
                f"Page {page_num}: low sharpness (blur_score={blur_score:.1f}). "
                "Image may be blurry."
            )

        # Grayscale conversion if configured
        if not self.config.preserve_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, warnings

    # ------------------------------------------------------------------
    # Session and I/O helpers
    # ------------------------------------------------------------------

    def _create_session(self, input_path: Path) -> SessionInfo:
        """Create a new session for this processing run."""
        session_id = (
            f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        config_hash = hashlib.md5(
            self.config.model_dump_json().encode()
        ).hexdigest()[:16]

        return SessionInfo(
            session_id=session_id,
            created_at=datetime.now(),
            source_file=input_path.absolute(),
            total_pages=1,
            config_hash=config_hash,
        )

    def _get_output_dir(self, session_id: str) -> Path:
        """Resolve and create the session output directory."""
        base = self.config.output_dir if self.config.output_dir else Path("data/processed")
        output_dir = base / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
