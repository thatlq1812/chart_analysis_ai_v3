"""
Stage 1: Ingestion & Sanitation

Load and normalize input files (PDF, DOCX, images) into clean images
ready for chart detection.

Responsibilities:
- Accept PDF, DOCX, PNG, JPG, JPEG inputs
- Convert multi-page documents to individual images
- Validate image quality (resolution, blur detection)
- Normalize dimensions and format
- Generate session metadata

Author: Geo-SLM Chart Analysis Team
Date: 2026-01-21
"""

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..exceptions import StageInputError, StageProcessingError
from ..schemas.common import SessionInfo
from ..schemas.stage_outputs import CleanImage, Stage1Output
from .base import BaseStage

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class IngestionConfig(BaseModel):
    """Configuration for Stage 1: Ingestion."""
    
    # DPI settings
    pdf_dpi: int = Field(default=150, ge=72, le=300, description="DPI for PDF rendering")
    
    # Image size limits
    max_image_size: int = Field(default=4096, description="Max dimension in pixels")
    min_image_size: int = Field(default=100, description="Min dimension in pixels")
    
    # Quality thresholds
    min_blur_threshold: float = Field(
        default=100.0,
        description="Minimum Laplacian variance (blur detection)",
    )
    
    # Output settings
    output_format: str = Field(default="PNG", description="Output image format")
    preserve_color: bool = Field(default=True, description="Keep original colors")
    
    # Paths
    output_dir: Optional[Path] = Field(
        default=None,
        description="Directory to save processed images",
    )


# =============================================================================
# Stage 1 Implementation
# =============================================================================

class Stage1Ingestion(BaseStage[Path, Stage1Output]):
    """
    Stage 1: Ingestion & Sanitation.
    
    Transforms diverse input formats into normalized images.
    
    Supported Formats:
    - PDF (multi-page, rendered at configurable DPI)
    - DOCX (embedded images extracted) - FUTURE
    - PNG, JPG, JPEG (direct load)
    
    Example:
        config = IngestionConfig(pdf_dpi=150)
        stage = Stage1Ingestion(config)
        output = stage.process(Path("report.pdf"))
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    
    def __init__(self, config: IngestionConfig | dict | None = None):
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
        
        # Ensure output directory exists
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_critical(self) -> bool:
        """Ingestion is critical - pipeline cannot start without input."""
        return True
    
    def validate_input(self, input_path: Path) -> bool:
        """
        Validate input file exists and is supported.
        
        Args:
            input_path: Path to input file
            
        Returns:
            True if valid
            
        Raises:
            StageInputError: If file doesn't exist or unsupported format
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        
        # Check existence
        if not input_path.exists():
            raise StageInputError(
                message=f"Input file not found: {input_path}",
                stage=self.name,
            )
        
        # Check extension
        ext = input_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise StageInputError(
                message=f"Unsupported file format: {ext}",
                stage=self.name,
                expected_type=str(self.SUPPORTED_EXTENSIONS),
                received_type=ext,
            )
        
        return True
    
    def process(self, input_path: Path) -> Stage1Output:
        """
        Process input file into normalized images.
        
        Args:
            input_path: Path to input file (PDF or image)
            
        Returns:
            Stage1Output with list of CleanImage objects
            
        Raises:
            StageProcessingError: If processing fails
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        
        # Generate session info
        session = self._create_session(input_path)
        
        logger.info(
            f"Processing file | session={session.session_id} | "
            f"path={input_path.name} | type={input_path.suffix}"
        )
        
        # Process based on file type
        ext = input_path.suffix.lower()
        warnings: List[str] = []
        
        try:
            if ext == ".pdf":
                images = self._process_pdf(input_path, session.session_id, warnings)
            else:
                images = self._process_image(input_path, session.session_id, warnings)
        except Exception as e:
            raise StageProcessingError(
                message=f"Failed to process {input_path.name}: {e}",
                stage=self.name,
                recoverable=False,
                original_error=e,
            )
        
        # Update session with page count
        session = SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at,
            source_file=session.source_file,
            total_pages=len(images),
            config_hash=session.config_hash,
        )
        
        logger.info(
            f"Ingestion complete | session={session.session_id} | "
            f"images={len(images)} | warnings={len(warnings)}"
        )
        
        return Stage1Output(
            session=session,
            images=images,
            warnings=warnings,
        )
    
    def _create_session(self, input_path: Path) -> SessionInfo:
        """Create session info for this processing run."""
        # Generate unique session ID
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Hash config for reproducibility
        config_str = self.config.model_dump_json()
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        
        return SessionInfo(
            session_id=session_id,
            created_at=datetime.now(),
            source_file=input_path.absolute(),
            total_pages=1,  # Updated after processing
            config_hash=config_hash,
        )
    
    def _process_pdf(
        self,
        pdf_path: Path,
        session_id: str,
        warnings: List[str],
    ) -> List[CleanImage]:
        """
        Process PDF file into images.
        
        Uses PyMuPDF (fitz) for high-quality rendering.
        
        Args:
            pdf_path: Path to PDF file
            session_id: Current session ID
            warnings: List to append warnings
            
        Returns:
            List of CleanImage objects
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise StageProcessingError(
                message="PyMuPDF not installed. Run: pip install pymupdf",
                stage=self.name,
                recoverable=False,
            )
        
        images: List[CleanImage] = []
        
        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        logger.debug(f"PDF loaded | pages={total_pages} | path={pdf_path.name}")
        
        # Set up output directory
        output_dir = self._get_output_dir(session_id)
        
        # Process each page
        for page_num, page in enumerate(doc, start=1):
            try:
                # Render page to image
                mat = fitz.Matrix(self.config.pdf_dpi / 72, self.config.pdf_dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to numpy array
                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_array.reshape(pix.height, pix.width, pix.n)
                
                # Convert RGB if needed (PyMuPDF returns RGB)
                if pix.n == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Validate and normalize
                clean_img, page_warnings = self._validate_and_normalize(
                    img_array, page_num
                )
                warnings.extend(page_warnings)
                
                if clean_img is None:
                    warnings.append(f"Page {page_num} skipped: failed quality check")
                    continue
                
                # Save image
                output_path = output_dir / f"page_{page_num:04d}.png"
                cv2.imwrite(str(output_path), clean_img)
                
                # Create CleanImage object
                height, width = clean_img.shape[:2]
                is_gray = len(clean_img.shape) == 2 or clean_img.shape[2] == 1
                
                images.append(CleanImage(
                    image_path=output_path,
                    original_path=pdf_path,
                    page_number=page_num,
                    width=width,
                    height=height,
                    is_grayscale=is_gray,
                ))
                
                logger.debug(
                    f"Page processed | page={page_num}/{total_pages} | "
                    f"size={width}x{height}"
                )
                
            except Exception as e:
                warnings.append(f"Page {page_num} error: {str(e)}")
                logger.warning(
                    f"Page processing failed | page={page_num} | error={e}"
                )
        
        doc.close()
        return images
    
    def _process_image(
        self,
        image_path: Path,
        session_id: str,
        warnings: List[str],
    ) -> List[CleanImage]:
        """
        Process single image file.
        
        Args:
            image_path: Path to image file
            session_id: Current session ID
            warnings: List to append warnings
            
        Returns:
            List with single CleanImage (or empty if failed)
        """
        # Load image
        img_array = cv2.imread(str(image_path))
        
        if img_array is None:
            raise StageProcessingError(
                message=f"Failed to load image: {image_path}",
                stage=self.name,
                recoverable=False,
            )
        
        # Validate and normalize
        clean_img, img_warnings = self._validate_and_normalize(img_array, page_num=1)
        warnings.extend(img_warnings)
        
        if clean_img is None:
            warnings.append(f"Image skipped: failed quality check")
            return []
        
        # Save to output directory
        output_dir = self._get_output_dir(session_id)
        output_path = output_dir / f"image_001.png"
        cv2.imwrite(str(output_path), clean_img)
        
        # Create CleanImage
        height, width = clean_img.shape[:2]
        is_gray = len(clean_img.shape) == 2 or clean_img.shape[2] == 1
        
        return [CleanImage(
            image_path=output_path,
            original_path=image_path,
            page_number=1,
            width=width,
            height=height,
            is_grayscale=is_gray,
        )]
    
    def _validate_and_normalize(
        self,
        img: np.ndarray,
        page_num: int,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Validate image quality and normalize dimensions.
        
        Checks:
        - Minimum size
        - Maximum size (resize if needed)
        - Blur detection
        
        Args:
            img: Input image array
            page_num: Page number for logging
            
        Returns:
            Tuple of (normalized image or None, list of warnings)
        """
        warnings: List[str] = []
        height, width = img.shape[:2]
        
        # Check minimum size
        if width < self.config.min_image_size or height < self.config.min_image_size:
            warnings.append(
                f"Page {page_num}: Image too small ({width}x{height})"
            )
            return None, warnings
        
        # Resize if too large
        if width > self.config.max_image_size or height > self.config.max_image_size:
            scale = self.config.max_image_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            warnings.append(
                f"Page {page_num}: Resized from {width}x{height} to {new_width}x{new_height}"
            )
        
        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.config.min_blur_threshold:
            warnings.append(
                f"Page {page_num}: Low quality (blur score: {laplacian_var:.1f})"
            )
            # Don't reject, just warn - let downstream handle it
        
        return img, warnings
    
    def _get_output_dir(self, session_id: str) -> Path:
        """Get or create output directory for session."""
        if self.config.output_dir:
            output_dir = self.config.output_dir / session_id
        else:
            # Default to data/processed/<session_id>
            output_dir = Path("data/processed") / session_id
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_fallback_output(self, input_path: Path) -> Stage1Output:
        """
        Fallback: Return empty output with error info.
        
        Stage 1 is critical, so fallback is minimal.
        """
        session = self._create_session(input_path)
        return Stage1Output(
            session=session,
            images=[],
            warnings=[f"Ingestion failed for {input_path}"],
        )


# =============================================================================
# Utility Functions
# =============================================================================

def detect_file_type(file_path: Path) -> str:
    """
    Detect file type from magic bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected type: 'pdf', 'png', 'jpeg', 'unknown'
    """
    magic_bytes = {
        b"%PDF": "pdf",
        b"\x89PNG": "png",
        b"\xff\xd8\xff": "jpeg",
        b"BM": "bmp",
        b"II*\x00": "tiff",
        b"MM\x00*": "tiff",
    }
    
    with open(file_path, "rb") as f:
        header = f.read(8)
    
    for magic, file_type in magic_bytes.items():
        if header.startswith(magic):
            return file_type
    
    return "unknown"


def estimate_dpi(width: int, height: int, paper_size: str = "letter") -> int:
    """
    Estimate DPI based on image dimensions and assumed paper size.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        paper_size: Paper size ('letter', 'a4')
        
    Returns:
        Estimated DPI
    """
    paper_sizes = {
        "letter": (8.5, 11),  # inches
        "a4": (8.27, 11.69),
    }
    
    paper_w, paper_h = paper_sizes.get(paper_size, paper_sizes["letter"])
    
    # Estimate based on larger dimension
    dpi_w = width / paper_w
    dpi_h = height / paper_h
    
    return int(max(dpi_w, dpi_h))
