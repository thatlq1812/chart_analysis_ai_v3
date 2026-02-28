#!/usr/bin/env python3
"""
Extract Text-Only Pages from PDFs as Backgrounds

This script scans PDF files (e.g., ArXiv papers) and extracts pages that:
1. Contain mostly text (minimal or no images)
2. Are suitable as document backgrounds for synthetic data generation

The resulting images will be used as realistic backgrounds when pasting
chart images to create training data for YOLO chart detection.

Why Text-Only Pages?
- If we use blank white backgrounds, YOLO learns "white = background"
- Real documents have text, headers, equations, tables
- Text-only pages teach YOLO: "ignore text, find charts"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.error("PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

from PIL import Image
import io


class BackgroundExtractor:
    """
    Extract text-heavy pages from PDFs as training backgrounds.
    
    Criteria for a "good" background page:
    1. Has substantial text content
    2. Has NO images, or only small images (logos, icons)
    3. Not a title page (usually sparse)
    4. Not a references-only page (too dense, similar across papers)
    """
    
    def __init__(
        self,
        output_dir: Path,
        dpi: int = 150,
        max_image_area_ratio: float = 0.05,  # Max 5% of page can be images
        min_text_chars: int = 500,  # Minimum characters on page
        max_text_chars: int = 10000,  # Max chars (avoid reference pages)
    ):
        """
        Initialize the extractor.
        
        Args:
            output_dir: Directory to save extracted background images
            dpi: Resolution for rendering (150 DPI = good quality, reasonable size)
            max_image_area_ratio: Maximum ratio of page area that can be images
            min_text_chars: Minimum text characters for a valid page
            max_text_chars: Maximum text characters (filter out reference pages)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dpi = dpi
        self.max_image_area_ratio = max_image_area_ratio
        self.min_text_chars = min_text_chars
        self.max_text_chars = max_text_chars
        
        self.stats = {
            "pdfs_processed": 0,
            "pages_scanned": 0,
            "pages_extracted": 0,
            "pages_skipped_has_images": 0,
            "pages_skipped_too_sparse": 0,
            "pages_skipped_too_dense": 0,
            "pages_skipped_errors": 0,
        }
    
    def _calculate_image_coverage(self, page: fitz.Page) -> float:
        """
        Calculate what fraction of the page is covered by images.
        
        Returns:
            Ratio of image area to page area (0.0 to 1.0)
        """
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            return 1.0  # Invalid page
        
        image_list = page.get_images(full=True)
        total_image_area = 0
        
        for img_info in image_list:
            try:
                # Get image bounding box on page
                xref = img_info[0]
                img_rects = page.get_image_rects(xref)
                for rect in img_rects:
                    total_image_area += rect.width * rect.height
            except Exception:
                # If we can't get rect, assume small image
                pass
        
        return total_image_area / page_area
    
    def _is_valid_background(self, page: fitz.Page) -> Tuple[bool, str]:
        """
        Check if a page is suitable as a background.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check text content
        text = page.get_text()
        text_len = len(text.strip())
        
        if text_len < self.min_text_chars:
            return False, "too_sparse"
        
        if text_len > self.max_text_chars:
            return False, "too_dense"
        
        # Check image coverage
        image_coverage = self._calculate_image_coverage(page)
        
        if image_coverage > self.max_image_area_ratio:
            return False, "has_images"
        
        return True, "valid"
    
    def _render_page(self, page: fitz.Page) -> Image.Image:
        """Render a PDF page to PIL Image."""
        # Calculate zoom for target DPI
        zoom = self.dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Render to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    
    def extract_from_pdf(self, pdf_path: Path, max_pages_per_pdf: int = 5) -> List[Path]:
        """
        Extract background pages from a single PDF.
        
        Args:
            pdf_path: Path to PDF file
            max_pages_per_pdf: Maximum backgrounds to extract per PDF
            
        Returns:
            List of paths to saved background images
        """
        saved_paths = []
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(f"Failed to open PDF | path={pdf_path} | error={e}")
            return saved_paths
        
        pages_extracted = 0
        
        for page_num in range(len(doc)):
            if pages_extracted >= max_pages_per_pdf:
                break
            
            self.stats["pages_scanned"] += 1
            
            try:
                page = doc[page_num]
                is_valid, reason = self._is_valid_background(page)
                
                if not is_valid:
                    self.stats[f"pages_skipped_{reason}"] += 1
                    continue
                
                # Render and save
                img = self._render_page(page)
                
                # Generate filename: pdfname_page{n}.png
                pdf_name = pdf_path.stem[:50]  # Truncate long names
                output_name = f"{pdf_name}_page{page_num + 1:03d}.png"
                output_path = self.output_dir / output_name
                
                img.save(output_path, "PNG")
                saved_paths.append(output_path)
                
                self.stats["pages_extracted"] += 1
                pages_extracted += 1
                
            except Exception as e:
                logger.debug(f"Error processing page | pdf={pdf_path.name} | page={page_num} | error={e}")
                self.stats["pages_skipped_errors"] += 1
        
        doc.close()
        self.stats["pdfs_processed"] += 1
        
        return saved_paths
    
    def extract_from_directory(
        self,
        pdf_dir: Path,
        max_backgrounds: int = 2000,
        max_pages_per_pdf: int = 3,
    ) -> dict:
        """
        Extract backgrounds from all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            max_backgrounds: Target number of background images
            max_pages_per_pdf: Maximum pages to extract per PDF
            
        Returns:
            Statistics dictionary
        """
        pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files | dir={pdf_dir}")
        
        if not pdf_files:
            logger.warning("No PDF files found")
            return self.stats
        
        for i, pdf_path in enumerate(pdf_files):
            if self.stats["pages_extracted"] >= max_backgrounds:
                logger.info(f"Reached target of {max_backgrounds} backgrounds")
                break
            
            if i % 50 == 0:
                logger.info(
                    f"Progress | pdfs={i}/{len(pdf_files)} | "
                    f"backgrounds={self.stats['pages_extracted']}/{max_backgrounds}"
                )
            
            self.extract_from_pdf(pdf_path, max_pages_per_pdf)
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract text-only pages from PDFs as backgrounds for synthetic data"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_pdfs",
        help="Directory containing PDF files (default: data/raw_pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic_source" / "backgrounds",
        help="Output directory for background images",
    )
    parser.add_argument(
        "--max-backgrounds",
        type=int,
        default=2000,
        help="Maximum number of backgrounds to extract (default: 2000)",
    )
    parser.add_argument(
        "--max-per-pdf",
        type=int,
        default=3,
        help="Maximum backgrounds per PDF (default: 3)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Render DPI (default: 150)",
    )
    parser.add_argument(
        "--max-image-ratio",
        type=float,
        default=0.05,
        help="Maximum image area ratio (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--min-text",
        type=int,
        default=500,
        help="Minimum text characters (default: 500)",
    )
    parser.add_argument(
        "--max-text",
        type=int,
        default=8000,
        help="Maximum text characters (default: 8000, filters reference pages)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.info("=" * 60)
    logger.info("PDF Background Extractor")
    logger.info("=" * 60)
    
    # Validate input directory
    if not args.pdf_dir.exists():
        logger.error(f"PDF directory not found | path={args.pdf_dir}")
        sys.exit(1)
    
    logger.info(f"Configuration:")
    logger.info(f"  PDF source:      {args.pdf_dir}")
    logger.info(f"  Output:          {args.output_dir}")
    logger.info(f"  Target:          {args.max_backgrounds} backgrounds")
    logger.info(f"  Max per PDF:     {args.max_per_pdf}")
    logger.info(f"  DPI:             {args.dpi}")
    logger.info(f"  Max image ratio: {args.max_image_ratio:.0%}")
    logger.info(f"  Text range:      {args.min_text}-{args.max_text} chars")
    logger.info("-" * 60)
    
    # Initialize extractor
    extractor = BackgroundExtractor(
        output_dir=args.output_dir,
        dpi=args.dpi,
        max_image_area_ratio=args.max_image_ratio,
        min_text_chars=args.min_text,
        max_text_chars=args.max_text,
    )
    
    # Extract backgrounds
    stats = extractor.extract_from_directory(
        pdf_dir=args.pdf_dir,
        max_backgrounds=args.max_backgrounds,
        max_pages_per_pdf=args.max_per_pdf,
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Extraction Complete!")
    logger.info("=" * 60)
    logger.info(f"Statistics:")
    logger.info(f"  PDFs processed:     {stats['pdfs_processed']}")
    logger.info(f"  Pages scanned:      {stats['pages_scanned']}")
    logger.info(f"  Pages extracted:    {stats['pages_extracted']}")
    logger.info(f"  Skipped (images):   {stats['pages_skipped_has_images']}")
    logger.info(f"  Skipped (sparse):   {stats['pages_skipped_too_sparse']}")
    logger.info(f"  Skipped (dense):    {stats['pages_skipped_too_dense']}")
    logger.info(f"  Skipped (errors):   {stats['pages_skipped_errors']}")
    logger.info("")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("")
    logger.info("Next step: Generate synthetic dataset")
    logger.info(f"  python scripts/generate_synthetic_dataset.py \\")
    logger.info(f"      --background-dir {args.output_dir} \\")
    logger.info(f"      --num-samples 10000")


if __name__ == "__main__":
    main()
