"""
PDF Miner - Extract chart images from PDF documents

This service processes PDF files to extract chart/figure images
along with their captions and surrounding context.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from loguru import logger
from PIL import Image

from ..config import DataFactoryConfig, IMAGES_DIR, METADATA_DIR
from ..schemas import ArxivPaper, BoundingBox, ChartImage, ChartType, DataSource


class PDFMiner:
    """
    Extract chart images from PDF documents.
    
    Uses PyMuPDF (fitz) for PDF processing with caption extraction.
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        self._ensure_fitz()
        
        # Ensure output directories exist
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _ensure_fitz(self) -> None:
        """Ensure PyMuPDF is available."""
        try:
            import fitz
            self.fitz = fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed: pip install pymupdf")
    
    def process_pdf(self, pdf_path: Path, paper: Optional[ArxivPaper] = None) -> List[ChartImage]:
        """
        Process a PDF and extract chart images.
        
        Args:
            pdf_path: Path to PDF file
            paper: Optional ArxivPaper metadata
            
        Returns:
            List of extracted ChartImage objects
        """
        if not pdf_path.exists():
            logger.error(f"PDF not found | path={pdf_path}")
            return []
        
        paper_id = paper.safe_id if paper else pdf_path.stem
        logger.info(f"Processing PDF | paper_id={paper_id} | path={pdf_path}")
        
        extracted_images: List[ChartImage] = []
        
        try:
            doc = self.fitz.open(pdf_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # Extract images from page
                page_images = self._extract_page_images(
                    doc, page, page_num, paper_id
                )
                
                # Extract captions for images
                page_text = page.get_text("dict")
                
                for img_data in page_images:
                    # Try to find caption
                    caption = self._find_caption(page_text, img_data["bbox"])
                    context = self._extract_context(page_text, img_data["bbox"])
                    
                    # Create ChartImage object
                    chart_image = self._save_image(
                        img_data=img_data,
                        paper_id=paper_id,
                        page_num=page_num + 1,
                        caption=caption,
                        context=context,
                        source_paper=paper,
                    )
                    
                    if chart_image:
                        extracted_images.append(chart_image)
            
            doc.close()
            
            logger.info(
                f"PDF processed | paper_id={paper_id} | "
                f"pages={total_pages} | images_extracted={len(extracted_images)}"
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed | paper_id={paper_id} | error={e}")
        
        return extracted_images
    
    def _extract_page_images(
        self,
        doc,
        page,
        page_num: int,
        paper_id: str,
    ) -> List[dict]:
        """
        Extract images from a PDF page.
        
        Returns:
            List of dicts with image data and metadata
        """
        images = []
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                
                # Extract image
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]
                
                # Filter by size
                if width < self.config.quality.min_width or height < self.config.quality.min_height:
                    logger.debug(
                        f"Image too small, skipping | page={page_num} | "
                        f"size={width}x{height}"
                    )
                    continue
                
                # Get image position on page
                bbox = self._get_image_bbox(page, xref)
                
                images.append({
                    "xref": xref,
                    "bytes": image_bytes,
                    "ext": image_ext,
                    "width": width,
                    "height": height,
                    "bbox": bbox,
                    "index": img_index,
                })
                
            except Exception as e:
                logger.warning(f"Failed to extract image | page={page_num} | error={e}")
                continue
        
        return images
    
    def _get_image_bbox(self, page, xref: int) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of image on page."""
        try:
            for item in page.get_images():
                if item[0] == xref:
                    # Get image rectangle
                    img_rect = page.get_image_rects(item)
                    if img_rect:
                        rect = img_rect[0]
                        return (int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1))
        except Exception:
            pass
        return None
    
    def _find_caption(self, page_text: dict, img_bbox: Optional[Tuple]) -> Optional[str]:
        """
        Find caption text for an image.
        
        Looks for "Figure X:" or "Fig. X:" patterns near the image.
        """
        if not img_bbox:
            return None
        
        caption_patterns = [
            r"Figure\s+\d+[.:]\s*(.+)",
            r"Fig\.\s*\d+[.:]\s*(.+)",
            r"FIGURE\s+\d+[.:]\s*(.+)",
            r"Chart\s+\d+[.:]\s*(.+)",
        ]
        
        # Get text blocks below the image
        blocks = page_text.get("blocks", [])
        img_bottom = img_bbox[3]
        
        caption_candidates = []
        
        for block in blocks:
            if block.get("type") != 0:  # Text block
                continue
            
            block_bbox = block.get("bbox", (0, 0, 0, 0))
            block_top = block_bbox[1]
            
            # Look for text below image (within 100 pixels)
            if img_bottom <= block_top <= img_bottom + 100:
                # Extract text from block
                text_lines = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_lines.append(span.get("text", ""))
                
                block_text = " ".join(text_lines).strip()
                
                # Check if it matches caption pattern
                for pattern in caption_patterns:
                    match = re.search(pattern, block_text, re.IGNORECASE)
                    if match:
                        return block_text
                
                caption_candidates.append(block_text)
        
        # Return first candidate if no pattern match
        if caption_candidates:
            return caption_candidates[0][:500]  # Limit length
        
        return None
    
    def _extract_context(self, page_text: dict, img_bbox: Optional[Tuple]) -> Optional[str]:
        """
        Extract surrounding text context for an image.
        """
        if not img_bbox:
            return None
        
        blocks = page_text.get("blocks", [])
        context_parts = []
        
        img_top, img_bottom = img_bbox[1], img_bbox[3]
        
        for block in blocks:
            if block.get("type") != 0:
                continue
            
            block_bbox = block.get("bbox", (0, 0, 0, 0))
            block_bottom = block_bbox[3]
            block_top = block_bbox[1]
            
            # Get text above image (within 200 pixels)
            if img_top - 200 <= block_bottom <= img_top:
                text_lines = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_lines.append(span.get("text", ""))
                context_parts.append(" ".join(text_lines))
        
        if context_parts:
            return " ".join(context_parts)[:1000]
        
        return None
    
    def _save_image(
        self,
        img_data: dict,
        paper_id: str,
        page_num: int,
        caption: Optional[str],
        context: Optional[str],
        source_paper: Optional[ArxivPaper],
    ) -> Optional[ChartImage]:
        """
        Save extracted image and create ChartImage object.
        """
        image_id = f"{paper_id}_p{page_num:02d}_img{img_data['index']:02d}"
        image_path = IMAGES_DIR / f"{image_id}.png"
        metadata_path = METADATA_DIR / f"{image_id}.json"
        
        try:
            # Convert to PNG and save
            from io import BytesIO
            img = Image.open(BytesIO(img_data["bytes"]))
            
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            img.save(image_path, "PNG", optimize=True)
            
            # Get final size
            file_size = image_path.stat().st_size
            
            # Create ChartImage object
            chart_image = ChartImage(
                image_id=image_id,
                source=DataSource.ARXIV,
                parent_paper_id=paper_id,
                page_number=page_num,
                image_path=image_path,
                width=img_data["width"],
                height=img_data["height"],
                file_size_bytes=file_size,
                caption_text=caption,
                context_text=context,
                bbox=BoundingBox(
                    x_min=img_data["bbox"][0],
                    y_min=img_data["bbox"][1],
                    x_max=img_data["bbox"][2],
                    y_max=img_data["bbox"][3],
                ) if img_data["bbox"] else None,
            )
            
            # Save metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(chart_image.model_dump(mode="json"), f, indent=2, default=str)
            
            logger.debug(f"Saved image | image_id={image_id} | size={img_data['width']}x{img_data['height']}")
            return chart_image
            
        except Exception as e:
            logger.warning(f"Failed to save image | image_id={image_id} | error={e}")
            return None
    
    def process_all_pdfs(self, pdf_dir: Path, papers: Optional[List[ArxivPaper]] = None) -> List[ChartImage]:
        """
        Process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs
            papers: Optional list of ArxivPaper objects for metadata
            
        Returns:
            List of all extracted ChartImage objects
        """
        all_images: List[ChartImage] = []
        
        # Build paper lookup
        paper_lookup = {}
        if papers:
            for paper in papers:
                paper_lookup[paper.safe_id] = paper
        
        # Process each PDF
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Processing PDFs | count={len(pdf_files)} | dir={pdf_dir}")
        
        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"Progress | {i+1}/{len(pdf_files)} | file={pdf_path.name}")
            
            # Try to find matching paper
            paper = None
            for key in paper_lookup:
                if key in pdf_path.stem:
                    paper = paper_lookup[key]
                    break
            
            images = self.process_pdf(pdf_path, paper)
            all_images.extend(images)
        
        logger.info(f"All PDFs processed | total_images={len(all_images)}")
        return all_images
