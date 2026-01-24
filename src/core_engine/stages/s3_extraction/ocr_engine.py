"""
OCR Engine Module

Wrapper for text extraction using PaddleOCR with role classification.

Key features:
- Text detection and recognition
- Spatial role classification (title, axis labels, legend, values)
- Confidence filtering
- Bounding box extraction

Reference: docs/instruction_p2_research.md
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ...schemas.common import BoundingBox
from ...schemas.enums import TextRole
from ...schemas.stage_outputs import OCRText

logger = logging.getLogger(__name__)


class OCRConfig(BaseModel):
    """Configuration for OCR engine."""
    
    # Engine selection
    engine: str = Field(
        default="easyocr",
        description="OCR engine: 'easyocr', 'paddleocr', or 'tesseract'"
    )
    
    # Language settings
    languages: List[str] = Field(
        default=["en"],
        description="Languages to detect"
    )
    
    # Detection settings
    use_angle_cls: bool = Field(
        default=False,
        description="Use angle classification for rotated text"
    )
    det_db_thresh: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Detection threshold"
    )
    
    # Recognition settings
    rec_batch_num: int = Field(default=6, ge=1, description="Recognition batch size")
    
    # Filtering
    min_confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Minimum confidence to keep"
    )
    min_text_length: int = Field(
        default=1,
        ge=1,
        description="Minimum text length to keep"
    )
    
    # Role classification
    classify_roles: bool = Field(
        default=True,
        description="Classify text roles based on position"
    )
    title_region_top: float = Field(
        default=0.15,
        gt=0,
        le=1,
        description="Top region proportion for title detection"
    )


@dataclass
class OCRResult:
    """Result of OCR operation."""
    
    texts: List[OCRText]
    raw_results: List[dict]  # Raw engine output
    processing_time_ms: float


class OCREngine:
    """
    OCR engine wrapper for text extraction from charts.
    
    Uses PaddleOCR (default) for high accuracy on diverse fonts
    and languages common in academic papers.
    
    Example:
        config = OCRConfig(engine="paddleocr", languages=["en"])
        ocr = OCREngine(config)
        result = ocr.extract_text(image)
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR engine.
        
        Args:
            config: OCR configuration (uses defaults if None)
        """
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._engine = None
        self._initialized = False
    
    def _init_engine(self) -> None:
        """Lazy initialization of OCR engine."""
        if self._initialized:
            return
        
        if self.config.engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                
                self._engine = PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.languages[0] if self.config.languages else "en",
                )
                self._initialized = True
                self.logger.info("PaddleOCR engine initialized")
                
            except ImportError:
                self.logger.error("PaddleOCR not installed. Run: pip install paddleocr")
                raise
                
        elif self.config.engine == "tesseract":
            try:
                import pytesseract
                self._engine = pytesseract
                self._initialized = True
                self.logger.info("Tesseract engine initialized")
                
            except ImportError:
                self.logger.error("pytesseract not installed. Run: pip install pytesseract")
                raise
                
        elif self.config.engine == "easyocr":
            try:
                import easyocr
                # EasyOCR Reader initialization
                self._engine = easyocr.Reader(
                    self.config.languages if self.config.languages else ["en"],
                    gpu=False,  # CPU mode for compatibility
                )
                self._initialized = True
                self.logger.info("EasyOCR engine initialized")
                
            except ImportError:
                self.logger.error("easyocr not installed. Run: pip install easyocr")
                raise
        else:
            raise ValueError(f"Unknown OCR engine: {self.config.engine}")
    
    def extract_text(
        self,
        image: np.ndarray,
        chart_id: str = "unknown",
    ) -> OCRResult:
        """
        Extract text from image.
        
        Args:
            image: BGR or grayscale image
            chart_id: Chart identifier for logging
        
        Returns:
            OCRResult with extracted texts
        """
        import time
        start_time = time.time()
        
        self._init_engine()
        
        self.logger.debug(f"OCR started | chart_id={chart_id}")
        
        # Convert grayscale to BGR if needed (PaddleOCR expects BGR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        
        if self.config.engine == "paddleocr":
            raw_results = self._extract_paddleocr(image)
        elif self.config.engine == "easyocr":
            raw_results = self._extract_easyocr(image)
        else:
            raw_results = self._extract_tesseract(image)
        
        # Convert to OCRText objects
        texts = []
        for result in raw_results:
            # Filter by confidence
            if result["confidence"] < self.config.min_confidence:
                continue
            
            # Filter by length
            if len(result["text"].strip()) < self.config.min_text_length:
                continue
            
            # Classify role if enabled
            role = None
            if self.config.classify_roles:
                role = self._classify_role(result["bbox"], w, h)
            
            ocr_text = OCRText(
                text=result["text"],
                bbox=BoundingBox(
                    x_min=int(result["bbox"][0]),
                    y_min=int(result["bbox"][1]),
                    x_max=int(result["bbox"][2]),
                    y_max=int(result["bbox"][3]),
                    confidence=result["confidence"],
                ),
                confidence=result["confidence"],
                role=role,
            )
            texts.append(ocr_text)
        
        elapsed = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"OCR complete | chart_id={chart_id} | "
            f"texts={len(texts)} | time={elapsed:.1f}ms"
        )
        
        return OCRResult(
            texts=texts,
            raw_results=raw_results,
            processing_time_ms=elapsed,
        )
    
    def _extract_paddleocr(self, image: np.ndarray) -> List[dict]:
        """Extract using PaddleOCR."""
        results = []
        
        # PaddleOCR returns: [[bbox, (text, confidence)], ...]
        # Note: New PaddleOCR API doesn't support cls parameter in ocr()
        output = self._engine.ocr(image)
        
        if output is None or len(output) == 0:
            return results
        
        # Handle different output formats
        if output[0] is None:
            return results
        
        for line in output[0]:
            if line is None or len(line) < 2:
                continue
            
            bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_conf = line[1]    # (text, confidence)
            
            if len(bbox_points) < 4:
                continue
            
            # Convert quadrilateral to axis-aligned bbox
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            
            results.append({
                "text": text_conf[0],
                "confidence": text_conf[1],
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            })
        
        return results
    
    def _extract_tesseract(self, image: np.ndarray) -> List[dict]:
        """Extract using Tesseract."""
        results = []
        
        # Get detailed data
        data = self._engine.image_to_data(
            image, output_type=self._engine.Output.DICT
        )
        
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i]) / 100.0  # Tesseract uses 0-100
            
            if not text or conf < 0:
                continue
            
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            
            results.append({
                "text": text,
                "confidence": conf,
                "bbox": (x, y, x + w, y + h),
            })
        
        return results
    
    def _extract_easyocr(self, image: np.ndarray) -> List[dict]:
        """Extract using EasyOCR."""
        results = []
        
        # EasyOCR returns: [[bbox, text, confidence], ...]
        # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (quadrilateral)
        output = self._engine.readtext(image)
        
        if output is None or len(output) == 0:
            return results
        
        for item in output:
            if item is None or len(item) < 3:
                continue
            
            bbox_points = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = item[1]         # text string
            confidence = item[2]   # confidence float
            
            if len(bbox_points) < 4:
                continue
            
            # Convert quadrilateral to axis-aligned bbox
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            
            results.append({
                "text": text,
                "confidence": confidence,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            })
        
        return results
    
    def _classify_role(
        self,
        bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
    ) -> str:
        """
        Classify text role based on spatial position.
        
        Args:
            bbox: (x_min, y_min, x_max, y_max)
            img_width: Image width
            img_height: Image height
        
        Returns:
            TextRole string value
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Center of text box
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Relative positions
        rel_x = cx / img_width
        rel_y = cy / img_height
        
        # Height of text box relative to image
        rel_height = (y_max - y_min) / img_height
        
        # Title: Top region, centered horizontally
        if rel_y < self.config.title_region_top and 0.2 < rel_x < 0.8:
            return TextRole.TITLE.value
        
        # Y-axis label: Left side, vertically centered
        if rel_x < 0.15 and 0.3 < rel_y < 0.7:
            return TextRole.Y_AXIS_LABEL.value
        
        # Y-tick labels: Left side
        if rel_x < 0.2 and 0.1 < rel_y < 0.9:
            return TextRole.Y_TICK.value
        
        # X-axis label: Bottom, centered
        if rel_y > 0.85 and 0.3 < rel_x < 0.7:
            return TextRole.X_AXIS_LABEL.value
        
        # X-tick labels: Bottom region
        if rel_y > 0.8:
            return TextRole.X_TICK.value
        
        # Legend: Usually top-right or bottom
        if rel_x > 0.7 and rel_y < 0.3:
            return TextRole.LEGEND.value
        
        # Data label: Inside plot area
        if 0.15 < rel_x < 0.85 and 0.15 < rel_y < 0.85:
            return TextRole.DATA_LABEL.value
        
        return TextRole.UNKNOWN.value
    
    def extract_axis_values(
        self,
        texts: List[OCRText],
        axis: str = "y",
    ) -> List[Tuple[float, float]]:
        """
        Extract numeric values from axis tick labels.
        
        Args:
            texts: List of OCRText objects
            axis: "x" or "y"
        
        Returns:
            List of (position, value) tuples
        """
        import re
        
        role = TextRole.Y_TICK.value if axis == "y" else TextRole.X_TICK.value
        values = []
        
        for text in texts:
            if text.role != role:
                continue
            
            # Try to parse numeric value
            try:
                # Remove common formatting
                clean = text.text.replace(",", "").replace(" ", "")
                
                # Handle percentages
                if "%" in clean:
                    clean = clean.replace("%", "")
                    value = float(clean)
                else:
                    value = float(clean)
                
                # Position: center of bbox
                if axis == "y":
                    pos = (text.bbox.y_min + text.bbox.y_max) / 2
                else:
                    pos = (text.bbox.x_min + text.bbox.x_max) / 2
                
                values.append((pos, value))
                
            except ValueError:
                # Not a numeric value
                continue
        
        return values
