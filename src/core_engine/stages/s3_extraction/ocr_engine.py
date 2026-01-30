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
        default="paddleocr",
        description="OCR engine: 'paddleocr' (recommended), 'easyocr', or 'tesseract'"
    )
    
    # Device settings
    use_gpu: bool = Field(
        default=True,
        description="Use GPU for OCR (PaddleOCR only)"
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
    
    # Post-processing (OCR correction)
    enable_post_processing: bool = Field(
        default=True,
        description="Enable OCR post-processing and correction"
    )
    fix_common_ocr_errors: bool = Field(
        default=True,
        description="Fix common OCR misreads (O->0, l->1, etc.)"
    )
    validate_numeric_ranges: bool = Field(
        default=True,
        description="Validate numeric values are in reasonable ranges"
    )
    
    # Image pre-enhancement for OCR
    enable_pre_enhancement: bool = Field(
        default=True,
        description="Enable image enhancement before OCR"
    )
    upscale_small_text: bool = Field(
        default=True,
        description="Upscale image regions with small text"
    )
    upscale_threshold: int = Field(
        default=15,
        ge=5,
        description="Text height threshold (px) below which to upscale"
    )
    upscale_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Upscale factor for small text regions"
    )
    denoise_before_ocr: bool = Field(
        default=True,
        description="Apply denoising before OCR"
    )
    sharpen_text: bool = Field(
        default=True,
        description="Apply sharpening filter to enhance text edges"
    )
    enhance_contrast: bool = Field(
        default=True,
        description="Apply CLAHE contrast enhancement"
    )
    
    # Cache settings
    use_cache: bool = Field(
        default=True,
        description="Use cached OCR results if available"
    )
    cache_file: Optional[Path] = Field(
        default=None,
        description="Path to OCR cache JSON file"
    )
    cache_base_dir: Optional[Path] = Field(
        default=None,
        description="Base directory for computing cache keys (relative paths)"
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
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_loaded = False
    
    def _init_engine(self) -> None:
        """Lazy initialization of OCR engine."""
        if self._initialized:
            return
        
        if self.config.engine == "paddleocr":
            try:
                # Disable oneDNN to avoid compatibility issues on Windows
                import os
                os.environ.setdefault("FLAGS_use_mkldnn", "0")
                os.environ.setdefault("PADDLE_MKL_NUM_THREADS", "1")
                
                from paddleocr import PaddleOCR
                
                # Use GPU if requested and available
                use_gpu = self.config.use_gpu
                if use_gpu:
                    try:
                        import paddle
                        if not paddle.device.is_compiled_with_cuda():
                            use_gpu = False
                            self.logger.info("CUDA not available, using CPU")
                    except Exception:
                        use_gpu = False
                        self.logger.info("Paddle GPU check failed, using CPU")
                
                # PaddleOCR 2.x initialization
                self._engine = PaddleOCR(
                    lang=self.config.languages[0] if self.config.languages else "en",
                    use_gpu=use_gpu,
                    show_log=False,
                )
                self._initialized = True
                self.logger.info(f"PaddleOCR engine initialized (use_gpu={use_gpu})")
                
            except Exception as e:
                self.logger.warning(f"PaddleOCR init failed: {e}. Falling back to EasyOCR.")
                self.config.engine = "easyocr"
                self._init_engine()  # Retry with EasyOCR
                return
                
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
    
    def _load_cache(self) -> None:
        """Load OCR cache from JSON file."""
        if self._cache_loaded:
            return
        
        if not self.config.use_cache or not self.config.cache_file:
            self._cache_loaded = True
            return
        
        cache_path = Path(self.config.cache_file)
        if cache_path.exists():
            try:
                import json
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = data.get("results", {})
                self.logger.info(
                    f"OCR cache loaded | entries={len(self._cache)} | file={cache_path}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load OCR cache: {e}")
                self._cache = {}
        else:
            self.logger.debug(f"OCR cache file not found: {cache_path}")
            self._cache = {}
        
        self._cache_loaded = True
    
    def _get_from_cache(self, image_path: Optional[Path]) -> Optional[List[dict]]:
        """
        Get OCR results from cache.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of raw OCR results if found in cache, None otherwise
        """
        if not self.config.use_cache or not image_path or self._cache is None:
            return None
        
        # Compute cache key (relative path)
        try:
            if self.config.cache_base_dir:
                key = str(image_path.relative_to(self.config.cache_base_dir))
            else:
                key = str(image_path.name)
        except ValueError:
            key = str(image_path.name)
        
        # Normalize path separators
        key = key.replace("\\", "/")
        
        # Check cache
        if key in self._cache:
            cached = self._cache[key]
            if cached.get("error"):
                self.logger.debug(f"Cache hit (error) | key={key}")
                return []
            
            # Convert cached format to raw_results format
            raw_results = []
            for text_entry in cached.get("texts", []):
                bbox = text_entry.get("bbox", {})
                raw_results.append({
                    "text": text_entry.get("text", ""),
                    "confidence": text_entry.get("confidence", 0.0),
                    "bbox": (
                        bbox.get("x_min", 0),
                        bbox.get("y_min", 0),
                        bbox.get("x_max", 0),
                        bbox.get("y_max", 0),
                    ),
                })
            
            self.logger.debug(f"Cache hit | key={key} | texts={len(raw_results)}")
            return raw_results
        
        return None
    
    def extract_text(
        self,
        image: np.ndarray,
        chart_id: str = "unknown",
        image_path: Optional[Path] = None,
    ) -> OCRResult:
        """
        Extract text from image.
        
        Args:
            image: BGR or grayscale image
            chart_id: Chart identifier for logging
            image_path: Optional path to image (for cache lookup)
        
        Returns:
            OCRResult with extracted texts
        """
        import time
        start_time = time.time()
        
        h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
        
        # Try to get from cache first
        self._load_cache()
        cached_results = self._get_from_cache(image_path)
        
        if cached_results is not None:
            # Use cached results
            raw_results = cached_results
            from_cache = True
            self.logger.debug(f"Using cached OCR | chart_id={chart_id} | texts={len(raw_results)}")
        else:
            # Run OCR engine
            from_cache = False
            self._init_engine()
            
            self.logger.debug(f"OCR started | chart_id={chart_id}")
            
            # Convert grayscale to BGR if needed (PaddleOCR expects BGR)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Apply pre-enhancement if enabled
            if self.config.enable_pre_enhancement:
                image = self._enhance_image_for_ocr(image, chart_id)
            
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
            
            # Apply post-processing if enabled
            text_value = result["text"]
            correction_applied = None
            if self.config.enable_post_processing:
                text_value, correction_applied = self._post_process_text(
                    text_value, result["confidence"]
                )
            
            # Classify role if enabled
            role = None
            if self.config.classify_roles:
                role = self._classify_role(
                    result["bbox"], w, h, text=text_value
                )
            
            ocr_text = OCRText(
                text=text_value,
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
            
            # Log corrections for debugging
            if correction_applied:
                self.logger.debug(
                    f"OCR correction | original='{result['text']}' | "
                    f"corrected='{text_value}' | type={correction_applied}"
                )
            
            texts.append(ocr_text)
        
        elapsed = (time.time() - start_time) * 1000
        
        cache_info = " (cached)" if from_cache else ""
        self.logger.info(
            f"OCR complete{cache_info} | chart_id={chart_id} | "
            f"texts={len(texts)} | time={elapsed:.1f}ms"
        )
        
        return OCRResult(
            texts=texts,
            raw_results=raw_results,
            processing_time_ms=elapsed,
        )
    
    def _extract_paddleocr(self, image: np.ndarray) -> List[dict]:
        """Extract using PaddleOCR.
        
        Compatible with PaddleOCR 2.x API using ocr() method.
        """
        results = []
        
        try:
            # PaddleOCR 2.x uses ocr() method
            # Returns: [[[box, (text, score)], ...], ...]
            output = self._engine.ocr(image)
            
            if output and output[0]:
                for line in output[0]:
                    if line is None or len(line) < 2:
                        continue
                    
                    bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_conf = line[1]    # (text, confidence)
                    
                    if len(bbox_points) < 4:
                        continue
                    
                    # Convert polygon to axis-aligned bbox
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    
                    results.append({
                        "text": text_conf[0],
                        "confidence": float(text_conf[1]),
                        "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    })
                    
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
        
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
        text: str = "",
    ) -> str:
        """
        Classify text role based on spatial position and text content.
        
        Enhanced with content-aware heuristics:
        - Numeric patterns for tick values
        - Unit patterns for axis labels  
        - Title patterns (capitalization, length)
        
        Args:
            bbox: (x_min, y_min, x_max, y_max)
            img_width: Image width
            img_height: Image height
            text: The actual text content
        
        Returns:
            TextRole string value
        """
        import re
        
        x_min, y_min, x_max, y_max = bbox
        
        # Center of text box
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Box dimensions
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        # Relative positions
        rel_x = cx / img_width
        rel_y = cy / img_height
        
        # Aspect ratio (helps detect rotated text)
        aspect_ratio = box_width / max(box_height, 1)
        
        # Relative sizes
        rel_box_width = box_width / img_width
        rel_box_height = box_height / img_height
        
        # ========== CONTENT ANALYSIS ==========
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Is numeric? (tick values)
        is_numeric = bool(re.match(r'^[-+]?[\d,.]+%?$', text_clean.replace(' ', '')))
        
        # Is short numeric with unit? (e.g., "100%", "5k", "10M")
        is_numeric_with_suffix = bool(re.match(r'^[-+]?[\d,.]+\s*[%kKmMbB]?$', text_clean))
        
        # Is axis label pattern? (contains unit words)
        axis_label_keywords = [
            'count', 'number', 'amount', 'value', 'score', 'rate', 'ratio',
            'percentage', 'percent', 'frequency', 'probability', 'density',
            'time', 'date', 'year', 'month', 'day', 'hour',
            'price', 'cost', 'revenue', 'sales', 'profit',
            'distance', 'height', 'width', 'size', 'area', 'volume',
            'temperature', 'weight', 'mass', 'speed', 'velocity',
            'accuracy', 'precision', 'recall', 'f1', 'loss', 'error',
            'epoch', 'iteration', 'step', 'batch',
            '(', ')', '[', ']',  # Units often in parentheses
        ]
        has_axis_keywords = any(kw in text_lower for kw in axis_label_keywords)
        
        # Is legend pattern? (short, possibly with color indicator)
        legend_keywords = [
            'group', 'class', 'category', 'type', 'series', 'label',
            'train', 'test', 'val', 'validation',
            'baseline', 'proposed', 'ours', 'method',
            'model', 'algorithm',
        ]
        has_legend_keywords = any(kw in text_lower for kw in legend_keywords)
        
        # Is title pattern? (longer, possibly capitalized)
        is_title_like = (
            len(text_clean) > 10 and
            (text_clean[0].isupper() or text_clean.isupper()) and
            not is_numeric
        )
        
        # ========== POSITION-BASED CLASSIFICATION ==========
        
        # Title: Top region, wide text, centered
        if rel_y < self.config.title_region_top:
            if 0.15 < rel_x < 0.85 and (is_title_like or rel_box_width > 0.3):
                return TextRole.TITLE.value
        
        # Y-axis label: Left side, often rotated (tall narrow box)
        # Rotated text has aspect_ratio < 1 (taller than wide)
        if rel_x < 0.15:
            if 0.25 < rel_y < 0.75:
                # Check if it looks like axis label (not numeric)
                if has_axis_keywords or (not is_numeric and len(text_clean) > 3):
                    return TextRole.Y_AXIS_LABEL.value
        
        # Y-tick labels: Left side, numeric
        if rel_x < 0.25 and 0.1 < rel_y < 0.9:
            if is_numeric or is_numeric_with_suffix:
                return TextRole.Y_TICK.value
        
        # X-axis label: Bottom, centered, not numeric
        if rel_y > 0.85:
            if 0.25 < rel_x < 0.75:
                if has_axis_keywords or (not is_numeric and len(text_clean) > 3):
                    return TextRole.X_AXIS_LABEL.value
        
        # X-tick labels: Bottom region, could be numeric or categorical
        if rel_y > 0.75:
            # Numeric ticks
            if is_numeric or is_numeric_with_suffix:
                return TextRole.X_TICK.value
            # Categorical ticks (short text at bottom)
            if len(text_clean) < 20 and not has_axis_keywords:
                return TextRole.X_TICK.value
        
        # Legend: Usually top-right, bottom, or right side
        # Often short categorical labels
        if rel_x > 0.65:
            if rel_y < 0.35 or rel_y > 0.65:
                if has_legend_keywords or (len(text_clean) < 25 and not is_numeric):
                    return TextRole.LEGEND.value
        
        # Data label: Inside plot area, usually numeric values on/near elements
        if 0.15 < rel_x < 0.85 and 0.15 < rel_y < 0.85:
            if is_numeric or is_numeric_with_suffix:
                return TextRole.DATA_LABEL.value
            # Short text near data could be label
            if len(text_clean) < 15:
                return TextRole.DATA_LABEL.value
        
        # Subtitle: Just below title area
        if 0.12 < rel_y < 0.25 and 0.2 < rel_x < 0.8:
            if not is_numeric:
                return TextRole.SUBTITLE.value
        
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
    
    def _post_process_text(
        self,
        text: str,
        confidence: float,
    ) -> Tuple[str, Optional[str]]:
        """
        Post-process OCR text to fix common errors.
        
        Args:
            text: Raw OCR text
            confidence: OCR confidence score
        
        Returns:
            Tuple of (corrected_text, correction_type or None)
        """
        import re
        
        original = text
        correction_type = None
        
        # ========== COMMON OCR CHARACTER ERRORS ==========
        # These are frequent misreads in chart contexts
        
        if self.config.fix_common_ocr_errors:
            # Only apply aggressive fixes to low-confidence results
            # or when text looks like it should be numeric
            looks_numeric = bool(re.search(r'\d', text))
            
            if looks_numeric or confidence < 0.8:
                # Common letter-digit confusions
                ocr_corrections = {
                    # Letter O -> digit 0 (in numeric context)
                    'O': '0',
                    'o': '0',
                    # Letter l/I -> digit 1 (in numeric context)
                    'l': '1',
                    'I': '1',
                    # Letter S -> digit 5
                    'S': '5',
                    's': '5',
                    # Letter B -> digit 8
                    'B': '8',
                    # Letter Z -> digit 2
                    'Z': '2',
                    # Letter G -> digit 6
                    'G': '6',
                    # Letter q -> digit 9
                    'q': '9',
                }
                
                # Apply corrections only if text looks like a number
                # (has digits mixed with potential letter errors)
                if re.match(r'^[\d\s,.\-+%OolISsBZGq]+$', text):
                    corrected = text
                    for wrong, right in ocr_corrections.items():
                        if wrong in corrected:
                            corrected = corrected.replace(wrong, right)
                    
                    if corrected != text:
                        text = corrected
                        correction_type = "char_substitution"
        
        # ========== NUMERIC FORMAT NORMALIZATION ==========
        
        # Fix common decimal point issues
        # "1,5" -> "1.5" (European decimal)
        if re.match(r'^\d+,\d{1,2}$', text):
            text = text.replace(',', '.')
            correction_type = correction_type or "decimal_format"
        
        # Fix thousand separators that look like decimals
        # "1.000" -> "1000" (if followed by 3 digits)
        if re.match(r'^\d{1,3}\.\d{3}$', text) and ',' not in original:
            text = text.replace('.', '')
            correction_type = correction_type or "thousand_separator"
        
        # ========== COMMON UNIT/SUFFIX FIXES ==========
        
        # Fix percentage signs that got split or misread
        # "100 %" -> "100%"
        text = re.sub(r'(\d)\s+%', r'\1%', text)
        
        # Fix "K" "M" "B" suffixes (common in charts)
        # "10 K" -> "10K"
        text = re.sub(r'(\d)\s+([KkMmBb])(?:\s|$)', r'\1\2', text)
        
        # ========== WHITESPACE CLEANUP ==========
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # ========== VALIDATE NUMERIC RANGES ==========
        
        if self.config.validate_numeric_ranges and correction_type:
            # After correction, verify the number is reasonable
            try:
                # Try to parse as number
                clean = text.replace(',', '').replace('%', '').replace(' ', '')
                clean = re.sub(r'[KkMmBb]$', '', clean)
                value = float(clean)
                
                # Flag suspicious corrections (extremely large/small)
                if abs(value) > 1e15 or (value != 0 and abs(value) < 1e-15):
                    # Revert to original - correction likely wrong
                    self.logger.debug(
                        f"Reverting OCR correction (out of range) | "
                        f"corrected='{text}' | original='{original}'"
                    )
                    return original, None
                    
            except ValueError:
                pass  # Not a pure number, that's OK
        
        if text != original and correction_type is None:
            correction_type = "cleanup"
        
        return text, correction_type if text != original else None
    
    def _enhance_image_for_ocr(
        self,
        image: np.ndarray,
        chart_id: str = "unknown",
    ) -> np.ndarray:
        """
        Enhance image quality before OCR.
        
        Applies multiple enhancement techniques:
        1. Contrast enhancement (CLAHE)
        2. Denoising (bilateral filter)
        3. Sharpening (unsharp mask)
        4. Upscaling for small text
        
        Args:
            image: Input BGR image
            chart_id: Chart identifier for logging
        
        Returns:
            Enhanced BGR image
        """
        enhanced = image.copy()
        h, w = enhanced.shape[:2]
        
        enhancements_applied = []
        
        # 1. Denoise (preserves edges better than Gaussian blur)
        if self.config.denoise_before_ocr:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            enhancements_applied.append("denoise")
        
        # 2. Contrast enhancement using CLAHE on L channel
        if self.config.enhance_contrast:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            enhancements_applied.append("clahe")
        
        # 3. Sharpening using unsharp mask
        if self.config.sharpen_text:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            enhancements_applied.append("sharpen")
        
        # 4. Upscale small images (helps with small text)
        if self.config.upscale_small_text:
            # If image is small overall, upscale it
            if h < 200 or w < 300:
                scale = self.config.upscale_factor
                new_w = int(w * scale)
                new_h = int(h * scale)
                enhanced = cv2.resize(
                    enhanced, (new_w, new_h),
                    interpolation=cv2.INTER_CUBIC
                )
                enhancements_applied.append(f"upscale_{scale}x")
        
        if enhancements_applied:
            self.logger.debug(
                f"OCR image enhancement | chart_id={chart_id} | "
                f"applied={','.join(enhancements_applied)}"
            )
        
        return enhanced
    
    def enhance_text_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        target_height: int = 32,
    ) -> np.ndarray:
        """
        Enhance a specific text region for better OCR accuracy.
        
        This method is useful for re-processing low-confidence text
        regions with more aggressive enhancement.
        
        Args:
            image: Full image
            bbox: Text bounding box
            target_height: Target height for text region
        
        Returns:
            Enhanced cropped region
        """
        # Crop with padding
        pad = 5
        x1 = max(0, bbox.x_min - pad)
        y1 = max(0, bbox.y_min - pad)
        x2 = min(image.shape[1], bbox.x_max + pad)
        y2 = min(image.shape[0], bbox.y_max + pad)
        
        region = image[y1:y2, x1:x2].copy()
        
        if region.size == 0:
            return region
        
        # Calculate scale to reach target height
        current_height = region.shape[0]
        if current_height < target_height:
            scale = target_height / current_height
            new_w = int(region.shape[1] * scale)
            new_h = target_height
            region = cv2.resize(
                region, (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )
        
        # Enhanced preprocessing for text
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Binarization using Otsu
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Invert if text is white on dark background
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Convert back to BGR for OCR engine compatibility
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return enhanced
