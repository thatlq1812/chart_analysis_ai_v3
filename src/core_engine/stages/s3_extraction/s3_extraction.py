"""
Stage 3: Structural Analysis (Extraction)

Main orchestrator for the Geo-SLM hybrid extraction pipeline.

This stage implements:
1. Negative image preprocessing for structural enhancement
2. Topology-preserving skeletonization (Lee algorithm)
3. RDP vectorization for piecewise linear representation
4. OCR text extraction with role classification
5. Geometric mapping from pixels to data values
6. Chart type classification from structural features

Architecture follows the "bottom-up" approach described in
instruction_p2_research.md: treat charts as collections of
geometric entities rather than pixel patterns.

Reference: docs/architecture/PIPELINE_FLOW.md#stage-3-extraction
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ...exceptions import StageProcessingError
from ...schemas.common import BoundingBox, Color, Point
from ...schemas.enums import ChartType, ElementType, TextRole
from ...schemas.extraction import (
    PointFloat,
    Polyline,
    SkeletonGraph,
    VectorizedChart,
)
from ...schemas.stage_outputs import (
    AxisInfo,
    ChartElement,
    ExtractionConfidence,
    OCRText,
    RawMetadata,
    Stage2Output,
    Stage3Output,
)
from ..base import BaseStage
from .classifier import ChartClassifier, ClassifierConfig
from .element_detector import ElementDetector, ElementDetectorConfig
from .resnet_classifier import ResNet18Classifier, create_resnet_classifier
from .geometric_mapper import GeometricMapper, MapperConfig
from .ml_classifier import MLChartClassifier
from .ocr_engine import OCREngine, OCRConfig
from .preprocessor import ImagePreprocessor, PreprocessConfig
from .skeletonizer import Skeletonizer, SkeletonConfig
from .vectorizer import Vectorizer, VectorizeConfig

logger = logging.getLogger(__name__)


class ExtractionConfig(BaseModel):
    """Configuration for Stage 3: Extraction."""
    
    # OCR engine selection (shorthand - passed to OCRConfig)
    ocr_engine: str = Field(
        default="easyocr",
        description="OCR engine: 'easyocr', 'paddleocr', or 'tesseract'"
    )
    
    # Submodule configs
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    skeleton: SkeletonConfig = Field(default_factory=SkeletonConfig)
    vectorize: VectorizeConfig = Field(default_factory=VectorizeConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
    elements: ElementDetectorConfig = Field(default_factory=ElementDetectorConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    
    # [FIX] Shortcut for element detector color segmentation
    # This is forwarded to elements.use_color_segmentation during init
    use_color_segmentation: bool = Field(
        default=True,
        description="Use color-based detection instead of binary contours"
    )
    
    # Processing options
    enable_vectorization: bool = Field(
        default=True,
        description="Enable skeleton-based vectorization for lines"
    )
    enable_element_detection: bool = Field(
        default=True,
        description="Enable discrete element detection (bars, markers)"
    )
    enable_ocr: bool = Field(default=True, description="Enable OCR text extraction")
    enable_classification: bool = Field(
        default=True,
        description="Enable automatic chart type classification"
    )
    use_ml_classifier: bool = Field(
        default=True,
        description="Use ML-based classifier instead of rule-based"
    )
    use_resnet_classifier: bool = Field(
        default=True,
        description="Use ResNet-18 classifier (94.14% accuracy) instead of Random Forest"
    )
    resnet_model_path: Optional[Path] = Field(
        default=None,
        description="Path to ResNet-18 model weights (defaults to models/weights/resnet18_chart_classifier_v2_best.pt)"
    )
    
    # [FIX] Post-negative cleaning options
    use_cleaned_for_skeleton: bool = Field(
        default=True,
        description="Use cleaned binary (grid/noise removed) for skeletonization"
    )
    mask_text_before_skeleton: bool = Field(
        default=True,
        description="Run OCR first and mask text regions before skeletonization"
    )
    
    # Output options
    save_debug_images: bool = Field(
        default=False,
        description="Save intermediate images for debugging"
    )
    debug_output_dir: Optional[Path] = Field(
        default=None,
        description="Directory for debug images"
    )


class Stage3Extraction(BaseStage):
    """
    Stage 3 Orchestrator: Structural Analysis (Extraction)
    
    Implements the Geo-SLM hybrid approach for extracting
    structured data from chart images.
    
    Pipeline:
        1. Preprocess (negative transform, denoising)
        2. Skeletonize (topology-preserving thinning)
        3. Vectorize (RDP simplification)
        4. Detect elements (bars, markers, slices)
        5. Extract text (OCR with role classification)
        6. Map coordinates (pixel to value calibration)
        7. Classify chart type
    
    Example:
        config = ExtractionConfig()
        stage = Stage3Extraction(config)
        result = stage.process(stage2_output)
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize Stage 3: Extraction.
        
        Args:
            config: Extraction configuration (uses defaults if None)
        """
        config = config or ExtractionConfig()
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize submodules
        self.preprocessor = ImagePreprocessor(config.preprocess)
        self.skeletonizer = Skeletonizer(config.skeleton)
        self.vectorizer = Vectorizer(config.vectorize)
        
        # Build OCR config with engine selection
        ocr_config = config.ocr.model_copy()
        ocr_config.engine = config.ocr_engine
        self.ocr_engine = OCREngine(ocr_config)
        
        self.mapper = GeometricMapper(config.mapper)
        
        # [FIX] Forward use_color_segmentation shortcut to element detector
        element_config = config.elements.model_copy(
            update={"use_color_segmentation": config.use_color_segmentation}
        )
        self.element_detector = ElementDetector(element_config)
        
        self.classifier = ChartClassifier(config.classifier)
        
        # ResNet-18 classifier (preferred, 94.14% accuracy)
        self.resnet_classifier = None
        if config.use_resnet_classifier:
            try:
                resnet_path = config.resnet_model_path
                if resnet_path is None:
                    resnet_path = Path(__file__).parent.parent.parent.parent.parent / "models/weights/resnet18_chart_classifier_v2_best.pt"
                self.resnet_classifier = create_resnet_classifier(resnet_path)
                self.logger.info(f"ResNet-18 classifier loaded | accuracy=94.14% | path={resnet_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load ResNet-18 classifier: {e}. Will try ML classifier.")
        
        # ML classifier fallback (Random Forest)
        self.ml_classifier = None
        if config.use_ml_classifier and self.resnet_classifier is None:
            try:
                self.ml_classifier = MLChartClassifier()
                self.logger.info("ML classifier (Random Forest) loaded as fallback")
            except Exception as e:
                self.logger.warning(f"Failed to load ML classifier: {e}. Using rule-based.")
        
        self.logger.info("Stage3Extraction initialized")
    
    def process(self, input_data: Stage2Output) -> Stage3Output:
        """
        Process Stage 2 output and extract chart data.
        
        Args:
            input_data: Stage2Output with detected charts
        
        Returns:
            Stage3Output with raw metadata for each chart
        
        Raises:
            StageProcessingError: If extraction fails critically
        """
        if not self.validate_input(input_data):
            raise StageProcessingError(
                "Stage3Extraction",
                f"Invalid input: expected Stage2Output, got {type(input_data)}"
            )
        
        session = input_data.session
        self.logger.info(
            f"Extraction started | session={session.session_id} | "
            f"charts={len(input_data.charts)}"
        )
        
        metadata_list = []
        
        for chart in input_data.charts:
            try:
                metadata = self._process_single_chart(chart)
                metadata_list.append(metadata)
                
            except Exception as e:
                self.logger.error(
                    f"Chart extraction failed | chart_id={chart.chart_id} | "
                    f"error={str(e)}"
                )
                # Create fallback metadata with minimal info
                fallback = RawMetadata(
                    chart_id=chart.chart_id,
                    chart_type=ChartType.UNKNOWN,
                    texts=[],
                    elements=[],
                    axis_info=None,
                )
                metadata_list.append(fallback)
        
        self.logger.info(
            f"Extraction complete | session={session.session_id} | "
            f"processed={len(metadata_list)}"
        )
        
        return Stage3Output(
            session=session,
            metadata=metadata_list,
        )
    
    def validate_input(self, input_data) -> bool:
        """Validate input is Stage2Output."""
        return isinstance(input_data, Stage2Output)
    
    def process_image(
        self,
        image_bgr: np.ndarray,
        chart_id: str = "test",
        image_path: Optional[Path] = None,
    ) -> RawMetadata:
        """
        Process a raw image directly (for testing/standalone use).
        
        [FIX] Updated to use cleaned image and OCR-first flow for skeletonization.
        
        Args:
            image_bgr: BGR image array
            chart_id: Identifier for logging
            image_path: Optional path to image (for OCR cache lookup)
        
        Returns:
            RawMetadata with extracted information
        """
        h, w = image_bgr.shape[:2]
        
        # Classify using best available classifier
        chart_type = ChartType.UNKNOWN
        if self.config.enable_classification:
            chart_type = self._classify_chart(image_bgr, chart_id)
        
        # Preprocessing
        preprocess_result = self.preprocessor.process(image_bgr, chart_id)
        
        # [FIX] OCR first (for text masking)
        texts = []
        if self.config.enable_ocr:
            ocr_result = self.ocr_engine.extract_text(
                image_bgr, chart_id, image_path=image_path
            )
            texts = ocr_result.texts
        
        # [FIX] Prepare skeleton input with cleaning
        if self.config.use_cleaned_for_skeleton:
            skeleton_input = preprocess_result.cleaned_image.copy()
        else:
            skeleton_input = preprocess_result.binary_image.copy()
        
        # [FIX] Apply text masking if enabled
        if self.config.mask_text_before_skeleton and texts:
            text_boxes = self.preprocessor.extract_text_boxes_for_masking(texts)
            if text_boxes:
                skeleton_input, _ = self.preprocessor.clean_for_skeleton(
                    skeleton_input,
                    text_boxes=text_boxes,
                    chart_id=chart_id,
                )
        
        # Vectorization (using cleaned input)
        polylines = []
        if self.config.enable_vectorization:
            skeleton_result = self.skeletonizer.process(
                skeleton_input,  # [FIX] Use cleaned image
                chart_id=chart_id,
            )
            paths = self.skeletonizer.trace_paths(
                skeleton_result.skeleton,
                skeleton_result.keypoints,
            )
            vector_result = self.vectorizer.process(
                paths,
                stroke_width_map=skeleton_result.stroke_width_map,
                grayscale_image=preprocess_result.grayscale_image,
                chart_id=chart_id,
            )
            polylines = vector_result.polylines
        
        # Element detection (uses original binary)
        bars = []
        markers = []
        slices = []
        elements = []
        if self.config.enable_element_detection:
            element_result = self.element_detector.detect(
                preprocess_result.binary_image,
                color_image=image_bgr,
                chart_id=chart_id,
            )
            bars = element_result.bars
            markers = element_result.markers
            slices = element_result.slices
            elements = self._convert_elements(bars, markers, slices, w, h)
        
        # Calculate confidence
        classification_conf = 0.0
        if self.resnet_classifier is not None and chart_type != ChartType.UNKNOWN:
            try:
                _, classification_conf = self.resnet_classifier.predict_with_confidence(image_bgr)
            except Exception:
                classification_conf = 0.9
        
        ocr_conf = sum(t.confidence for t in texts) / len(texts) if texts else 0.0
        element_conf = min(1.0, len(elements) * 0.1) if elements else 0.0
        
        confidence = ExtractionConfidence(
            classification_confidence=classification_conf,
            ocr_mean_confidence=ocr_conf,
            axis_calibration_confidence=0.0,
            element_detection_confidence=element_conf,
        )
        
        return RawMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            texts=texts,
            elements=elements,
            axis_info=None,
            confidence=confidence,
        )

    def _process_single_chart(self, chart) -> RawMetadata:
        """
        Process a single chart through extraction pipeline.
        
        [FIX] Updated flow to address "Post-Negative Issue":
        1. Preprocess (negative + threshold + grid/noise cleaning)
        2. OCR (extract text first for masking)
        3. Text masking (optional, mask OCR regions in binary)
        4. Skeletonize (on cleaned image)
        5. Vectorize
        6. Element detection
        7. Axis calibration
        8. Classification
        
        Args:
            chart: DetectedChart from Stage 2
        
        Returns:
            RawMetadata with extracted information
        """
        chart_id = chart.chart_id
        self.logger.debug(f"Processing chart | chart_id={chart_id}")
        
        # Load image
        image_bgr = cv2.imread(str(chart.cropped_path))
        if image_bgr is None:
            raise StageProcessingError(
                "Stage3Extraction",
                f"Failed to load image: {chart.cropped_path}"
            )
        
        h, w = image_bgr.shape[:2]
        
        # Step 1: Preprocessing (negative image + thresholding + initial cleaning)
        preprocess_result = self.preprocessor.process(image_bgr, chart_id)
        
        # Save debug images if configured
        if self.config.save_debug_images and self.config.debug_output_dir:
            self._save_debug_images(chart_id, preprocess_result)
        
        # [FIX] Step 2: OCR text extraction (run BEFORE skeletonization for text masking)
        texts = []
        if self.config.enable_ocr:
            ocr_result = self.ocr_engine.extract_text(image_bgr, chart_id)
            texts = ocr_result.texts
        
        # [FIX] Step 3: Prepare binary image for skeletonization
        # Choose between original binary or cleaned version
        if self.config.use_cleaned_for_skeleton:
            skeleton_input = preprocess_result.cleaned_image.copy()
        else:
            skeleton_input = preprocess_result.binary_image.copy()
        
        # [FIX] Apply text masking if enabled and we have OCR results
        if self.config.mask_text_before_skeleton and texts:
            text_boxes = self.preprocessor.extract_text_boxes_for_masking(texts)
            if text_boxes:
                skeleton_input, mask_stats = self.preprocessor.clean_for_skeleton(
                    skeleton_input,
                    text_boxes=text_boxes,
                    chart_id=chart_id,
                )
                self.logger.debug(
                    f"Text masking applied | chart_id={chart_id} | "
                    f"regions_masked={mask_stats['text_regions_masked']}"
                )
        
        # Step 4: Skeleton extraction (now using cleaned input)
        skeleton_result = None
        polylines = []
        keypoints = []
        
        if self.config.enable_vectorization:
            skeleton_result = self.skeletonizer.process(
                skeleton_input,  # [FIX] Use cleaned image instead of raw binary
                chart_id=chart_id,
            )
            
            # Step 5: Trace paths from skeleton
            paths = self.skeletonizer.trace_paths(
                skeleton_result.skeleton,
                skeleton_result.keypoints,
            )
            
            # Step 6: Vectorize paths with RDP
            vector_result = self.vectorizer.process(
                paths,
                stroke_width_map=skeleton_result.stroke_width_map,
                grayscale_image=preprocess_result.grayscale_image,
                chart_id=chart_id,
            )
            
            polylines = vector_result.polylines
            keypoints = skeleton_result.keypoints
        
        # Step 7: Early chart type classification (before element detection)
        # Use unified _classify_chart method which prioritizes ResNet > ML > rule-based
        chart_type = ChartType.UNKNOWN
        if self.config.enable_classification:
            chart_type = self._classify_chart(image_bgr, chart_id)
        
        # Step 8: Detect discrete elements (bars, markers)
        # Note: Uses original binary_image (not cleaned) to preserve element shapes
        # [FIX] Pass chart_type to enable routing (skip bar detection for line/scatter/pie)
        elements = []
        bars = []
        markers = []
        slices = []
        
        if self.config.enable_element_detection:
            element_result = self.element_detector.detect(
                preprocess_result.binary_image,
                color_image=image_bgr,
                chart_id=chart_id,
                chart_type=chart_type.value if chart_type != ChartType.UNKNOWN else None,
            )
            
            bars = element_result.bars
            markers = element_result.markers
            slices = element_result.slices
            
            # Convert to ChartElement format
            elements = self._convert_elements(bars, markers, slices, w, h)
        
        # [FIX] Note: OCR already extracted in Step 2 (before skeletonization)
        # No duplicate extraction needed
        
        # Step 9: Axis calibration from OCR
        axis_info = self._calibrate_axes(texts, w, h)
        
        # Step 10: Final chart type classification (if still unknown)
        # Already classified in Step 7 using _classify_chart, but fallback to rule-based if needed
        if chart_type == ChartType.UNKNOWN and self.config.enable_classification:
            # Rule-based fallback using extracted features
            class_result = self.classifier.classify(
                bars=bars,
                polylines=polylines,
                markers=markers,
                slices=slices,
                texts=texts,
                image_shape=(h, w),
                chart_id=chart_id,
            )
            chart_type = class_result.chart_type
        
        # Build RawMetadata with confidence scores
        # Calculate confidence components
        warnings = []
        
        # Classification confidence - get from ResNet or ML classifier
        classification_conf = 0.0
        if self.resnet_classifier is not None and chart_type != ChartType.UNKNOWN:
            try:
                _, classification_conf = self.resnet_classifier.predict_with_confidence(image_bgr)
            except Exception:
                classification_conf = 0.9  # ResNet default high confidence
        elif self.ml_classifier is not None and chart_type != ChartType.UNKNOWN:
            try:
                ml_result = self.ml_classifier.classify(image_bgr, chart_id=chart_id)
                classification_conf = ml_result.confidence
            except Exception:
                classification_conf = 0.5  # Default moderate confidence
        elif chart_type != ChartType.UNKNOWN:
            classification_conf = 0.7  # Rule-based has lower default confidence
        
        if classification_conf < 0.7:
            warnings.append(f"Low classification confidence: {classification_conf:.2f}")
        
        # OCR mean confidence
        ocr_conf = 0.0
        if texts:
            ocr_conf = sum(t.confidence for t in texts) / len(texts)
            if ocr_conf < 0.7:
                warnings.append(f"Low OCR confidence: {ocr_conf:.2f}")
        else:
            warnings.append("No text detected by OCR")
        
        # Axis calibration confidence (average of both axes)
        axis_conf = 0.0
        if axis_info:
            confs = []
            if axis_info.y_calibration_confidence > 0:
                confs.append(axis_info.y_calibration_confidence)
            if axis_info.x_calibration_confidence > 0:
                confs.append(axis_info.x_calibration_confidence)
            if confs:
                axis_conf = sum(confs) / len(confs)
            
            # Warn if outliers were removed
            total_outliers = axis_info.x_outliers_removed + axis_info.y_outliers_removed
            if total_outliers > 0:
                warnings.append(f"Removed {total_outliers} outlier tick labels during calibration")
        else:
            warnings.append("Axis calibration failed or insufficient data")
        
        # Element detection confidence (based on count and consistency)
        element_conf = 0.0
        if elements:
            # Heuristic: more elements = more confidence (up to a point)
            element_conf = min(1.0, len(elements) / 10.0) * 0.8 + 0.2
            
            # Reduce confidence if element count seems inconsistent with chart type
            if chart_type == ChartType.PIE and len([e for e in elements if e.element_type == ElementType.SLICE.value]) == 0:
                element_conf *= 0.5
                warnings.append("Pie chart detected but no slices found")
            elif chart_type == ChartType.BAR and len([e for e in elements if e.element_type == ElementType.BAR.value]) == 0:
                element_conf *= 0.5
                warnings.append("Bar chart detected but no bars found")
        
        # Compute overall confidence
        confidence = ExtractionConfidence.compute_overall(
            classification=classification_conf,
            ocr=ocr_conf,
            axis=axis_conf,
            elements=element_conf,
        )
        
        self.logger.debug(
            f"Extraction confidence | chart_id={chart_id} | "
            f"overall={confidence.overall_confidence:.2f} | "
            f"class={classification_conf:.2f} | ocr={ocr_conf:.2f} | "
            f"axis={axis_conf:.2f} | elem={element_conf:.2f}"
        )
        
        return RawMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            texts=texts,
            elements=elements,
            axis_info=axis_info,
            confidence=confidence,
            warnings=warnings,
        )
    
    def _convert_elements(
        self,
        bars,
        markers,
        slices,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> List[ChartElement]:
        """Convert detected elements to ChartElement format."""
        elements = []
        
        # Convert bars
        for bar in bars:
            elements.append(ChartElement(
                element_type=ElementType.BAR.value,
                bbox=BoundingBox.from_coords(
                    x_min=int(bar.x_min),
                    y_min=int(bar.y_min),
                    x_max=int(bar.x_max),
                    y_max=int(bar.y_max),
                    image_width=image_width,
                    image_height=image_height,
                ),
                center=Point(
                    x=int(bar.center.x),
                    y=int(bar.center.y),
                ),
                color=bar.color,
                area_pixels=int(bar.area),
            ))
        
        # Convert markers
        for marker in markers:
            size = int(marker.size)
            elements.append(ChartElement(
                element_type=ElementType.POINT.value,
                bbox=BoundingBox.from_coords(
                    x_min=int(marker.center.x - size // 2),
                    y_min=int(marker.center.y - size // 2),
                    x_max=int(marker.center.x + size // 2),
                    y_max=int(marker.center.y + size // 2),
                    image_width=image_width,
                    image_height=image_height,
                ),
                center=Point(
                    x=int(marker.center.x),
                    y=int(marker.center.y),
                ),
                color=marker.color,
            ))
        
        # Convert slices
        for slice_elem in slices:
            # Approximate bounding box for slice
            r = int(slice_elem.radius_outer)
            cx, cy = int(slice_elem.center.x), int(slice_elem.center.y)
            elements.append(ChartElement(
                element_type=ElementType.SLICE.value,
                bbox=BoundingBox.from_coords(
                    x_min=cx - r,
                    y_min=cy - r,
                    x_max=cx + r,
                    y_max=cy + r,
                    image_width=image_width,
                    image_height=image_height,
                ),
                center=Point(x=cx, y=cy),
                color=slice_elem.color,
            ))
        
        return elements
    
    def _classify_chart(
        self,
        image_bgr: np.ndarray,
        chart_id: str,
    ) -> ChartType:
        """
        Classify chart type using best available classifier.
        
        Priority order:
        1. ResNet-18 (94.14% accuracy)
        2. Random Forest ML classifier
        3. Rule-based classifier
        
        Args:
            image_bgr: BGR image array
            chart_id: Chart identifier for logging
        
        Returns:
            ChartType enum value
        """
        # Try ResNet-18 first (best accuracy)
        if self.resnet_classifier is not None:
            try:
                chart_type_str, confidence = self.resnet_classifier.predict_with_confidence(image_bgr)
                chart_type = ChartType(chart_type_str)
                self.logger.debug(
                    f"ResNet classification | chart_id={chart_id} | "
                    f"type={chart_type.value} | confidence={confidence:.2f}"
                )
                return chart_type
            except Exception as e:
                self.logger.warning(f"ResNet classification failed: {e}. Trying fallback.")
        
        # Fall back to Random Forest
        if self.ml_classifier is not None:
            try:
                ml_result = self.ml_classifier.classify(image_bgr, chart_id=chart_id)
                self.logger.debug(
                    f"ML classification | chart_id={chart_id} | "
                    f"type={ml_result.chart_type.value} | confidence={ml_result.confidence:.2f}"
                )
                return ml_result.chart_type
            except Exception as e:
                self.logger.warning(f"ML classification failed: {e}. Using rule-based.")
        
        # Final fallback: rule-based (requires extracted features)
        self.logger.debug(f"Using rule-based classification | chart_id={chart_id}")
        return ChartType.UNKNOWN
    
    def _calibrate_axes(
        self,
        texts: List[OCRText],
        img_width: int,
        img_height: int,
    ) -> Optional[AxisInfo]:
        """
        Calibrate axis mapping from OCR tick labels.
        
        Args:
            texts: OCR results
            img_width: Image width
            img_height: Image height
        
        Returns:
            AxisInfo if calibration successful (includes confidence scores)
        """
        # Extract Y-axis tick values
        y_ticks = self.ocr_engine.extract_axis_values(texts, axis="y")
        
        # Extract X-axis tick values
        x_ticks = self.ocr_engine.extract_axis_values(texts, axis="x")
        
        axis_info = AxisInfo(
            x_axis_detected=len(x_ticks) >= 2,
            y_axis_detected=len(y_ticks) >= 2,
        )
        
        # Calibrate Y-axis
        if y_ticks:
            y_result = self.mapper.calibrate_y_axis(y_ticks)
            if y_result:
                # Get value range from ticks
                y_values = [t[1] for t in y_ticks]
                axis_info.y_min = min(y_values)
                axis_info.y_max = max(y_values)
                axis_info.y_scale_factor = y_result.scale.slope
                # Add confidence from calibration
                axis_info.y_calibration_confidence = y_result.confidence
                axis_info.y_outliers_removed = y_result.outliers_removed
        
        # Calibrate X-axis
        if x_ticks:
            x_result = self.mapper.calibrate_x_axis(x_ticks)
            if x_result:
                x_values = [t[1] for t in x_ticks]
                axis_info.x_min = min(x_values)
                axis_info.x_max = max(x_values)
                axis_info.x_scale_factor = x_result.scale.slope
                # Add confidence from calibration
                axis_info.x_calibration_confidence = x_result.confidence
                axis_info.x_outliers_removed = x_result.outliers_removed
        
        return axis_info
    
    def _save_debug_images(self, chart_id: str, preprocess_result) -> None:
        """Save intermediate images for debugging."""
        if not self.config.debug_output_dir:
            return
        
        output_dir = Path(self.config.debug_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save negative image
        neg_path = output_dir / f"{chart_id}_1_negative.png"
        cv2.imwrite(str(neg_path), preprocess_result.negative_image)
        
        # Save binary image
        bin_path = output_dir / f"{chart_id}_2_binary.png"
        cv2.imwrite(str(bin_path), preprocess_result.binary_image)
        
        self.logger.debug(f"Debug images saved to {output_dir}")
    
    def get_vectorized_representation(
        self,
        chart,
        include_values: bool = True,
    ) -> VectorizedChart:
        """
        Get full vectorized representation of a chart.
        
        This is the "mathematical identity" output described in
        the Geo-SLM research: pure geometric data ready for
        SLM reasoning without any pixel dependencies.
        
        Args:
            chart: DetectedChart from Stage 2
            include_values: Whether to map to data values
        
        Returns:
            VectorizedChart with complete geometric data
        """
        chart_id = chart.chart_id
        
        # Load and process image
        image_bgr = cv2.imread(str(chart.cropped_path))
        h, w = image_bgr.shape[:2]
        
        # Preprocess
        preprocess_result = self.preprocessor.process(image_bgr, chart_id)
        
        # Skeletonize
        skeleton_result = self.skeletonizer.process(
            preprocess_result.binary_image,
            chart_id=chart_id,
        )
        
        # Trace and vectorize
        paths = self.skeletonizer.trace_paths(
            skeleton_result.skeleton,
            skeleton_result.keypoints,
        )
        
        vector_result = self.vectorizer.process(
            paths,
            stroke_width_map=skeleton_result.stroke_width_map,
            grayscale_image=preprocess_result.grayscale_image,
            chart_id=chart_id,
        )
        
        # Detect elements
        element_result = self.element_detector.detect(
            preprocess_result.binary_image,
            color_image=image_bgr,
            chart_id=chart_id,
        )
        
        # Build skeleton graph
        skeleton_graph = SkeletonGraph(
            keypoints=skeleton_result.keypoints,
            edges=[],
            polylines=vector_result.polylines,
            bars=element_result.bars,
            slices=element_result.slices,
            markers=element_result.markers,
        )
        
        return VectorizedChart(
            chart_id=chart_id,
            image_width=w,
            image_height=h,
            is_negative_processed=True,
            skeleton=skeleton_graph,
            preprocessing_applied=preprocess_result.operations_applied,
            vectorization_epsilon=self.config.vectorize.epsilon,
        )
