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
    OCRText,
    RawMetadata,
    Stage2Output,
    Stage3Output,
)
from ..base import BaseStage
from .classifier import ChartClassifier, ClassifierConfig
from .element_detector import ElementDetector, ElementDetectorConfig
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
        default="paddleocr",
        description="OCR engine: 'paddleocr' or 'tesseract'"
    )
    
    # Submodule configs
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    skeleton: SkeletonConfig = Field(default_factory=SkeletonConfig)
    vectorize: VectorizeConfig = Field(default_factory=VectorizeConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
    elements: ElementDetectorConfig = Field(default_factory=ElementDetectorConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    
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
        self.element_detector = ElementDetector(config.elements)
        self.classifier = ChartClassifier(config.classifier)
        
        # ML classifier (optional, used if use_ml_classifier=True)
        self.ml_classifier = None
        if config.use_ml_classifier:
            try:
                self.ml_classifier = MLChartClassifier()
                self.logger.info("ML classifier loaded successfully")
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
    
    def process_image(self, image_bgr: np.ndarray, chart_id: str = "test") -> RawMetadata:
        """
        Process a raw image directly (for testing/standalone use).
        
        Args:
            image_bgr: BGR image array
            chart_id: Identifier for logging
        
        Returns:
            RawMetadata with extracted information
        """
        h, w = image_bgr.shape[:2]
        
        # Use ML classifier if available
        chart_type = ChartType.UNKNOWN
        if self.config.enable_classification:
            if self.ml_classifier is not None:
                try:
                    ml_result = self.ml_classifier.classify(image_bgr, chart_id=chart_id)
                    chart_type = ml_result.chart_type
                except Exception as e:
                    self.logger.warning(f"ML classification failed: {e}")
        
        # Preprocessing
        preprocess_result = self.preprocessor.process(image_bgr, chart_id)
        
        # Vectorization
        polylines = []
        if self.config.enable_vectorization:
            skeleton_result = self.skeletonizer.process(
                preprocess_result.binary_image,
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
        
        # Element detection
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
            elements = self._convert_elements(bars, markers, slices)
        
        # OCR
        texts = []
        if self.config.enable_ocr:
            ocr_result = self.ocr_engine.extract_text(image_bgr, chart_id)
            texts = ocr_result.texts
        
        return RawMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            texts=texts,
            elements=elements,
            axis_info=None,
        )

    def _process_single_chart(self, chart) -> RawMetadata:
        """
        Process a single chart through extraction pipeline.
        
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
        
        # Step 1: Preprocessing (negative image + thresholding)
        preprocess_result = self.preprocessor.process(image_bgr, chart_id)
        
        # Save debug images if configured
        if self.config.save_debug_images and self.config.debug_output_dir:
            self._save_debug_images(chart_id, preprocess_result)
        
        # Step 2: Skeleton extraction
        skeleton_result = None
        polylines = []
        keypoints = []
        
        if self.config.enable_vectorization:
            skeleton_result = self.skeletonizer.process(
                preprocess_result.binary_image,
                chart_id=chart_id,
            )
            
            # Step 3: Trace paths from skeleton
            paths = self.skeletonizer.trace_paths(
                skeleton_result.skeleton,
                skeleton_result.keypoints,
            )
            
            # Step 4: Vectorize paths with RDP
            vector_result = self.vectorizer.process(
                paths,
                stroke_width_map=skeleton_result.stroke_width_map,
                grayscale_image=preprocess_result.grayscale_image,
                chart_id=chart_id,
            )
            
            polylines = vector_result.polylines
            keypoints = skeleton_result.keypoints
        
        # Step 5: Detect discrete elements (bars, markers)
        elements = []
        bars = []
        markers = []
        slices = []
        
        if self.config.enable_element_detection:
            element_result = self.element_detector.detect(
                preprocess_result.binary_image,
                color_image=image_bgr,
                chart_id=chart_id,
            )
            
            bars = element_result.bars
            markers = element_result.markers
            slices = element_result.slices
            
            # Convert to ChartElement format
            elements = self._convert_elements(bars, markers, slices)
        
        # Step 6: OCR text extraction
        texts = []
        if self.config.enable_ocr:
            ocr_result = self.ocr_engine.extract_text(image_bgr, chart_id)
            texts = ocr_result.texts
        
        # Step 7: Axis calibration from OCR
        axis_info = self._calibrate_axes(texts, w, h)
        
        # Step 8: Chart type classification
        chart_type = ChartType.UNKNOWN
        if self.config.enable_classification:
            # Use ML classifier if available, otherwise fall back to rule-based
            if self.ml_classifier is not None:
                try:
                    ml_result = self.ml_classifier.classify(image_bgr, chart_id=chart_id)
                    chart_type = ml_result.chart_type
                    self.logger.debug(
                        f"ML classification | chart_id={chart_id} | "
                        f"type={chart_type.value} | confidence={ml_result.confidence:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"ML classification failed: {e}. Using rule-based.")
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
            else:
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
        
        # Build RawMetadata
        return RawMetadata(
            chart_id=chart_id,
            chart_type=chart_type,
            texts=texts,
            elements=elements,
            axis_info=axis_info,
        )
    
    def _convert_elements(
        self,
        bars,
        markers,
        slices,
    ) -> List[ChartElement]:
        """Convert detected elements to ChartElement format."""
        elements = []
        
        # Convert bars
        for bar in bars:
            elements.append(ChartElement(
                element_type=ElementType.BAR.value,
                bbox=BoundingBox(
                    x_min=int(bar.x_min),
                    y_min=int(bar.y_min),
                    x_max=int(bar.x_max),
                    y_max=int(bar.y_max),
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
                bbox=BoundingBox(
                    x_min=int(marker.center.x - size // 2),
                    y_min=int(marker.center.y - size // 2),
                    x_max=int(marker.center.x + size // 2),
                    y_max=int(marker.center.y + size // 2),
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
                bbox=BoundingBox(
                    x_min=cx - r,
                    y_min=cy - r,
                    x_max=cx + r,
                    y_max=cy + r,
                ),
                center=Point(x=cx, y=cy),
                color=slice_elem.color,
            ))
        
        return elements
    
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
            AxisInfo if calibration successful
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
        
        # Calibrate X-axis
        if x_ticks:
            x_result = self.mapper.calibrate_x_axis(x_ticks)
            if x_result:
                x_values = [t[1] for t in x_ticks]
                axis_info.x_min = min(x_values)
                axis_info.x_max = max(x_values)
                axis_info.x_scale_factor = x_result.scale.slope
        
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
