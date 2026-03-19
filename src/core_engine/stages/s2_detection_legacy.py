"""
[DEPRECATED] Legacy Stage 2: Detection & Localization

This module is superseded by s2_detection.py which uses the updated
detection pipeline with improved NMS, multi-chart support, and
integration with the VLM extraction backend.

Retained for backward compatibility with older scripts and tests.
Do NOT use in new code.

Original responsibilities:
- Load trained YOLO model
- Run inference on clean images from Stage 1
- Filter detections by confidence threshold
- Crop detected charts to individual images
- Record bounding box information for traceability

Reference: docs/architecture/PIPELINE_FLOW.md#stage-2-detection
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..exceptions import StageProcessingError, ModelNotLoadedError
from ..schemas.common import BoundingBox
from ..schemas.stage_outputs import (
    Stage1Output,
    Stage2Output,
    DetectedChart,
)
from .base import BaseStage

logger = logging.getLogger(__name__)


class DetectionConfig(BaseModel):
    """Configuration for Stage 2: Detection."""
    
    # Model settings
    model_path: Path = Field(..., description="Path to trained YOLO model (.pt file)")
    device: str = Field(default="auto", description="Device: 'cpu', 'cuda', 'mps', or 'auto'")
    
    # Inference settings
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IOU threshold for NMS")
    imgsz: int = Field(default=640, description="Inference image size")
    
    # Detection filtering
    min_area_pixels: int = Field(default=100, description="Minimum detection area in pixels")
    max_detections_per_image: int = Field(default=50, description="Max detections per image")
    
    # Output settings
    save_cropped_images: bool = Field(default=True, description="Save cropped chart images")
    save_annotations: bool = Field(default=True, description="Save detection annotations")


class Stage2Detection(BaseStage):
    """
    Stage 2 Orchestrator: Detection & Localization
    
    Uses trained YOLO model to detect chart regions in clean images
    from Stage 1, then crops and saves individual charts.
    
    Examples:
        config = DetectionConfig(
            model_path=Path("results/training_runs/.../weights/best.pt"),
            conf_threshold=0.5
        )
        stage = Stage2Detection(config)
        result = stage.process(stage1_output)
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize Stage 2: Detection.
        
        Args:
            config: Configuration object with model path and settings
        
        Raises:
            ModelNotLoadedError: If model cannot be loaded
        """
        super().__init__(config)
        self.config = config
        self.model = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load model
        self._load_model()
    
    def process(self, input_data: Stage1Output) -> Stage2Output:
        """
        Process Stage 1 output and detect charts.
        
        Args:
            input_data: Stage1Output with list of clean images
        
        Returns:
            Stage2Output with detected charts
        
        Raises:
            StageProcessingError: If detection fails
        """
        if not self.validate_input(input_data):
            raise StageProcessingError(
                "Stage2Detection",
                f"Invalid input: expected Stage1Output, got {type(input_data)}"
            )
        
        session = input_data.session
        self.logger.info(
            f"Detection started | session={session.session_id} | "
            f"images={len(input_data.images)}"
        )
        
        detected_charts = []
        skipped_count = 0
        
        # Process each image
        for page_num, clean_image in enumerate(input_data.images, 1):
            try:
                # Load image
                image = Image.open(str(clean_image.image_path))
                image_array = np.array(image)
                
                # Run YOLO inference
                results = self.model.predict(
                    image_array,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.imgsz,
                    verbose=False,
                )
                
                if not results or len(results) == 0:
                    self.logger.debug(
                        f"No detections | session={session.session_id} | "
                        f"page={page_num}"
                    )
                    continue
                
                # Extract and filter detections
                detections = self._extract_detections(results[0], page_num)
                
                if not detections:
                    self.logger.debug(
                        f"No valid detections after filtering | session={session.session_id} | "
                        f"page={page_num}"
                    )
                    continue
                
                # Crop and save charts
                for det_idx, detection in enumerate(detections, 1):
                    try:
                        chart_id = f"{session.session_id}_p{page_num:03d}_c{det_idx:03d}"
                        
                        # YOLO returns coordinates in original image space (no scaling needed)
                        # bbox.x_min, etc. are already in pixels of the original image
                        cropped = image.crop((
                            detection['bbox'].x_min,
                            detection['bbox'].y_min,
                            detection['bbox'].x_max,
                            detection['bbox'].y_max,
                        ))
                        
                        # Save cropped image
                        if self.config.save_cropped_images:
                            output_dir = clean_image.image_path.parent / "cropped_charts"
                            output_dir.mkdir(exist_ok=True)
                            cropped_path = output_dir / f"{chart_id}.png"
                            cropped.save(str(cropped_path))
                        else:
                            cropped_path = None
                        
                        # Create detection record
                        detected_chart = DetectedChart(
                            chart_id=chart_id,
                            source_image=clean_image.image_path,
                            cropped_path=cropped_path or clean_image.image_path,
                            bbox=detection['bbox'],
                            page_number=page_num,
                        )
                        detected_charts.append(detected_chart)
                        
                        self.logger.debug(
                            f"Chart detected and cropped | session={session.session_id} | "
                            f"chart_id={chart_id} | conf={detection['bbox'].confidence:.3f}"
                        )
                        
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to crop chart | session={session.session_id} | "
                            f"page={page_num} | det={det_idx} | error={str(e)}"
                        )
                        continue
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to process image | session={session.session_id} | "
                    f"page={page_num} | error={str(e)}"
                )
                skipped_count += 1
                continue
        
        # Create result
        result = Stage2Output(
            session=session,
            charts=detected_charts,
            total_detected=len(detected_charts),
            skipped_low_confidence=skipped_count,
        )
        
        self.logger.info(
            f"Detection complete | session={session.session_id} | "
            f"charts_detected={len(detected_charts)} | skipped={skipped_count}"
        )
        
        return result
    
    def _load_model(self):
        """
        Load YOLO model from file.
        
        Raises:
            ModelNotLoadedError: If model cannot be loaded
        """
        try:
            from ultralytics import YOLO
            
            model_path = self.config.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
            
            # Set device
            if self.config.device == "auto":
                device = None  # YOLO will auto-detect
            else:
                device = self.config.device
            
            if device:
                self.model.to(device)
            
            self.logger.info(f"Model loaded successfully on device: {device or 'auto'}")
            
        except ImportError as e:
            raise ModelNotLoadedError(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to load model: {str(e)}") from e
    
    def _extract_detections(self, yolo_result, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract and filter detections from YOLO results.
        
        Args:
            yolo_result: YOLO detection result object
            page_num: Page number for logging
        
        Returns:
            List of valid detections with bbox information
        """
        detections = []
        
        if not hasattr(yolo_result, 'boxes') or yolo_result.boxes is None:
            return detections
        
        boxes = yolo_result.boxes
        
        # Extract coordinates and confidence
        for i, box in enumerate(boxes):
            try:
                # Get coordinates (YOLO returns xyxy format)
                coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(box.conf[0].cpu().numpy())
                
                x_min, y_min, x_max, y_max = coords
                x_min, y_min = int(x_min), int(y_min)
                x_max, y_max = int(x_max), int(y_max)
                
                # Calculate area
                area = (x_max - x_min) * (y_max - y_min)
                
                # Filter by confidence (already done by model, but double-check)
                if confidence < self.config.conf_threshold:
                    continue
                
                # Filter by minimum area
                if area < self.config.min_area_pixels:
                    self.logger.debug(
                        f"Detection skipped (too small) | page={page_num} | "
                        f"area={area} | conf={confidence:.3f}"
                    )
                    continue
                
                # Create bbox record
                bbox = BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=confidence,
                )
                
                detections.append({
                    'bbox': bbox,
                    'area': area,
                    'index': i,
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to extract detection {i}: {str(e)}")
                continue
        
        # Sort by confidence (descending) and limit
        detections.sort(key=lambda x: x['bbox'].confidence, reverse=True)
        detections = detections[:self.config.max_detections_per_image]
        
        return detections
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate stage input.
        
        Args:
            input_data: Should be Stage1Output
        
        Returns:
            True if valid
        """
        return isinstance(input_data, Stage1Output) and len(input_data.images) > 0
