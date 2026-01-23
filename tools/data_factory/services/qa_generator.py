"""
Chart QA Pipeline - Orchestrates chart classification and QA generation

This module provides the main pipeline for:
1. Extracting images from PDFs (using existing miner)
2. Classifying images as charts using Gemini
3. Generating QA pairs for training data
4. Managing multiprocessing and checkpointing
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

from loguru import logger
from tqdm import tqdm

from ..config import (
    ACADEMIC_DATASET_DIR,
    DATA_DIR,
    IMAGES_DIR,
    METADATA_DIR,
    RAW_PDFS_DIR,
)
from ..schemas import ChartImage, ChartType, DataSource
from .gemini_classifier import GeminiChartClassifier, GeminiRateLimiter


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

CHART_QA_DIR = ACADEMIC_DATASET_DIR / "chart_qa"
CLASSIFIED_DIR = CHART_QA_DIR / "classified"
CHARTS_DIR = CLASSIFIED_DIR / "charts"
NON_CHARTS_DIR = CLASSIFIED_DIR / "non_charts"
QA_PAIRS_DIR = CHART_QA_DIR / "qa_pairs"
CHECKPOINTS_DIR = CHART_QA_DIR / "checkpoints"
STATS_DIR = CHART_QA_DIR / "stats"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QAPipelineConfig:
    """Configuration for QA generation pipeline."""
    
    # Processing
    max_api_workers: int = 10
    max_pdf_workers: int = 4
    checkpoint_frequency: int = 100
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 32000
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Filtering
    min_image_size: int = 10000  # bytes
    skip_processed: bool = True


@dataclass
class PipelineProgress:
    """Track pipeline progress for checkpointing."""
    
    total_images: int = 0
    processed_images: int = 0
    charts_found: int = 0
    non_charts: int = 0
    qa_pairs_generated: int = 0
    errors: int = 0
    
    processed_ids: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "charts_found": self.charts_found,
            "non_charts": self.non_charts,
            "qa_pairs_generated": self.qa_pairs_generated,
            "errors": self.errors,
            "processed_ids": list(self.processed_ids),
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> "PipelineProgress":
        """Load progress from checkpoint file."""
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        
        progress = cls(
            total_images=data.get("total_images", 0),
            processed_images=data.get("processed_images", 0),
            charts_found=data.get("charts_found", 0),
            non_charts=data.get("non_charts", 0),
            qa_pairs_generated=data.get("qa_pairs_generated", 0),
            errors=data.get("errors", 0),
            processed_ids=set(data.get("processed_ids", [])),
        )
        return progress


@dataclass
class ChartQAResult:
    """Result of processing a single image."""
    
    image_id: str
    image_path: Path
    is_chart: bool
    chart_type: str
    confidence: float
    qa_pairs: List[Dict[str, str]]
    classification_raw: Dict[str, Any]
    error: Optional[str] = None
    processing_time: float = 0.0


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class ChartQAPipeline:
    """
    Main pipeline for chart classification and QA generation.
    
    Features:
    - Multiprocessing for API calls
    - Checkpointing for resume
    - Rate limiting for API
    - Progress tracking
    """
    
    def __init__(
        self,
        config: Optional[QAPipelineConfig] = None,
        classifier: Optional[GeminiChartClassifier] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            classifier: Pre-initialized Gemini classifier (optional)
        """
        self.config = config or QAPipelineConfig()
        self.classifier = classifier or GeminiChartClassifier()
        self.rate_limiter = GeminiRateLimiter(
            rpm=self.config.requests_per_minute,
            tpm=self.config.tokens_per_minute,
        )
        self.progress = PipelineProgress()
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info("ChartQAPipeline initialized")
    
    def _ensure_directories(self) -> None:
        """Create all required output directories."""
        for directory in [
            CHART_QA_DIR,
            CLASSIFIED_DIR,
            CHARTS_DIR,
            NON_CHARTS_DIR,
            QA_PAIRS_DIR,
            CHECKPOINTS_DIR,
            STATS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_checkpoint_path(self, session_id: str) -> Path:
        """Get checkpoint file path for a session."""
        return CHECKPOINTS_DIR / f"checkpoint_{session_id}.json"
    
    def _save_checkpoint(self, session_id: str) -> None:
        """Save current progress to checkpoint."""
        checkpoint_path = self._get_checkpoint_path(session_id)
        with open(checkpoint_path, "w") as f:
            json.dump(self.progress.to_dict(), f, indent=2)
        logger.debug(f"Checkpoint saved | processed={self.progress.processed_images}")
    
    def _load_checkpoint(self, session_id: str) -> bool:
        """Load progress from checkpoint if exists."""
        checkpoint_path = self._get_checkpoint_path(session_id)
        if checkpoint_path.exists():
            self.progress = PipelineProgress.from_checkpoint(checkpoint_path)
            logger.info(
                f"Checkpoint loaded | processed={self.progress.processed_images} | "
                f"charts={self.progress.charts_found}"
            )
            return True
        return False
    
    def _save_qa_result(self, result: ChartQAResult) -> None:
        """Save QA result to appropriate directory."""
        if result.error:
            return
        
        # Save classification result
        if result.is_chart:
            output_dir = CHARTS_DIR
        else:
            output_dir = NON_CHARTS_DIR
        
        # Create result JSON
        result_data = {
            "image_id": result.image_id,
            "image_path": str(result.image_path),
            "is_chart": result.is_chart,
            "chart_type": result.chart_type,
            "confidence": result.confidence,
            "classification": result.classification_raw,
            "processing_time_seconds": result.processing_time,
        }
        
        # Save classification
        class_path = output_dir / f"{result.image_id}.json"
        with open(class_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Save QA pairs if chart
        if result.is_chart and result.qa_pairs:
            qa_data = {
                "image_id": result.image_id,
                "image_path": str(result.image_path),
                "chart_type": result.chart_type,
                "qa_pairs": result.qa_pairs,
                "generated_at": datetime.now().isoformat(),
            }
            qa_path = QA_PAIRS_DIR / f"{result.image_id}_qa.json"
            with open(qa_path, "w") as f:
                json.dump(qa_data, f, indent=2)
    
    def process_single_image(self, image_path: Path) -> ChartQAResult:
        """
        Process a single image through the full pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ChartQAResult with classification and QA pairs
        """
        start_time = time.time()
        image_id = image_path.stem
        
        try:
            # Rate limit check
            self.rate_limiter.wait_if_needed(estimated_tokens=1500)
            
            # Full processing (classify + QA if chart)
            result_data = self.classifier.process_image_full(image_path)
            
            # Record API usage
            self.rate_limiter.record_request(tokens_used=1500)
            
            # Extract results
            classification = result_data.get("classification", {})
            is_chart = classification.get("is_chart", False)
            chart_type = classification.get("chart_type", "none")
            confidence = classification.get("confidence", 0.0)
            qa_pairs = result_data.get("qa_pairs", [])
            
            processing_time = time.time() - start_time
            
            return ChartQAResult(
                image_id=image_id,
                image_path=image_path,
                is_chart=is_chart,
                chart_type=chart_type,
                confidence=confidence,
                qa_pairs=qa_pairs,
                classification_raw=classification,
                processing_time=processing_time,
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return ChartQAResult(
                image_id=image_id,
                image_path=image_path,
                is_chart=False,
                chart_type="none",
                confidence=0.0,
                qa_pairs=[],
                classification_raw={},
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    def collect_images(
        self,
        source_dir: Optional[Path] = None,
        pattern: str = "*.png",
    ) -> List[Path]:
        """
        Collect all images to process.
        
        Args:
            source_dir: Directory containing images (default: IMAGES_DIR)
            pattern: Glob pattern for image files
            
        Returns:
            List of image paths
        """
        source_dir = source_dir or IMAGES_DIR
        
        # Support multiple patterns
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        images = []
        
        for pat in patterns:
            images.extend(source_dir.glob(pat))
        
        # Filter by size
        min_size = self.config.min_image_size
        images = [p for p in images if p.stat().st_size >= min_size]
        
        # Filter already processed if resuming
        if self.config.skip_processed and self.progress.processed_ids:
            images = [p for p in images if p.stem not in self.progress.processed_ids]
        
        logger.info(f"Collected {len(images)} images from {source_dir}")
        return sorted(images)
    
    def run(
        self,
        source_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> PipelineProgress:
        """
        Run the full pipeline.
        
        Args:
            source_dir: Directory containing images
            session_id: Session ID for checkpointing
            limit: Maximum images to process
            show_progress: Show progress bar
            
        Returns:
            Final pipeline progress
        """
        session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try to load checkpoint
        if self.config.skip_processed:
            self._load_checkpoint(session_id)
        
        # Collect images
        images = self.collect_images(source_dir)
        if limit:
            images = images[:limit]
        
        self.progress.total_images = len(images) + self.progress.processed_images
        
        logger.info(
            f"Starting pipeline | session={session_id} | "
            f"images={len(images)} | workers={self.config.max_api_workers}"
        )
        
        # Process with thread pool (I/O bound)
        with ThreadPoolExecutor(max_workers=self.config.max_api_workers) as executor:
            futures = {
                executor.submit(self.process_single_image, img): img
                for img in images
            }
            
            # Progress bar
            pbar = tqdm(
                total=len(images),
                desc="Processing images",
                disable=not show_progress,
            )
            
            for future in as_completed(futures):
                image_path = futures[future]
                
                try:
                    result = future.result()
                    
                    # Update progress
                    self.progress.processed_images += 1
                    self.progress.processed_ids.add(result.image_id)
                    
                    if result.error:
                        self.progress.errors += 1
                    elif result.is_chart:
                        self.progress.charts_found += 1
                        self.progress.qa_pairs_generated += len(result.qa_pairs)
                    else:
                        self.progress.non_charts += 1
                    
                    # Save result
                    self._save_qa_result(result)
                    
                    # Checkpoint
                    if self.progress.processed_images % self.config.checkpoint_frequency == 0:
                        self._save_checkpoint(session_id)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "charts": self.progress.charts_found,
                        "qa": self.progress.qa_pairs_generated,
                    })
                    
                except Exception as e:
                    logger.error(f"Future error for {image_path}: {e}")
                    self.progress.errors += 1
                    pbar.update(1)
            
            pbar.close()
        
        # Final checkpoint
        self._save_checkpoint(session_id)
        
        # Save final stats
        self._save_final_stats(session_id)
        
        logger.info(
            f"Pipeline complete | session={session_id} | "
            f"charts={self.progress.charts_found} | "
            f"qa_pairs={self.progress.qa_pairs_generated} | "
            f"errors={self.progress.errors}"
        )
        
        return self.progress
    
    def _save_final_stats(self, session_id: str) -> None:
        """Save final statistics."""
        stats = {
            **self.progress.to_dict(),
            "session_id": session_id,
            "completed_at": datetime.now().isoformat(),
            "config": {
                "max_api_workers": self.config.max_api_workers,
                "checkpoint_frequency": self.config.checkpoint_frequency,
                "requests_per_minute": self.config.requests_per_minute,
            },
        }
        
        stats_path = STATS_DIR / f"stats_{session_id}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Stats saved to {stats_path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def collect_qa_dataset(
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Collect all QA pairs into a single dataset file.
    
    Args:
        output_path: Output file path (default: chart_qa/dataset.json)
        
    Returns:
        Dataset statistics
    """
    output_path = output_path or (CHART_QA_DIR / "dataset.json")
    
    # Collect all QA files
    qa_files = list(QA_PAIRS_DIR.glob("*_qa.json"))
    
    dataset = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "total_images": len(qa_files),
        "total_qa_pairs": 0,
        "samples": [],
    }
    
    for qa_file in qa_files:
        with open(qa_file, "r") as f:
            data = json.load(f)
        
        dataset["samples"].append(data)
        dataset["total_qa_pairs"] += len(data.get("qa_pairs", []))
    
    # Save dataset
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(
        f"Dataset collected | images={dataset['total_images']} | "
        f"qa_pairs={dataset['total_qa_pairs']} | path={output_path}"
    )
    
    return {
        "total_images": dataset["total_images"],
        "total_qa_pairs": dataset["total_qa_pairs"],
        "output_path": str(output_path),
    }


def get_pipeline_status() -> Dict[str, Any]:
    """Get current pipeline status and statistics."""
    status = {
        "directories": {
            "charts": len(list(CHARTS_DIR.glob("*.json"))) if CHARTS_DIR.exists() else 0,
            "non_charts": len(list(NON_CHARTS_DIR.glob("*.json"))) if NON_CHARTS_DIR.exists() else 0,
            "qa_pairs": len(list(QA_PAIRS_DIR.glob("*.json"))) if QA_PAIRS_DIR.exists() else 0,
        },
        "checkpoints": [],
    }
    
    # List checkpoints
    if CHECKPOINTS_DIR.exists():
        for cp in CHECKPOINTS_DIR.glob("checkpoint_*.json"):
            with open(cp, "r") as f:
                data = json.load(f)
            status["checkpoints"].append({
                "session": cp.stem.replace("checkpoint_", ""),
                "processed": data.get("processed_images", 0),
                "charts": data.get("charts_found", 0),
            })
    
    return status


if __name__ == "__main__":
    # Quick test
    pipeline = ChartQAPipeline()
    status = get_pipeline_status()
    print(json.dumps(status, indent=2))
