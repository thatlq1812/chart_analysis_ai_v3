"""
Data Factory Schemas

Pydantic models for data integrity and validation.
All data structures used in the data collection pipeline.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field


# =============================================================================
# ENUMS
# =============================================================================

class DataSource(str, Enum):
    """Source of the data."""
    
    ARXIV = "arxiv"
    HUGGINGFACE = "huggingface"
    PMC = "pmc"
    ACL = "acl"
    ROBOFLOW = "roboflow"
    SYNTHETIC = "synthetic"
    MANUAL = "manual"


class ChartType(str, Enum):
    """Chart type classification."""
    
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    STACKED_BAR = "stacked_bar"
    GROUPED_BAR = "grouped_bar"
    DONUT = "donut"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Status of processing."""
    
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# ARXIV PAPER SCHEMA
# =============================================================================

class ArxivPaper(BaseModel):
    """Metadata for an Arxiv paper."""
    
    arxiv_id: str = Field(..., description="Arxiv paper ID (e.g., '2203.10244')")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(default="", description="Paper abstract")
    published_date: datetime = Field(..., description="Publication date")
    pdf_url: str = Field(..., description="Direct PDF download URL")
    
    # Local paths
    local_pdf_path: Optional[Path] = Field(default=None, description="Local PDF path")
    
    # Processing status
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    extracted_images_count: int = Field(default=0)
    
    # Metadata
    categories: List[str] = Field(default_factory=list)
    search_query: Optional[str] = Field(default=None, description="Query that found this paper")
    
    @computed_field
    @property
    def safe_id(self) -> str:
        """ID safe for filenames (replace dots with underscores)."""
        return self.arxiv_id.replace(".", "_").replace("/", "_")


# =============================================================================
# CHART IMAGE SCHEMA
# =============================================================================

class BoundingBox(BaseModel):
    """Bounding box in XYXY format."""
    
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., gt=0)
    y_max: int = Field(..., gt=0)
    
    @computed_field
    @property
    def width(self) -> int:
        return self.x_max - self.x_min
    
    @computed_field
    @property
    def height(self) -> int:
        return self.y_max - self.y_min
    
    def to_yolo(self, img_width: int, img_height: int) -> tuple:
        """Convert to YOLO format (x_center, y_center, width, height) normalized."""
        x_center = ((self.x_min + self.x_max) / 2) / img_width
        y_center = ((self.y_min + self.y_max) / 2) / img_height
        width = self.width / img_width
        height = self.height / img_height
        return (x_center, y_center, width, height)


class ChartImage(BaseModel):
    """Metadata for an extracted chart image."""
    
    # Identification
    image_id: str = Field(..., description="Unique image ID")
    source: DataSource = Field(..., description="Data source")
    
    # Source reference
    parent_paper_id: Optional[str] = Field(default=None, description="Arxiv ID if from paper")
    source_url: Optional[str] = Field(default=None, description="Original URL")
    page_number: Optional[int] = Field(default=None, description="Page in PDF")
    
    # Local paths
    image_path: Path = Field(..., description="Path to saved image")
    thumbnail_path: Optional[Path] = Field(default=None)
    
    # Image properties
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    file_size_bytes: int = Field(..., gt=0)
    format: str = Field(default="png")
    
    # Chart metadata
    chart_type: ChartType = Field(default=ChartType.UNKNOWN)
    caption_text: Optional[str] = Field(default=None, description="Figure caption from paper")
    context_text: Optional[str] = Field(default=None, description="Surrounding text")
    
    # Bounding box (if from larger image)
    bbox: Optional[BoundingBox] = Field(default=None)
    
    # Quality metrics
    is_valid: bool = Field(default=True)
    quality_score: float = Field(default=1.0, ge=0, le=1)
    validation_notes: List[str] = Field(default_factory=list)
    
    # Timestamps
    extracted_at: datetime = Field(default_factory=datetime.now)
    
    @computed_field
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0


# =============================================================================
# DATA MANIFEST SCHEMA
# =============================================================================

class DatasetStatistics(BaseModel):
    """Statistics about the dataset."""
    
    total_images: int = 0
    total_papers: int = 0
    
    # By source
    images_from_arxiv: int = 0
    images_from_google: int = 0
    images_from_roboflow: int = 0
    images_from_synthetic: int = 0
    
    # By chart type
    chart_type_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Quality
    valid_images: int = 0
    invalid_images: int = 0
    average_quality_score: float = 0.0


class DataManifest(BaseModel):
    """Manifest file tracking dataset contents."""
    
    # Identification
    dataset_name: str = Field(default="geo_slm_chart_dataset")
    version: str = Field(default="1.0.0")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Content hash for versioning
    content_hash: Optional[str] = Field(default=None)
    
    # Statistics
    statistics: DatasetStatistics = Field(default_factory=DatasetStatistics)
    
    # Configuration used
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing log
    papers_processed: List[str] = Field(default_factory=list, description="List of processed paper IDs")
    queries_executed: List[str] = Field(default_factory=list)
    
    # Data splits
    train_ids: List[str] = Field(default_factory=list)
    val_ids: List[str] = Field(default_factory=list)
    test_ids: List[str] = Field(default_factory=list)


# =============================================================================
# SEARCH RESULT SCHEMAS
# =============================================================================

class GoogleSearchResult(BaseModel):
    """Result from Google image search."""
    
    query: str
    image_url: str
    thumbnail_url: Optional[str] = None
    source_page_url: Optional[str] = None
    title: Optional[str] = None
    
    # Local path after download
    local_path: Optional[Path] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)


class RoboflowDataset(BaseModel):
    """Information about a Roboflow dataset."""
    
    name: str
    workspace: str
    project: str
    version: int
    description: str
    classes: List[str]
    
    # Download status
    local_path: Optional[Path] = None
    images_count: int = 0
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
