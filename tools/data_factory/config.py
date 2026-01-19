"""
Data Factory Configuration

Centralized configuration for data collection pipeline.
All paths and constants are defined here.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (tools/data_factory/config.py -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input directories
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
SEARCH_CACHE_DIR = DATA_DIR / "search_cache"

# Output directories
ACADEMIC_DATASET_DIR = DATA_DIR / "academic_dataset"
IMAGES_DIR = ACADEMIC_DATASET_DIR / "images"
METADATA_DIR = ACADEMIC_DATASET_DIR / "metadata"
ANNOTATIONS_DIR = ACADEMIC_DATASET_DIR / "annotations"
MANIFESTS_DIR = ACADEMIC_DATASET_DIR / "manifests"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Training output
TRAINING_DIR = DATA_DIR / "training"
YOLO_DATASET_DIR = TRAINING_DIR / "yolo_format"


# =============================================================================
# SEARCH QUERIES
# =============================================================================

ARXIV_QUERIES: List[str] = [
    # Computer Vision & Charts
    'cat:cs.CV AND (chart OR visualization OR diagram)',
    'cat:cs.CV AND "bar chart"',
    'cat:cs.CV AND "line chart"',
    'cat:cs.CV AND "pie chart"',
    'cat:cs.CV AND "scatter plot"',
    'cat:cs.CV AND figure',
    
    # Machine Learning (often has performance charts)
    'cat:cs.LG AND (benchmark OR comparison OR performance)',
    'cat:cs.LG AND "accuracy" AND "training"',
    'cat:cs.LG AND ablation',
    'cat:cs.LG AND results',
    'cat:cs.LG AND evaluation',
    
    # NLP (lots of benchmark charts)
    'cat:cs.CL AND benchmark',
    'cat:cs.CL AND evaluation',
    'cat:cs.CL AND performance',
    'cat:cs.CL AND results',
    
    # AI General
    'cat:cs.AI AND benchmark',
    'cat:cs.AI AND evaluation',
    'cat:cs.AI AND performance',
    
    # Statistics
    'cat:stat.ML AND visualization',
    'cat:stat.AP AND (chart OR graph)',
    'cat:stat.ME AND figure',
    
    # Economics/Finance (lots of charts)
    'cat:econ.GN AND (chart OR figure)',
    'cat:q-fin.ST AND (chart OR visualization)',
    'cat:econ.EM AND figure',
    
    # Physics (experimental results)
    'cat:physics.data-an AND figure',
    'cat:hep-ex AND figure',
    
    # Biology/Medicine
    'cat:q-bio.QM AND figure',
    
    # Data Science
    'cat:cs.DB AND visualization',
    'cat:cs.IR AND evaluation',
    
    # Specific chart analysis papers
    '"chart understanding"',
    '"chart question answering"',
    '"chart parsing"',
    '"data visualization" AND extraction',
    '"figure extraction"',
    '"document understanding"',
    
    # Recent popular topics
    'large language model AND benchmark',
    'transformer AND evaluation',
    'deep learning AND comparison',
    'neural network AND performance',
]

GOOGLE_SEARCH_QUERIES: List[str] = [
    # Basic chart types
    "bar chart example high quality",
    "line chart data visualization",
    "pie chart infographic",
    "scatter plot example",
    "area chart visualization",
    
    # Business/Finance charts
    "financial chart stock market",
    "sales chart business report",
    "revenue growth chart",
    
    # Scientific charts
    "scientific data chart research",
    "statistical chart academic",
    "experiment results chart",
    
    # Specific styles
    "matplotlib chart example",
    "excel chart high resolution",
    "tableau visualization chart",
    "powerpoint chart presentation",
]


# =============================================================================
# HUGGINGFACE DATASETS CONFIGURATION
# =============================================================================

HUGGINGFACE_DATASETS = {
    "chartqa": {
        "repo_id": "ahmed-masry/ChartQA",
        "description": "9.6k chart images with QA pairs",
        "expected_samples": 9608,
        "has_images": True,
        "priority": 1,  # Download first
    },
    "plotqa": {
        "repo_id": "cmu-rl/plotqa",
        "description": "224k scientific plots with QA",
        "expected_samples": 224377,
        "has_images": True,
        "priority": 2,
    },
    "dvqa": {
        "repo_id": "dvqa/dvqa",
        "description": "3.5M synthetic bar charts",
        "expected_samples": 300000,
        "has_images": True,
        "priority": 3,
    },
    "chart2text": {
        "repo_id": "khangnn/chart2text",
        "description": "Charts with text summaries",
        "expected_samples": 44096,
        "has_images": True,
        "priority": 4,
    },
    "unichart": {
        "repo_id": "ahmed-masry/UniChart",
        "description": "Multi-task chart dataset",
        "expected_samples": 20000,
        "has_images": True,
        "priority": 5,
    },
}

# Priority order for quick start (most diverse first)
HF_QUICK_START_DATASETS = ["chartqa", "chart2text", "unichart"]


# =============================================================================
# PMC (PUBMED CENTRAL) CONFIGURATION
# =============================================================================

PMC_SEARCH_QUERIES = [
    # Clinical trials with charts
    '"clinical trial" AND (chart OR figure OR graph)',
    '"meta-analysis" AND (forest plot OR bar chart)',
    '"systematic review" AND (visualization OR diagram)',
    
    # Epidemiology (lots of trend charts)
    'epidemiology AND (time series OR trend chart)',
    'prevalence AND (bar chart OR pie chart)',
    
    # Bioinformatics (performance benchmarks)
    'bioinformatics AND (benchmark OR comparison)',
    'genomics AND (visualization OR heatmap)',
]


# =============================================================================
# ACL ANTHOLOGY CONFIGURATION
# =============================================================================

ACL_ANTHOLOGY_VENUES = [
    "acl",      # ACL main conference
    "emnlp",    # EMNLP
    "naacl",    # NAACL
    "eacl",     # EACL
    "findings", # ACL Findings
]

ACL_SEARCH_KEYWORDS = [
    "benchmark",
    "evaluation",
    "performance",
    "comparison",
    "analysis",
    "visualization",
]


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

class QualityConfig(BaseModel):
    """Image quality filtering thresholds."""
    
    # Minimum dimensions (pixels)
    min_width: int = Field(default=300, ge=100)
    min_height: int = Field(default=300, ge=100)
    
    # Maximum dimensions (to avoid huge images)
    max_width: int = Field(default=4096, le=8192)
    max_height: int = Field(default=4096, le=8192)
    
    # Aspect ratio limits (width/height)
    min_aspect_ratio: float = Field(default=0.3, ge=0.1)
    max_aspect_ratio: float = Field(default=3.0, le=10.0)
    
    # Minimum file size (bytes) - filter tiny images
    min_file_size: int = Field(default=5000, ge=1000)
    
    # Uniformity threshold (0-1, lower = more uniform = likely blank)
    max_uniformity: float = Field(default=0.95, le=1.0)
    
    # Minimum unique colors (filter solid color images)
    min_unique_colors: int = Field(default=50, ge=10)


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

class DataFactoryConfig(BaseModel):
    """Main configuration for Data Factory."""
    
    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_pdfs_dir: Path = RAW_PDFS_DIR
    images_dir: Path = IMAGES_DIR
    metadata_dir: Path = METADATA_DIR
    
    # Arxiv settings
    arxiv_queries: List[str] = Field(default_factory=lambda: ARXIV_QUERIES)
    arxiv_max_results_per_query: int = Field(default=50, ge=1)
    arxiv_rate_limit_seconds: float = Field(default=3.0, ge=1.0)
    
    # Google Search settings
    google_queries: List[str] = Field(default_factory=lambda: GOOGLE_SEARCH_QUERIES)
    google_max_results_per_query: int = Field(default=20, ge=1)
    serpapi_key: str | None = Field(default=None)
    
    # Quality settings
    quality: QualityConfig = Field(default_factory=QualityConfig)
    
    # Processing settings
    max_workers: int = Field(default=4, ge=1, le=16)
    request_timeout: int = Field(default=60, ge=10)
    
    # Random seed for reproducibility
    random_seed: int = Field(default=42)
    
    def ensure_directories(self) -> None:
        """Create all required directories."""
        directories = [
            self.raw_pdfs_dir,
            self.images_dir,
            self.metadata_dir,
            ANNOTATIONS_DIR,
            SEARCH_CACHE_DIR,
            YOLO_DATASET_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# VERIFIED ARXIV IDS (Known good papers with charts)
# =============================================================================

VERIFIED_ARXIV_IDS: List[str] = [
    # Chart Understanding Papers
    "1906.02337",  # Chart understanding
    "2203.10244",  # ChartQA paper
    "2109.02226",  # Chart parsing
    "2010.09710",  # PlotQA paper
    "2205.00557",  # Pix2Struct
    "2201.08264",  # Matcha
    
    # ML Papers with Performance Charts
    "1512.03385",  # ResNet
    "1706.03762",  # Transformer
    "2010.11929",  # ViT
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    
    # Recent Papers
    "2302.13971",  # LLaMA
    "2307.09288",  # LLaMA 2
    "2303.08774",  # GPT-4
    "2312.11805",  # Gemini
    
    # Data Visualization Research
    "1808.00257",  # Data visualization survey
    "2007.14330",  # Chart classification
    "2111.04509",  # Figure extraction
]


# =============================================================================
# ROBOFLOW DATASETS (Pre-annotated)
# =============================================================================

ROBOFLOW_DATASETS = {
    "chart-datasets": {
        "workspace": "object-detection-u4mcr",
        "project": "chart-datasets",
        "version": 1,
        "description": "2.9k images - line, bar, pie charts",
        "classes": ["line", "bar", "pie"],
    },
    "chart-classification": {
        "workspace": "chartclassification",
        "project": "chart-classification",
        "version": 1,
        "description": "1.3k images - 19 chart types",
        "classes": [
            "Area-chart", "Bar-chart", "Line-chart", "Pie-chart",
            "Scatter-chart", "Doughnut-chart", "Radar-chart",
        ],
    },
}
