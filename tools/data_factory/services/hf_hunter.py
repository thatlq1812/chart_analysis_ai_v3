"""
Hugging Face Dataset Hunter

Fast way to get high-quality chart datasets:
- ChartQA: 20k+ real-world charts with Q&A
- PlotQA: 224k charts with annotated data
- DVQA: Bar chart understanding dataset

This is the FASTEST path to 1,000+ samples.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from loguru import logger

from ..config import (
    DataFactoryConfig,
    IMAGES_DIR,
    METADATA_DIR,
)
from ..schemas import (
    ChartImage,
    ChartType,
    DataSource,
    ProcessingStatus,
)


# =============================================================================
# HUGGING FACE DATASETS REGISTRY
# =============================================================================

HF_CHART_DATASETS = {
    "chartqa": {
        "name": "HuggingFaceM4/ChartQA",
        "description": "Real-world charts with question answering",
        "image_column": "image",
        "splits": ["train", "val", "test"],
        "estimated_size": 20000,
        "chart_types": ["bar", "line", "pie", "scatter"],
    },
    "plotqa": {
        "name": "cmarkea/plotqa",
        "description": "Scientific plots with data extraction",
        "image_column": "image",
        "splits": ["train", "validation", "test"],
        "estimated_size": 224000,
        "chart_types": ["bar", "line", "scatter"],
    },
    "dvqa": {
        "name": "lmms-lab/DVQA",
        "description": "Bar charts with visual QA",
        "image_column": "image",
        "splits": ["train", "val", "test_familiar", "test_novel"],
        "estimated_size": 300000,
        "chart_types": ["bar"],
    },
    "chart2text": {
        "name": "khhuang/chart-to-text",
        "description": "Charts with natural language summaries",
        "image_column": "image",
        "splits": ["train", "validation", "test"],
        "estimated_size": 44000,
        "chart_types": ["bar", "line", "pie"],
    },
    "unichart": {
        "name": "ahmed-masry/UniChart",
        "description": "Universal chart understanding benchmark",
        "image_column": "image",
        "splits": ["train", "validation", "test"],
        "estimated_size": 611000,
        "chart_types": ["bar", "line", "pie", "scatter", "area"],
    },
}


# =============================================================================
# HUGGING FACE HUNTER
# =============================================================================

class HuggingFaceHunter:
    """
    Hunter for Hugging Face chart datasets.
    
    This is the fastest way to get quality chart data:
    - Pre-labeled with chart types
    - Has Q&A pairs (useful for SLM training)
    - Clean images, no scraping issues
    
    Example:
        hunter = HuggingFaceHunter(config)
        images = hunter.download_dataset("chartqa", limit=1000)
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        self._ensure_datasets_library()
    
    def _ensure_datasets_library(self) -> None:
        """Ensure Hugging Face datasets library is available."""
        try:
            from datasets import load_dataset
            self.load_dataset = load_dataset
            self._available = True
        except ImportError:
            logger.error(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )
            self._available = False
    
    @property
    def available_datasets(self) -> Dict:
        """Return available datasets info."""
        return HF_CHART_DATASETS
    
    def list_datasets(self) -> None:
        """Print available datasets."""
        print("\n" + "=" * 60)
        print("AVAILABLE HUGGING FACE CHART DATASETS")
        print("=" * 60)
        
        for key, info in HF_CHART_DATASETS.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Estimated Size: {info['estimated_size']:,} images")
            print(f"  Chart Types: {', '.join(info['chart_types'])}")
        
        print("\n" + "=" * 60)
    
    def download_dataset(
        self,
        dataset_key: str,
        split: str = "train",
        limit: Optional[int] = None,
        streaming: bool = True,
    ) -> Generator[ChartImage, None, None]:
        """
        Download images from a Hugging Face dataset.
        
        Args:
            dataset_key: Key from HF_CHART_DATASETS
            split: Dataset split (train/val/test)
            limit: Max number of images to download
            streaming: Use streaming to avoid downloading entire dataset
            
        Yields:
            ChartImage objects
        """
        if not self._available:
            logger.error("datasets library not available")
            return
        
        if dataset_key not in HF_CHART_DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            logger.info(f"Available: {list(HF_CHART_DATASETS.keys())}")
            return
        
        dataset_info = HF_CHART_DATASETS[dataset_key]
        dataset_name = dataset_info["name"]
        image_column = dataset_info["image_column"]
        
        logger.info(
            f"Loading dataset | dataset={dataset_name} | split={split} | "
            f"streaming={streaming}"
        )
        
        try:
            # Load dataset (streaming to avoid full download)
            dataset = self.load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                trust_remote_code=True,
            )
            
            # Create output directories
            output_dir = IMAGES_DIR / "huggingface" / dataset_key
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_dir = METADATA_DIR / "huggingface" / dataset_key
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for idx, item in enumerate(dataset):
                if limit and count >= limit:
                    break
                
                try:
                    chart_image = self._process_item(
                        item=item,
                        idx=idx,
                        dataset_key=dataset_key,
                        dataset_info=dataset_info,
                        output_dir=output_dir,
                        metadata_dir=metadata_dir,
                    )
                    
                    if chart_image:
                        count += 1
                        yield chart_image
                        
                        if count % 100 == 0:
                            logger.info(
                                f"Progress | dataset={dataset_key} | "
                                f"downloaded={count}"
                            )
                
                except Exception as e:
                    logger.warning(
                        f"Failed to process item | idx={idx} | error={e}"
                    )
                    continue
            
            logger.success(
                f"Download complete | dataset={dataset_key} | "
                f"total={count}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load dataset | error={e}")
    
    def _process_item(
        self,
        item: Dict,
        idx: int,
        dataset_key: str,
        dataset_info: Dict,
        output_dir: Path,
        metadata_dir: Path,
    ) -> Optional[ChartImage]:
        """Process a single dataset item."""
        from PIL import Image
        
        image_column = dataset_info["image_column"]
        
        # Get image
        image = item.get(image_column)
        if image is None:
            return None
        
        # Generate unique ID
        image_id = f"hf_{dataset_key}_{idx:06d}"
        
        # Save image
        image_path = output_dir / f"{image_id}.png"
        
        if isinstance(image, Image.Image):
            image.save(image_path, "PNG")
        elif isinstance(image, dict) and "bytes" in image:
            # Some datasets store images as bytes
            with open(image_path, "wb") as f:
                f.write(image["bytes"])
        elif isinstance(image, bytes):
            with open(image_path, "wb") as f:
                f.write(image)
        else:
            logger.warning(f"Unknown image type: {type(image)}")
            return None
        
        # Extract metadata
        metadata = self._extract_metadata(item, dataset_key, dataset_info)
        
        # Infer chart type from metadata or dataset default
        chart_type = self._infer_chart_type(item, dataset_info)
        # Get image dimensions and file size
        from PIL import Image as PILImage
        
        # Load saved image to get dimensions
        with PILImage.open(image_path) as saved_img:
            img_width, img_height = saved_img.size
        
        file_size = image_path.stat().st_size
        
        # Build ChartImage
        chart_image = ChartImage(
            image_id=image_id,
            image_path=image_path,
            source=DataSource.HUGGINGFACE,
            source_url=f"hf://{dataset_info['name']}",
            chart_type=chart_type,
            caption_text=metadata.get("caption"),
            width=img_width,
            height=img_height,
            file_size_bytes=file_size,
            quality_score=0.9,  # HF datasets are generally high quality
        )
        
        # Save metadata
        metadata_path = metadata_dir / f"{image_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_id": image_id,
                    "dataset": dataset_key,
                    "original_index": idx,
                    "chart_type": chart_type.value,
                    **metadata,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        return chart_image
    
    def _extract_metadata(
        self,
        item: Dict,
        dataset_key: str,
        dataset_info: Dict,
    ) -> Dict:
        """Extract relevant metadata from dataset item."""
        metadata = {}
        
        # Common fields across chart datasets
        common_fields = [
            "question", "answer", "label", "caption", "title",
            "x_title", "y_title", "data_table", "summary",
            "source", "query", "explanation",
        ]
        
        for field in common_fields:
            if field in item and item[field]:
                metadata[field] = item[field]
        
        # Dataset-specific extraction
        if dataset_key == "chartqa":
            # ChartQA has human and machine generated Q&A
            if "human_answer" in item:
                metadata["human_answer"] = item["human_answer"]
            if "human_question" in item:
                metadata["human_question"] = item["human_question"]
        
        elif dataset_key == "plotqa":
            # PlotQA has structured data extraction
            if "data" in item:
                metadata["extracted_data"] = item["data"]
        
        elif dataset_key == "chart2text":
            # Chart2Text has summaries
            if "caption" in item:
                metadata["summary"] = item["caption"]
        
        return metadata
    
    def _infer_chart_type(self, item: Dict, dataset_info: Dict) -> ChartType:
        """Infer chart type from item or use default."""
        # Try to get from item
        type_fields = ["chart_type", "type", "plot_type", "visualization_type"]
        
        for field in type_fields:
            if field in item:
                type_str = str(item[field]).lower()
                
                # Map common variations
                type_mapping = {
                    "bar": ChartType.BAR,
                    "horizontal_bar": ChartType.BAR,
                    "vertical_bar": ChartType.BAR,
                    "grouped_bar": ChartType.GROUPED_BAR,
                    "stacked_bar": ChartType.STACKED_BAR,
                    "line": ChartType.LINE,
                    "pie": ChartType.PIE,
                    "scatter": ChartType.SCATTER,
                    "dot": ChartType.SCATTER,
                    "area": ChartType.AREA,
                    "histogram": ChartType.HISTOGRAM,
                    "donut": ChartType.DONUT,
                }
                
                for key, chart_type in type_mapping.items():
                    if key in type_str:
                        return chart_type
        
        # Default based on dataset's primary chart types
        chart_types = dataset_info.get("chart_types", ["unknown"])
        if chart_types:
            type_str = chart_types[0]
            return ChartType(type_str) if type_str in ChartType._value2member_map_ else ChartType.UNKNOWN
        
        return ChartType.UNKNOWN
    
    def download_multiple(
        self,
        datasets: List[str],
        limit_per_dataset: int = 500,
    ) -> Dict[str, int]:
        """
        Download from multiple datasets.
        
        Args:
            datasets: List of dataset keys
            limit_per_dataset: Max images per dataset
            
        Returns:
            Dict with download counts per dataset
        """
        results = {}
        
        for dataset_key in datasets:
            logger.info(f"Starting download | dataset={dataset_key}")
            
            count = 0
            for _ in self.download_dataset(dataset_key, limit=limit_per_dataset):
                count += 1
            
            results[dataset_key] = count
            logger.info(f"Completed | dataset={dataset_key} | count={count}")
        
        return results
    
    def get_quick_start_samples(self, total: int = 1000) -> int:
        """
        Quick start: Download balanced samples from multiple datasets.
        
        This is the FASTEST way to get 1,000 diverse chart images.
        
        Args:
            total: Total number of images to download
            
        Returns:
            Total number of images downloaded
        """
        if not self._available:
            logger.error("datasets library not available")
            return 0
        
        # Balanced distribution across datasets
        datasets_to_use = ["chartqa", "chart2text"]
        per_dataset = total // len(datasets_to_use)
        
        logger.info(
            f"Quick start download | target={total} | "
            f"per_dataset={per_dataset}"
        )
        
        total_downloaded = 0
        
        for dataset_key in datasets_to_use:
            try:
                count = 0
                for _ in self.download_dataset(dataset_key, limit=per_dataset):
                    count += 1
                total_downloaded += count
                
            except Exception as e:
                logger.error(f"Failed {dataset_key}: {e}")
                continue
        
        logger.success(f"Quick start complete | total={total_downloaded}")
        return total_downloaded


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def download_chartqa_quick(limit: int = 500) -> int:
    """
    Convenience function to quickly download ChartQA samples.
    
    Example:
        from tools.data_factory.services.hf_hunter import download_chartqa_quick
        count = download_chartqa_quick(500)
    """
    from ..config import DataFactoryConfig
    
    config = DataFactoryConfig()
    hunter = HuggingFaceHunter(config)
    
    count = 0
    for _ in hunter.download_dataset("chartqa", limit=limit):
        count += 1
    
    return count
