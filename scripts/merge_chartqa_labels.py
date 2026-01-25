"""
Merge chart_type labels from ChartQA dataset into metadata files.

This updates metadata/*.json files with chart_type from chart_qa/dataset.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_chartqa_labels(chartqa_path: Path) -> Dict[str, str]:
    """Load chart_type labels from ChartQA dataset."""
    with open(chartqa_path, encoding="utf-8") as f:
        data = json.load(f)
    
    labels = {}
    for sample in data["samples"]:
        image_id = sample["image_id"]
        chart_type = sample["chart_type"]
        labels[image_id] = chart_type
    
    logger.info(f"Loaded {len(labels)} labels from ChartQA")
    return labels


def update_metadata(metadata_dir: Path, labels: Dict[str, str]):
    """Update metadata files with ChartQA labels."""
    metadata_files = list(metadata_dir.glob("*.json"))
    updated = 0
    not_found = 0
    
    for meta_file in metadata_files:
        try:
            with open(meta_file, encoding="utf-8") as f:
                chart = json.load(f)
            
            image_id = chart.get("image_id", "")
            if image_id in labels:
                chart["chart_type"] = labels[image_id]
                
                # Save updated metadata
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(chart, f, indent=2, ensure_ascii=False)
                
                updated += 1
            else:
                not_found += 1
        
        except Exception as e:
            logger.warning(f"Failed to update {meta_file.name}: {e}")
    
    logger.info(f"Updated {updated} metadata files")
    logger.info(f"Not found in ChartQA: {not_found}")


def main():
    base_dir = Path("data/academic_dataset")
    chartqa_path = base_dir / "chart_qa" / "dataset.json"
    metadata_dir = base_dir / "metadata"
    
    logger.info("Loading ChartQA labels...")
    labels = load_chartqa_labels(chartqa_path)
    
    logger.info("Updating metadata files...")
    update_metadata(metadata_dir, labels)
    
    logger.info("\n[SUCCESS] Labels merged!")
    logger.info("Next: Re-run prepare_training_data.py to create new manifests")


if __name__ == "__main__":
    main()
