"""
Migrate dataset.json (flat) to sharded structure (Option B).

This script converts the existing flat QA dataset to a scalable sharded format
organized by chart type, suitable for 50k+ images.

New Structure:
    chart_qa_v2/
    ├── metadata.json         # Header with stats, schema version
    ├── shards/
    │   ├── line/
    │   │   ├── shard_000.json  (~1000 samples each)
    │   │   └── shard_001.json
    │   ├── bar/
    │   └── ...
    └── splits/
        ├── train.txt         # Image IDs (80%)
        ├── val.txt           # Image IDs (10%)
        └── test.txt          # Image IDs (10%)

Usage:
    python scripts/migrate_dataset_v2.py
    python scripts/migrate_dataset_v2.py --dry-run
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "academic_dataset"
OLD_DATASET = DATA_DIR / "chart_qa" / "dataset.json"
NEW_DATASET_DIR = DATA_DIR / "chart_qa_v2"
CLASSIFIED_CHARTS_DIR = DATA_DIR / "classified_charts"

# Config
SHARD_SIZE = 1000  # Samples per shard
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42

# Schema version
SCHEMA_VERSION = "2.0.0"


def load_old_dataset(path: Path) -> dict[str, Any]:
    """Load the existing flat dataset.json."""
    logger.info(f"Loading old dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        f"Loaded {data.get('total_images', 0)} images, "
        f"{data.get('total_qa_pairs', 0)} QA pairs"
    )
    return data


def get_chart_type_from_path(image_id: str) -> str | None:
    """
    Determine chart type by checking which classified_charts subfolder contains the image.
    This ensures consistency with the classification results.
    """
    for type_dir in CLASSIFIED_CHARTS_DIR.iterdir():
        if type_dir.is_dir():
            # Check for image file (could be .png or .jpg)
            for ext in [".png", ".jpg", ".jpeg"]:
                if (type_dir / f"{image_id}{ext}").exists():
                    return type_dir.name
    return None


def convert_to_relative_path(abs_path: str) -> str:
    """Convert absolute Windows path to relative path."""
    # Extract image filename from absolute path
    path = Path(abs_path)
    image_id = path.stem
    ext = path.suffix
    
    # Return relative path from project root
    # Will be like: data/academic_dataset/images/arxiv_xxx.png
    return f"images/{image_id}{ext}"


def group_samples_by_type(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by chart type."""
    grouped = defaultdict(list)
    unknown_type_count = 0
    
    for sample in samples:
        image_id = sample["image_id"]
        
        # First try the chart_type from the sample
        chart_type = sample.get("chart_type", "unknown")
        
        # Verify by checking classified_charts directory
        actual_type = get_chart_type_from_path(image_id)
        if actual_type and actual_type != chart_type:
            logger.warning(
                f"Type mismatch for {image_id}: "
                f"dataset says '{chart_type}', folder says '{actual_type}'"
            )
            chart_type = actual_type
        elif not actual_type:
            # Image not found in classified_charts - use dataset type
            unknown_type_count += 1
        
        # Convert to new format
        new_sample = {
            "image_id": image_id,
            "image_path": convert_to_relative_path(sample["image_path"]),
            "chart_type": chart_type,
            "qa_pairs": sample["qa_pairs"],
            "generated_at": sample.get("generated_at", ""),
        }
        
        grouped[chart_type].append(new_sample)
    
    if unknown_type_count > 0:
        logger.warning(f"{unknown_type_count} images not found in classified_charts")
    
    return dict(grouped)


def create_shards(
    samples: list[dict], chart_type: str, shard_size: int = SHARD_SIZE
) -> list[dict]:
    """Split samples into shards."""
    shards = []
    for i in range(0, len(samples), shard_size):
        shard_samples = samples[i : i + shard_size]
        shard = {
            "chart_type": chart_type,
            "shard_index": len(shards),
            "sample_count": len(shard_samples),
            "samples": shard_samples,
        }
        shards.append(shard)
    return shards


def create_splits(
    grouped_samples: dict[str, list[dict]], seed: int = RANDOM_SEED
) -> dict[str, list[str]]:
    """
    Create train/val/test splits stratified by chart type.
    Returns dict with lists of image_ids.
    """
    random.seed(seed)
    
    splits = {"train": [], "val": [], "test": []}
    
    for chart_type, samples in grouped_samples.items():
        # Shuffle samples for this type
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * SPLIT_RATIOS["train"])
        val_end = train_end + int(n * SPLIT_RATIOS["val"])
        
        # Extract image_ids for each split
        for sample in shuffled[:train_end]:
            splits["train"].append(sample["image_id"])
        for sample in shuffled[train_end:val_end]:
            splits["val"].append(sample["image_id"])
        for sample in shuffled[val_end:]:
            splits["test"].append(sample["image_id"])
    
    # Shuffle each split to mix chart types
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    return splits


def create_metadata(
    grouped_samples: dict[str, list[dict]],
    splits: dict[str, list[str]],
) -> dict:
    """Create metadata.json content."""
    # Calculate stats
    total_images = sum(len(samples) for samples in grouped_samples.values())
    total_qa_pairs = sum(
        len(s["qa_pairs"])
        for samples in grouped_samples.values()
        for s in samples
    )
    
    type_distribution = {
        chart_type: len(samples)
        for chart_type, samples in sorted(grouped_samples.items())
    }
    
    split_counts = {name: len(ids) for name, ids in splits.items()}
    
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "migrated_from": "chart_qa/dataset.json v1.0.0",
        "description": "Sharded Chart QA dataset organized by chart type",
        "statistics": {
            "total_images": total_images,
            "total_qa_pairs": total_qa_pairs,
            "chart_type_distribution": type_distribution,
            "split_distribution": split_counts,
        },
        "config": {
            "shard_size": SHARD_SIZE,
            "split_ratios": SPLIT_RATIOS,
            "random_seed": RANDOM_SEED,
        },
        "paths": {
            "shards": "shards/{chart_type}/shard_{index:03d}.json",
            "splits": "splits/{split_name}.txt",
            "images_base": "../classified_charts/{chart_type}/",
        },
    }


def write_sharded_dataset(
    grouped_samples: dict[str, list[dict]],
    splits: dict[str, list[str]],
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Write the new sharded dataset structure."""
    
    if dry_run:
        logger.info("[DRY RUN] Would create directory structure at: %s", output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "shards").mkdir(exist_ok=True)
        (output_dir / "splits").mkdir(exist_ok=True)
    
    # Write metadata
    metadata = create_metadata(grouped_samples, splits)
    metadata_path = output_dir / "metadata.json"
    
    if dry_run:
        logger.info("[DRY RUN] Would write metadata.json with stats:")
        logger.info(f"  Total images: {metadata['statistics']['total_images']}")
        logger.info(f"  Total QA pairs: {metadata['statistics']['total_qa_pairs']}")
    else:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote metadata to {metadata_path}")
    
    # Write shards by chart type
    total_shards = 0
    for chart_type, samples in grouped_samples.items():
        type_dir = output_dir / "shards" / chart_type
        
        if not dry_run:
            type_dir.mkdir(parents=True, exist_ok=True)
        
        shards = create_shards(samples, chart_type)
        total_shards += len(shards)
        
        for shard in shards:
            shard_path = type_dir / f"shard_{shard['shard_index']:03d}.json"
            
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would write {shard_path.name} "
                    f"({shard['sample_count']} samples)"
                )
            else:
                with open(shard_path, "w", encoding="utf-8") as f:
                    json.dump(shard, f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Chart type '{chart_type}': {len(samples)} samples -> "
            f"{len(shards)} shards"
        )
    
    # Write split files
    for split_name, image_ids in splits.items():
        split_path = output_dir / "splits" / f"{split_name}.txt"
        
        if dry_run:
            logger.info(f"[DRY RUN] Would write {split_path.name} ({len(image_ids)} IDs)")
        else:
            with open(split_path, "w", encoding="utf-8") as f:
                f.write("\n".join(image_ids))
            logger.info(f"Wrote {split_name} split: {len(image_ids)} images")
    
    logger.info(f"Total: {total_shards} shard files created")


def main(dry_run: bool = False) -> None:
    """Main migration function."""
    logger.info("=" * 60)
    logger.info("Starting dataset migration to v2 (sharded)")
    logger.info("=" * 60)
    
    # Check source exists
    if not OLD_DATASET.exists():
        logger.error(f"Source dataset not found: {OLD_DATASET}")
        return
    
    # Load old dataset
    old_data = load_old_dataset(OLD_DATASET)
    
    if "samples" not in old_data:
        logger.error("No 'samples' key in dataset.json")
        return
    
    samples = old_data["samples"]
    logger.info(f"Processing {len(samples)} samples...")
    
    # Group by chart type
    grouped = group_samples_by_type(samples)
    logger.info(f"Found {len(grouped)} chart types: {list(grouped.keys())}")
    
    # Create train/val/test splits
    splits = create_splits(grouped)
    logger.info(
        f"Splits created - train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )
    
    # Write new structure
    write_sharded_dataset(grouped, splits, NEW_DATASET_DIR, dry_run=dry_run)
    
    logger.info("=" * 60)
    if dry_run:
        logger.info("[DRY RUN] Migration preview complete. No files written.")
        logger.info(f"Run without --dry-run to create: {NEW_DATASET_DIR}")
    else:
        logger.info("Migration complete!")
        logger.info(f"New dataset location: {NEW_DATASET_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
