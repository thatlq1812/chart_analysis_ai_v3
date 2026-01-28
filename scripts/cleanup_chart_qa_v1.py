#!/usr/bin/env python3
"""Cleanup script for old chart_qa v1 structure after migration.

This script:
1. Archives the old dataset.json to chart_qa_archive/
2. Removes redundant folders (checkpoints, classified, qa_pairs)
3. Keeps only the latest stats file
4. Renames chart_qa to chart_qa_v1_archived
"""

import shutil
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cleanup_chart_qa_v1(
    base_dir: Path,
    dry_run: bool = True,
) -> None:
    """Clean up old chart_qa v1 structure.
    
    Args:
        base_dir: Path to academic_dataset folder
        dry_run: If True, only print actions without executing
    """
    chart_qa_dir = base_dir / "chart_qa"
    archive_dir = base_dir / "chart_qa_v1_archived"
    
    if not chart_qa_dir.exists():
        logger.warning(f"chart_qa directory not found: {chart_qa_dir}")
        return
    
    # Verify chart_qa_v2 exists
    v2_dir = base_dir / "chart_qa_v2"
    if not v2_dir.exists():
        logger.error("chart_qa_v2 not found! Run migration first.")
        return
    
    logger.info("=" * 60)
    logger.info(f"Cleanup v1 structure {'[DRY RUN]' if dry_run else ''}")
    logger.info("=" * 60)
    
    # 1. Calculate sizes
    def get_folder_size(path: Path) -> int:
        total = 0
        if path.is_file():
            return path.stat().st_size
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    
    folders_to_remove = ["checkpoints", "classified", "qa_pairs"]
    for folder in folders_to_remove:
        folder_path = chart_qa_dir / folder
        if folder_path.exists():
            size_mb = get_folder_size(folder_path) / (1024 * 1024)
            logger.info(f"  Will remove: {folder}/ ({size_mb:.1f} MB)")
    
    # Dataset.json size
    dataset_json = chart_qa_dir / "dataset.json"
    if dataset_json.exists():
        size_mb = dataset_json.stat().st_size / (1024 * 1024)
        logger.info(f"  Will archive: dataset.json ({size_mb:.1f} MB)")
    
    if dry_run:
        logger.info("\nDry run complete. Use --execute to perform cleanup.")
        return
    
    # 2. Create archive directory
    archive_dir.mkdir(exist_ok=True)
    logger.info(f"\nCreated archive directory: {archive_dir}")
    
    # 3. Move dataset.json to archive
    if dataset_json.exists():
        archive_json = archive_dir / "dataset.json"
        shutil.move(str(dataset_json), str(archive_json))
        logger.info(f"Archived: dataset.json -> {archive_json}")
    
    # 4. Archive stats folder (keep for reference)
    stats_dir = chart_qa_dir / "stats"
    if stats_dir.exists():
        archive_stats = archive_dir / "stats"
        shutil.move(str(stats_dir), str(archive_stats))
        logger.info(f"Archived: stats/ -> {archive_stats}")
    
    # 5. Remove redundant folders
    for folder in folders_to_remove:
        folder_path = chart_qa_dir / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            logger.info(f"Removed: {folder}/")
    
    # 6. Remove empty chart_qa directory
    if chart_qa_dir.exists() and not any(chart_qa_dir.iterdir()):
        chart_qa_dir.rmdir()
        logger.info(f"Removed empty directory: chart_qa/")
    
    # 7. Create a migration note
    note_path = archive_dir / "MIGRATION_NOTE.md"
    with open(note_path, "w") as f:
        f.write(f"""# Chart QA V1 Archive

Migrated to chart_qa_v2 on: {datetime.now().isoformat()}

## What was archived:
- dataset.json - Original flat dataset file
- stats/ - Processing statistics

## What was removed:
- checkpoints/ - Temporary processing checkpoints
- classified/ - Duplicate of classified_charts/
- qa_pairs/ - Individual QA pair files (now consolidated in shards)

## New structure location:
See `../chart_qa_v2/` for the sharded dataset structure.
""")
    logger.info(f"Created migration note: {note_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Cleanup complete!")
    logger.info("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cleanup old chart_qa v1 structure after migration"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the cleanup (default is dry-run)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "academic_dataset",
        help="Base directory containing chart_qa folder",
    )
    
    args = parser.parse_args()
    
    cleanup_chart_qa_v1(
        base_dir=args.base_dir,
        dry_run=not args.execute,
    )


if __name__ == "__main__":
    main()
