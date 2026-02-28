#!/usr/bin/env python
"""
Batch download Arxiv PDFs with progress tracking and resume capability.

Usage:
    python scripts/download_arxiv_batch.py --limit 5000

This script:
1. Searches Arxiv for papers with charts
2. Downloads PDFs with rate limiting
3. Saves progress to resume if interrupted
4. Logs all activity
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)
logger.add(
    PROJECT_ROOT / "logs" / "arxiv_batch_{time:YYYY-MM-DD}.log",
    rotation="100 MB",
    retention="30 days",
    level="DEBUG",
)

from tools.data_factory.config import DataFactoryConfig, RAW_PDFS_DIR
from tools.data_factory.services.hunter import ArxivHunter


def load_progress(progress_file: Path) -> dict:
    """Load progress from checkpoint file."""
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "last_query_idx": 0}


def save_progress(progress_file: Path, progress: dict) -> None:
    """Save progress to checkpoint file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch download Arxiv PDFs")
    parser.add_argument("--limit", type=int, default=500, help="Total papers to download")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Setup
    config = DataFactoryConfig()
    config.ensure_directories()
    hunter = ArxivHunter(config)
    
    progress_file = PROJECT_ROOT / "data" / "arxiv_progress.json"
    
    # Load or initialize progress
    if args.resume:
        progress = load_progress(progress_file)
        logger.info(f"Resuming | already_downloaded={len(progress['downloaded'])}")
    else:
        progress = {"downloaded": [], "failed": [], "last_query_idx": 0}
    
    downloaded_ids = set(progress["downloaded"])
    target = args.limit
    
    logger.info("=" * 60)
    logger.info(f"ARXIV BATCH DOWNLOAD | target={target}")
    logger.info("=" * 60)
    
    # Search for papers
    logger.info("Searching Arxiv for papers...")
    try:
        papers = hunter.search(limit=target * 2)  # Get extra in case of failures
        logger.info(f"Found {len(papers)} papers")
    except Exception as e:
        logger.error(f"Search failed | error={e}")
        return 1
    
    # Download PDFs
    success_count = len(downloaded_ids)
    fail_count = len(progress["failed"])
    
    for i, paper in enumerate(papers):
        if success_count >= target:
            break
            
        if paper.arxiv_id in downloaded_ids:
            continue
        
        try:
            success = hunter.download(paper)
            
            if success:
                success_count += 1
                progress["downloaded"].append(paper.arxiv_id)
                logger.info(
                    f"[{success_count}/{target}] Downloaded | "
                    f"arxiv_id={paper.arxiv_id}"
                )
            else:
                fail_count += 1
                progress["failed"].append(paper.arxiv_id)
                logger.warning(f"Failed | arxiv_id={paper.arxiv_id}")
            
            # Save progress every 10 downloads
            if success_count % 10 == 0:
                save_progress(progress_file, progress)
                
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving progress...")
            save_progress(progress_file, progress)
            return 130
            
        except Exception as e:
            logger.error(f"Error | arxiv_id={paper.arxiv_id} | error={e}")
            progress["failed"].append(paper.arxiv_id)
            fail_count += 1
    
    # Final save
    save_progress(progress_file, progress)
    
    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed:  {fail_count}")
    logger.info(f"  Progress saved to: {progress_file}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
