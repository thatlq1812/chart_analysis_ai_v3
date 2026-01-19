"""
Data Factory CLI - Main entry point for data collection pipeline

Commands:
- hunt: Download data from external sources (Arxiv, Google, Roboflow)
- mine: Extract chart images from downloaded PDFs
- sanitize: Filter and validate extracted images
- generate: Create synthetic chart images
- run-all: Execute complete pipeline
- stats: Show dataset statistics
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/data_factory_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

from .config import (
    DataFactoryConfig,
    RAW_PDFS_DIR,
    IMAGES_DIR,
    METADATA_DIR,
    ANNOTATIONS_DIR,
    MANIFESTS_DIR,
    LOGS_DIR,
)
from .schemas import DataManifest, ChartImage, DataSource, ProcessingStatus


def setup_directories() -> None:
    """Ensure all required directories exist."""
    for directory in [RAW_PDFS_DIR, IMAGES_DIR, METADATA_DIR, ANNOTATIONS_DIR, MANIFESTS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logger.debug("Directories initialized")


def cmd_hunt(args) -> int:
    """Hunt for data from external sources."""
    from .services.hunter import ArxivHunter, RoboflowHunter
    from .services.hf_hunter import HuggingFaceHunter
    from .services.pmc_hunter import PMCHunter
    from .services.acl_hunter import ACLHunter
    
    config = DataFactoryConfig()
    config.random_seed = args.seed
    
    sources = args.sources.split(",") if args.sources else ["arxiv"]
    total_downloaded = 0
    
    logger.info(f"Starting hunt | sources={sources} | limit={args.limit}")
    
    for source in sources:
        source = source.strip().lower()
        
        if source == "arxiv":
            hunter = ArxivHunter(config)
            papers = hunter.search(limit=args.limit)
            
            logger.info(f"Found {len(papers)} papers from Arxiv")
            
            for paper in papers[:args.limit]:
                success = hunter.download(paper)
                if success:
                    total_downloaded += 1
        
        elif source == "huggingface" or source == "hf":
            hunter = HuggingFaceHunter(config)
            
            # Quick start mode: get samples from multiple datasets
            if args.quick_start:
                images = hunter.get_quick_start_samples(
                    samples_per_dataset=args.limit or 100
                )
                total_downloaded += len(images)
            else:
                # Download specific dataset
                dataset_key = args.dataset or "chartqa"
                for image in hunter.download_dataset(
                    dataset_key=dataset_key,
                    limit=args.limit or 500,
                ):
                    total_downloaded += 1
        
        elif source == "pmc":
            hunter = PMCHunter(config)
            
            images = hunter.hunt(
                max_papers=args.limit or 50,
                max_images_per_paper=5,
            )
            total_downloaded += len(images)
        
        elif source == "acl":
            hunter = ACLHunter(config)
            
            images = hunter.hunt(
                max_papers=args.limit or 30,
                years=[2023, 2024],
            )
            total_downloaded += len(images)
                    
        elif source == "roboflow":
            if not args.api_key:
                logger.error("Roboflow requires --api-key")
                continue
                
            hunter = RoboflowHunter(config, api_key=args.api_key)
            
            for dataset_name in ["chart-datasets", "chart-classification"]:
                success = hunter.download_dataset(dataset_name)
                if success:
                    total_downloaded += 1
                    
        else:
            logger.warning(f"Unknown source: {source}")
    
    logger.info(f"Hunt complete | total_downloaded={total_downloaded}")
    return 0


def cmd_mine(args) -> int:
    """Extract chart images from PDFs."""
    from .services.miner import PDFMiner
    
    config = DataFactoryConfig()
    miner = PDFMiner(config)
    
    # Find PDFs
    pdf_dir = Path(args.input_dir) if args.input_dir else RAW_PDFS_DIR
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if args.limit:
        pdf_files = pdf_files[:args.limit]
    
    logger.info(f"Mining PDFs | count={len(pdf_files)} | dir={pdf_dir}")
    
    total_images = 0
    
    for pdf_path in pdf_files:
        try:
            images = miner.process_pdf(pdf_path)
            total_images += len(images)
            logger.info(f"Extracted images | pdf={pdf_path.name} | count={len(images)}")
        except Exception as e:
            logger.error(f"Failed to process PDF | path={pdf_path} | error={e}")
    
    logger.info(f"Mining complete | total_images={total_images}")
    return 0


def cmd_sanitize(args) -> int:
    """Filter and validate extracted images."""
    from .services.sanitizer import ImageSanitizer, ChartDetector
    
    config = DataFactoryConfig()
    sanitizer = ImageSanitizer(config)
    detector = ChartDetector()
    
    # Find images
    image_dir = Path(args.input_dir) if args.input_dir else IMAGES_DIR
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    logger.info(f"Sanitizing images | count={len(image_files)}")
    
    passed = 0
    failed = 0
    
    for image_path in image_files:
        # Quality check
        quality_ok, reasons = sanitizer.check_quality(image_path)
        
        if not quality_ok:
            logger.debug(f"Quality failed | image={image_path.name} | reasons={reasons}")
            failed += 1
            
            if args.move_failed:
                failed_dir = image_dir / "failed"
                failed_dir.mkdir(exist_ok=True)
                image_path.rename(failed_dir / image_path.name)
            continue
        
        # Chart detection (optional, slower)
        if args.detect_charts:
            is_chart = detector.is_likely_chart(image_path)
            if not is_chart:
                logger.debug(f"Not a chart | image={image_path.name}")
                failed += 1
                continue
        
        passed += 1
    
    logger.info(f"Sanitization complete | passed={passed} | failed={failed}")
    return 0


def cmd_generate(args) -> int:
    """Generate synthetic chart images."""
    from .services.generator import SyntheticChartGenerator
    from .schemas import ChartType
    
    config = DataFactoryConfig()
    config.random_seed = args.seed
    
    generator = SyntheticChartGenerator(config)
    
    # Parse chart types
    chart_types = None
    if args.types:
        chart_types = [ChartType(t.strip()) for t in args.types.split(",")]
    
    images = generator.generate_dataset(
        count=args.count,
        chart_types=chart_types,
    )
    
    logger.info(f"Generated {len(images)} synthetic charts")
    return 0


def cmd_run_all(args) -> int:
    """Execute complete data collection pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING FULL DATA FACTORY PIPELINE")
    logger.info("=" * 60)
    
    setup_directories()
    
    # Phase 1: Hunt from Arxiv
    logger.info("\n[PHASE 1/4] Hunting from Arxiv...")
    hunt_args = argparse.Namespace(
        sources="arxiv",
        limit=args.limit,
        api_key=args.api_key,
        seed=args.seed,
    )
    cmd_hunt(hunt_args)
    
    # Phase 2: Mine PDFs
    logger.info("\n[PHASE 2/4] Mining PDFs for chart images...")
    mine_args = argparse.Namespace(
        input_dir=None,
        limit=None,
    )
    cmd_mine(mine_args)
    
    # Phase 3: Sanitize images
    logger.info("\n[PHASE 3/4] Sanitizing extracted images...")
    sanitize_args = argparse.Namespace(
        input_dir=None,
        move_failed=True,
        detect_charts=True,
    )
    cmd_sanitize(sanitize_args)
    
    # Phase 4: Generate synthetic data
    if args.synthetic_count > 0:
        logger.info(f"\n[PHASE 4/4] Generating {args.synthetic_count} synthetic charts...")
        generate_args = argparse.Namespace(
            count=args.synthetic_count,
            types=None,
            seed=args.seed,
        )
        cmd_generate(generate_args)
    
    # Generate manifest
    logger.info("\n[COMPLETE] Generating dataset manifest...")
    cmd_stats(args)
    
    return 0


def cmd_stats(args) -> int:
    """Show dataset statistics and generate manifest."""
    
    # Count files
    pdf_count = len(list(RAW_PDFS_DIR.glob("*.pdf")))
    image_count = len(list(IMAGES_DIR.glob("*.png"))) + len(list(IMAGES_DIR.glob("*.jpg")))
    metadata_count = len(list(METADATA_DIR.glob("*.json")))
    annotation_count = len(list(ANNOTATIONS_DIR.glob("*.json")))
    
    # Load metadata and count by source
    source_counts = {source.value: 0 for source in DataSource}
    chart_type_counts = {}
    
    for meta_file in METADATA_DIR.glob("*.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
                source = meta.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
                
                chart_type = meta.get("chart_type")
                if chart_type:
                    chart_type_counts[chart_type] = chart_type_counts.get(chart_type, 0) + 1
        except Exception:
            pass
    
    # Print statistics
    print("\n" + "=" * 50)
    print("DATA FACTORY STATISTICS")
    print("=" * 50)
    print(f"\n{'Category':<25} {'Count':>10}")
    print("-" * 36)
    print(f"{'PDFs downloaded':<25} {pdf_count:>10}")
    print(f"{'Images extracted':<25} {image_count:>10}")
    print(f"{'Metadata files':<25} {metadata_count:>10}")
    print(f"{'Annotation files':<25} {annotation_count:>10}")
    
    print(f"\n{'By Source':<25} {'Count':>10}")
    print("-" * 36)
    for source, count in source_counts.items():
        if count > 0:
            print(f"  {source:<23} {count:>10}")
    
    if chart_type_counts:
        print(f"\n{'By Chart Type':<25} {'Count':>10}")
        print("-" * 36)
        for chart_type, count in sorted(chart_type_counts.items()):
            print(f"  {chart_type:<23} {count:>10}")
    
    print("\n" + "=" * 50)
    
    # Generate manifest with proper schema
    from .schemas import DatasetStatistics
    
    stats = DatasetStatistics(
        total_images=image_count,
        by_source=source_counts,
        by_chart_type=chart_type_counts,
        valid_images=image_count,
    )
    
    manifest = DataManifest(
        dataset_name="geo_slm_chart_dataset",
        version="1.0.0",
        statistics=stats,
    )
    
    manifest_id = f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    manifest_path = MANIFESTS_DIR / f"{manifest_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, default=str)
    
    print(f"\nManifest saved to: {manifest_path}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Factory - Automated chart dataset collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 10 papers from Arxiv
  python -m tools.data_factory.main hunt --sources arxiv --limit 10
  
  # [FASTEST] Download from HuggingFace (pre-labeled datasets)
  python -m tools.data_factory.main hunt --sources huggingface --quick-start --limit 100
  
  # Download specific HuggingFace dataset
  python -m tools.data_factory.main hunt --sources hf --dataset chartqa --limit 500
  
  # Download from PubMed Central (biomedical charts)
  python -m tools.data_factory.main hunt --sources pmc --limit 50
  
  # Download from ACL Anthology (NLP papers)
  python -m tools.data_factory.main hunt --sources acl --limit 30
  
  # Download from multiple sources
  python -m tools.data_factory.main hunt --sources huggingface,arxiv,pmc --limit 100
  
  # Extract images from all PDFs
  python -m tools.data_factory.main mine
  
  # Generate 100 synthetic charts
  python -m tools.data_factory.main generate --count 100
  
  # Run complete pipeline
  python -m tools.data_factory.main run-all --limit 10
  
  # Show statistics
  python -m tools.data_factory.main stats
        """,
    )
    
    # Global arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Hunt command
    hunt_parser = subparsers.add_parser("hunt", help="Download data from external sources")
    hunt_parser.add_argument(
        "--sources", 
        default="arxiv", 
        help="Comma-separated sources: arxiv,huggingface,pmc,acl,roboflow"
    )
    hunt_parser.add_argument("--limit", type=int, default=50, help="Maximum items to download")
    hunt_parser.add_argument("--api-key", help="API key (for Roboflow)")
    hunt_parser.add_argument("--dataset", help="Specific dataset name (for HuggingFace)")
    hunt_parser.add_argument(
        "--quick-start", 
        action="store_true", 
        help="Quick start mode: sample from multiple datasets"
    )
    
    # Mine command
    mine_parser = subparsers.add_parser("mine", help="Extract chart images from PDFs")
    mine_parser.add_argument("--input-dir", help="Directory containing PDFs")
    mine_parser.add_argument("--limit", type=int, help="Maximum PDFs to process")
    
    # Sanitize command
    sanitize_parser = subparsers.add_parser("sanitize", help="Filter and validate images")
    sanitize_parser.add_argument("--input-dir", help="Directory containing images")
    sanitize_parser.add_argument("--move-failed", action="store_true", help="Move failed images to subfolder")
    sanitize_parser.add_argument("--detect-charts", action="store_true", help="Use heuristic chart detection")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic charts")
    generate_parser.add_argument("--count", type=int, default=100, help="Number of charts to generate")
    generate_parser.add_argument("--types", help="Comma-separated chart types: bar,line,pie,scatter,area")
    
    # Run-all command
    run_all_parser = subparsers.add_parser("run-all", help="Execute complete pipeline")
    run_all_parser.add_argument("--limit", type=int, default=10, help="Limit for hunting phase")
    run_all_parser.add_argument("--api-key", help="API key for optional sources")
    run_all_parser.add_argument("--synthetic-count", type=int, default=50, help="Synthetic charts to generate")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup directories
    setup_directories()
    
    # Execute command
    commands = {
        "hunt": cmd_hunt,
        "mine": cmd_mine,
        "sanitize": cmd_sanitize,
        "generate": cmd_generate,
        "run-all": cmd_run_all,
        "stats": cmd_stats,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
