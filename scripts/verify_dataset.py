"""
Dataset Verification Script

This script verifies the extracted dataset from both HuggingFace and ArXiv sources.
It counts valid images and checks metadata quality.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --show-samples 10
    python scripts/verify_dataset.py --check-captions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def count_images_in_dir(directory: Path, extensions: List[str] = None) -> int:
    """Count images in a directory."""
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".webp"]
    
    count = 0
    if directory.exists():
        for ext in extensions:
            count += len(list(directory.glob(f"*{ext}")))
            count += len(list(directory.glob(f"**/*{ext}")))  # Recursive
    return count


def get_dataset_stats() -> Dict[str, Dict]:
    """Get statistics for all dataset sources."""
    data_root = project_root / "data"
    
    stats = {
        "huggingface": {
            "chartqa": {
                "path": data_root / "academic_dataset" / "images" / "huggingface" / "chartqa",
                "count": 0,
                "metadata_count": 0,
            },
        },
        "arxiv": {
            "mined_images": {
                "path": data_root / "academic_dataset" / "images",
                "count": 0,
                "metadata_count": 0,
            },
            "raw_pdfs": {
                "path": data_root / "raw_pdfs",
                "count": 0,
            },
        },
        "metadata": {
            "path": data_root / "academic_dataset" / "metadata",
            "count": 0,
        },
    }
    
    # Count HuggingFace ChartQA
    hf_path = stats["huggingface"]["chartqa"]["path"]
    if hf_path.exists():
        stats["huggingface"]["chartqa"]["count"] = count_images_in_dir(hf_path)
    
    # Count ArXiv mined images
    arxiv_img_path = stats["arxiv"]["mined_images"]["path"]
    if arxiv_img_path.exists():
        # Only count direct images (not in subdirectories like huggingface/)
        for ext in [".png", ".jpg", ".jpeg"]:
            stats["arxiv"]["mined_images"]["count"] += len(list(arxiv_img_path.glob(f"*{ext}")))
    
    # Count raw PDFs
    pdf_path = stats["arxiv"]["raw_pdfs"]["path"]
    if pdf_path.exists():
        stats["arxiv"]["raw_pdfs"]["count"] = len(list(pdf_path.glob("*.pdf")))
    
    # Count metadata files
    meta_path = stats["metadata"]["path"]
    if meta_path.exists():
        stats["metadata"]["count"] = len(list(meta_path.glob("*.json")))
    
    return stats


def load_sample_metadata(limit: int = 5) -> List[Dict]:
    """Load sample metadata JSON files."""
    metadata_dir = project_root / "data" / "academic_dataset" / "metadata"
    
    samples = []
    if metadata_dir.exists():
        json_files = list(metadata_dir.glob("*.json"))[:limit]
        
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    samples.append({
                        "file": json_path.name,
                        "data": data,
                    })
            except Exception as e:
                samples.append({
                    "file": json_path.name,
                    "error": str(e),
                })
    
    return samples


def check_caption_quality(limit: int = 100) -> Dict:
    """Check caption quality in metadata files."""
    metadata_dir = project_root / "data" / "academic_dataset" / "metadata"
    
    results = {
        "total_checked": 0,
        "has_caption": 0,
        "empty_caption": 0,
        "has_context": 0,
        "empty_context": 0,
        "caption_lengths": [],
    }
    
    if not metadata_dir.exists():
        return results
    
    json_files = list(metadata_dir.glob("*.json"))[:limit]
    
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results["total_checked"] += 1
            
            caption = data.get("caption_text")
            context = data.get("context_text")
            
            if caption and len(caption.strip()) > 0:
                results["has_caption"] += 1
                results["caption_lengths"].append(len(caption))
            else:
                results["empty_caption"] += 1
            
            if context and len(context.strip()) > 0:
                results["has_context"] += 1
            else:
                results["empty_context"] += 1
                
        except Exception:
            continue
    
    # Calculate average caption length
    if results["caption_lengths"]:
        results["avg_caption_length"] = sum(results["caption_lengths"]) / len(results["caption_lengths"])
    else:
        results["avg_caption_length"] = 0
    
    return results


def print_report(stats: Dict, samples: List[Dict] = None, caption_quality: Dict = None):
    """Print verification report."""
    print("\n" + "=" * 60)
    print("  DATASET VERIFICATION REPORT")
    print("=" * 60)
    
    # HuggingFace stats
    print("\n[1] HUGGINGFACE DATASETS")
    print("-" * 40)
    hf_total = 0
    for name, info in stats["huggingface"].items():
        count = info["count"]
        hf_total += count
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {name}: {count:,} images [{status}]")
    print(f"  TOTAL: {hf_total:,} images")
    
    # ArXiv stats
    print("\n[2] ARXIV DATA")
    print("-" * 40)
    arxiv_mined = stats["arxiv"]["mined_images"]["count"]
    arxiv_pdfs = stats["arxiv"]["raw_pdfs"]["count"]
    print(f"  Raw PDFs: {arxiv_pdfs:,} files")
    print(f"  Mined Images: {arxiv_mined:,} images")
    if arxiv_pdfs > 0:
        ratio = arxiv_mined / arxiv_pdfs
        print(f"  Extraction Ratio: {ratio:.1f} images/PDF")
    
    # Metadata stats
    print("\n[3] METADATA")
    print("-" * 40)
    meta_count = stats["metadata"]["count"]
    print(f"  JSON Metadata Files: {meta_count:,}")
    
    # Caption quality
    if caption_quality:
        print("\n[4] CAPTION QUALITY CHECK")
        print("-" * 40)
        total = caption_quality["total_checked"]
        has_cap = caption_quality["has_caption"]
        has_ctx = caption_quality["has_context"]
        avg_len = caption_quality["avg_caption_length"]
        
        if total > 0:
            print(f"  Checked: {total} samples")
            print(f"  Has Caption: {has_cap}/{total} ({100*has_cap/total:.1f}%)")
            print(f"  Has Context: {has_ctx}/{total} ({100*has_ctx/total:.1f}%)")
            print(f"  Avg Caption Length: {avg_len:.0f} chars")
        else:
            print("  No metadata files to check")
    
    # Sample metadata
    if samples:
        print("\n[5] SAMPLE METADATA (First 5)")
        print("-" * 40)
        for i, sample in enumerate(samples):
            print(f"\n  Sample {i+1}: {sample['file']}")
            if "error" in sample:
                print(f"    ERROR: {sample['error']}")
            else:
                data = sample["data"]
                print(f"    - image_id: {data.get('image_id', 'N/A')}")
                print(f"    - source: {data.get('source', 'N/A')}")
                print(f"    - size: {data.get('width', '?')}x{data.get('height', '?')}")
                caption = data.get("caption_text", "")
                if caption:
                    caption_preview = caption[:80] + "..." if len(caption) > 80 else caption
                    print(f"    - caption: \"{caption_preview}\"")
                else:
                    print("    - caption: [EMPTY]")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_images = hf_total + arxiv_mined
    print(f"  Total Images: {total_images:,}")
    print(f"    - From HuggingFace: {hf_total:,}")
    print(f"    - From ArXiv Mining: {arxiv_mined:,}")
    print(f"  Total PDFs Processed: {arxiv_pdfs:,}")
    print(f"  Metadata Files: {meta_count:,}")
    
    if total_images > 0:
        print(f"\n  STATUS: DATASET READY FOR NEXT PHASE")
    else:
        print(f"\n  STATUS: DATASET EMPTY - RUN MINING FIRST")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Verify dataset extraction results")
    parser.add_argument(
        "--show-samples", 
        type=int, 
        default=5,
        help="Number of sample metadata to show (default: 5)"
    )
    parser.add_argument(
        "--check-captions",
        action="store_true",
        help="Check caption quality in metadata"
    )
    parser.add_argument(
        "--caption-limit",
        type=int,
        default=100,
        help="Number of files to check for caption quality (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Gather stats
    print("Gathering dataset statistics...")
    stats = get_dataset_stats()
    
    # Load samples
    samples = None
    if args.show_samples > 0:
        samples = load_sample_metadata(args.show_samples)
    
    # Check captions
    caption_quality = None
    if args.check_captions:
        print("Checking caption quality...")
        caption_quality = check_caption_quality(args.caption_limit)
    
    # Print report
    print_report(stats, samples, caption_quality)


if __name__ == "__main__":
    main()
