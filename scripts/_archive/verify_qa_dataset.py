"""
Verify QA Dataset - Compare generated JSONs with classified images.

This script analyzes:
1. JSON files in generated/ vs PNG files in classified_charts/
2. Charts that were reclassified by Gemini (type_matches_folder=False)
3. Missing files (images without JSON or vice versa)
"""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class VerificationStats:
    """Statistics from verification."""
    total_images: int = 0
    total_jsons: int = 0
    matched: int = 0
    images_without_json: List[str] = field(default_factory=list)
    jsons_without_image: List[str] = field(default_factory=list)
    reclassified: List[dict] = field(default_factory=list)
    invalid_charts: List[dict] = field(default_factory=list)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    actual_type_distribution: Dict[str, int] = field(default_factory=dict)


def get_image_ids_from_classified(classified_dir: Path, exclude_uncertain: bool = True) -> Dict[str, Set[str]]:
    """Get all image IDs from classified_charts folder."""
    image_ids = defaultdict(set)
    
    for subdir in classified_dir.iterdir():
        if not subdir.is_dir():
            continue
        if exclude_uncertain and subdir.name == "uncertain":
            continue
            
        for img_file in subdir.glob("*.png"):
            image_id = img_file.stem
            image_ids[subdir.name].add(image_id)
    
    return image_ids


def get_json_data_from_generated(generated_dir: Path, sample_size: int = 0) -> Tuple[Dict[str, Set[str]], Dict[str, dict]]:
    """Get all JSON data from generated folder.
    
    Args:
        generated_dir: Path to generated folder
        sample_size: If > 0, only load this many files per folder for analysis
    """
    json_ids = defaultdict(set)
    json_data = {}
    
    for subdir in generated_dir.iterdir():
        if not subdir.is_dir() or subdir.name == "test_batch":
            continue
        
        files = list(subdir.glob("*.json"))
        
        # Register all IDs
        for json_file in files:
            json_ids[subdir.name].add(json_file.stem)
        
        # Only load sample for analysis
        files_to_load = files[:sample_size] if sample_size > 0 else files
        
        for json_file in files_to_load:
            image_id = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                json_data[image_id] = {
                    'folder': subdir.name,
                    'data': data
                }
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    return json_ids, json_data


def verify_dataset(
    classified_dir: Path,
    generated_dir: Path,
    exclude_uncertain: bool = True
) -> VerificationStats:
    """Main verification function."""
    
    stats = VerificationStats()
    
    # Get data from both directories
    print("Loading classified images...")
    image_ids = get_image_ids_from_classified(classified_dir, exclude_uncertain)
    
    print("Loading generated JSONs...")
    json_ids, json_data = get_json_data_from_generated(generated_dir, sample_size=500)
    
    # Count totals
    all_images = set()
    for chart_type, ids in image_ids.items():
        stats.type_distribution[chart_type] = len(ids)
        all_images.update(ids)
    stats.total_images = len(all_images)
    
    all_jsons = set()
    for chart_type, ids in json_ids.items():
        all_jsons.update(ids)
    stats.total_jsons = len(all_jsons)
    
    # Find matches and mismatches
    stats.matched = len(all_images & all_jsons)
    stats.images_without_json = list(all_images - all_jsons)
    stats.jsons_without_image = list(all_jsons - all_images)
    
    # Analyze reclassified charts
    print("Analyzing reclassifications...")
    for image_id, info in json_data.items():
        data = info['data']
        verification = data.get('verification') or {}
        
        # Count actual types
        actual_type = verification.get('actual_chart_type') or data.get('chart_type') or 'unknown'
        stats.actual_type_distribution[actual_type] = stats.actual_type_distribution.get(actual_type, 0) + 1
        
        # Check if type matches
        type_matches = verification.get('type_matches_folder', True)
        if not type_matches:
            original_type = data.get('chart_type', 'unknown')
            stats.reclassified.append({
                'image_id': image_id,
                'original_type': original_type,
                'actual_type': actual_type,
                'saved_in': info['folder'],
                'is_valid': verification.get('is_valid_chart', True),
                'quality': verification.get('chart_quality', 'unknown')
            })
        
        # Check for invalid charts
        if not verification.get('is_valid_chart', True):
            stats.invalid_charts.append({
                'image_id': image_id,
                'original_type': data.get('chart_type', 'unknown'),
                'actual_type': actual_type,
                'reason': verification.get('verification_notes', 'N/A')
            })
    
    return stats


def print_report(stats: VerificationStats):
    """Print verification report."""
    
    print("\n" + "=" * 70)
    print("                    QA DATASET VERIFICATION REPORT")
    print("=" * 70)
    
    # Summary
    print("\n[1] SUMMARY")
    print("-" * 40)
    print(f"  Total classified images: {stats.total_images:,}")
    print(f"  Total generated JSONs:   {stats.total_jsons:,}")
    print(f"  Matched (both exist):    {stats.matched:,}")
    print(f"  Images without JSON:     {len(stats.images_without_json):,}")
    print(f"  JSONs without image:     {len(stats.jsons_without_image):,}")
    
    # Original type distribution
    print("\n[2] ORIGINAL TYPE DISTRIBUTION (classified_charts)")
    print("-" * 40)
    for chart_type, count in sorted(stats.type_distribution.items(), key=lambda x: -x[1]):
        print(f"  {chart_type:15s}: {count:,}")
    
    # Actual type distribution (after Gemini verification)
    print("\n[3] ACTUAL TYPE DISTRIBUTION (after Gemini verification)")
    print("-" * 40)
    for chart_type, count in sorted(stats.actual_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  {chart_type:15s}: {count:,}")
    
    # Reclassification summary
    print(f"\n[4] RECLASSIFIED CHARTS: {len(stats.reclassified):,}")
    print("-" * 40)
    
    # Group by original -> actual
    reclassify_matrix = defaultdict(lambda: defaultdict(int))
    for item in stats.reclassified:
        reclassify_matrix[item['original_type']][item['actual_type']] += 1
    
    for original in sorted(reclassify_matrix.keys()):
        actuals = reclassify_matrix[original]
        for actual, count in sorted(actuals.items(), key=lambda x: -x[1]):
            if count >= 10:  # Only show significant changes
                print(f"  {original:12s} -> {actual:15s}: {count:,}")
    
    # Invalid charts
    print(f"\n[5] INVALID CHARTS: {len(stats.invalid_charts):,}")
    print("-" * 40)
    invalid_by_type = defaultdict(int)
    for item in stats.invalid_charts:
        invalid_by_type[item['actual_type']] += 1
    for actual_type, count in sorted(invalid_by_type.items(), key=lambda x: -x[1]):
        print(f"  {actual_type:15s}: {count:,}")
    
    # Sample mismatches
    if stats.images_without_json:
        print(f"\n[6] SAMPLE IMAGES WITHOUT JSON (first 10)")
        print("-" * 40)
        for img_id in stats.images_without_json[:10]:
            print(f"  {img_id}")
    
    if stats.jsons_without_image:
        print(f"\n[7] SAMPLE JSONS WITHOUT IMAGE (first 10)")
        print("-" * 40)
        for json_id in stats.jsons_without_image[:10]:
            print(f"  {json_id}")
    
    print("\n" + "=" * 70)


def export_reclassified_list(stats: VerificationStats, output_path: Path):
    """Export list of reclassified charts for review."""
    
    output = {
        'summary': {
            'total_reclassified': len(stats.reclassified),
            'total_invalid': len(stats.invalid_charts),
        },
        'reclassified': stats.reclassified,
        'invalid_charts': stats.invalid_charts,
        'images_without_json': stats.images_without_json,
        'jsons_without_image': stats.jsons_without_image,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nExported detailed report to: {output_path}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    classified_dir = base_dir / "data" / "academic_dataset" / "classified_charts"
    generated_dir = base_dir / "data" / "academic_dataset" / "chart_qa_v2" / "generated"
    output_path = base_dir / "data" / "output" / "qa_verification_report.json"
    
    print(f"Classified charts: {classified_dir}")
    print(f"Generated JSONs:   {generated_dir}")
    
    # Run verification
    stats = verify_dataset(classified_dir, generated_dir, exclude_uncertain=True)
    
    # Print report
    print_report(stats)
    
    # Export detailed report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_reclassified_list(stats, output_path)


if __name__ == "__main__":
    main()
