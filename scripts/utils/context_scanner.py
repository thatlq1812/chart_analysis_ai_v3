"""
Context Scanner - Scan project directory structure with size analysis.

This script scans the project tree, reports directory sizes, and identifies
which directories contain large data (to be skipped during content reading).

Output:
    - Console report with directory sizes
    - JSON report at docs/thesis_capstone/drafts/project_structure_scan.json

Usage:
    .venv/Scripts/python.exe scripts/context_scanner.py
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories to skip entirely (version control, venv, etc.)
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}

# Directories flagged as "large data" - content should NOT be read by AI
LARGE_DATA_DIRS = {"data", "runs", "models/weights", "models/onnx", "models/slm"}

SIZE_THRESHOLD_MB = 50  # Directories above this are flagged as "large"


def get_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def count_files(path: Path) -> int:
    """Count total files in a directory recursively."""
    count = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                count += 1
    except (OSError, PermissionError):
        pass
    return count


def scan_directory(root: Path, max_depth: int = 2) -> list[dict]:
    """
    Scan directory tree up to max_depth.

    Returns list of directory info dicts.
    """
    results = []

    def _scan(current: Path, depth: int, rel_prefix: str = "") -> None:
        if depth > max_depth:
            return

        try:
            entries = sorted(current.iterdir())
        except (OSError, PermissionError):
            return

        dirs = [e for e in entries if e.is_dir() and e.name not in SKIP_DIRS]
        files = [e for e in entries if e.is_file()]

        for d in dirs:
            rel_path = f"{rel_prefix}/{d.name}" if rel_prefix else d.name
            size = get_dir_size(d)
            file_count = count_files(d)
            is_large = (
                size > SIZE_THRESHOLD_MB * 1024 * 1024
                or rel_path in LARGE_DATA_DIRS
            )

            info = {
                "path": rel_path,
                "size_bytes": size,
                "size_human": format_size(size),
                "file_count": file_count,
                "depth": depth,
                "is_large_data": is_large,
                "skip_content_read": is_large,
            }
            results.append(info)

            if not is_large or depth < 2:
                _scan(d, depth + 1, rel_path)

    _scan(root, 0)
    return results


def print_tree(results: list[dict]) -> None:
    """Print a formatted directory tree with sizes."""
    logger.info("=" * 80)
    logger.info("PROJECT DIRECTORY SCAN - chart_analysis_ai_v3")
    logger.info(f"Scan time: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Directory':<45} {'Size':>10} {'Files':>7} {'Flag':>12}")
    logger.info("-" * 80)

    for item in results:
        indent = "  " * item["depth"]
        name = indent + item["path"].split("/")[-1] + "/"
        flag = "[LARGE-SKIP]" if item["is_large_data"] else ""
        logger.info(
            f"{name:<45} {item['size_human']:>10} {item['file_count']:>7} {flag:>12}"
        )

    logger.info("-" * 80)

    # Summary
    total_size = sum(r["size_bytes"] for r in results if r["depth"] == 0)
    large_dirs = [r for r in results if r["is_large_data"]]
    safe_dirs = [r for r in results if not r["is_large_data"]]

    logger.info("")
    logger.info("SUMMARY:")
    logger.info(f"  Total scanned (top-level): {format_size(total_size)}")
    logger.info(f"  Large data dirs (skip content read): {len(large_dirs)}")
    for ld in large_dirs:
        logger.info(f"    - {ld['path']}: {ld['size_human']} ({ld['file_count']} files)")
    logger.info(f"  Safe dirs (read content): {len(safe_dirs)}")
    logger.info("")
    logger.info("SAFE DIRECTORIES FOR CONTENT READING:")
    for sd in safe_dirs:
        if sd["depth"] <= 1:
            logger.info(f"    [OK] {sd['path']}/  ({sd['size_human']})")


def main() -> None:
    """Run the context scanner."""
    logger.info(f"Scanning project root: {PROJECT_ROOT}")
    results = scan_directory(PROJECT_ROOT, max_depth=2)

    print_tree(results)

    # Save JSON report
    output_dir = PROJECT_ROOT / "docs" / "thesis_capstone" / "drafts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "project_structure_scan.json"

    report = {
        "scan_time": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "size_threshold_mb": SIZE_THRESHOLD_MB,
        "skip_dirs": list(SKIP_DIRS),
        "large_data_dirs_config": list(LARGE_DATA_DIRS),
        "directories": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nJSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
