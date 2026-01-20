#!/usr/bin/env python3
"""Fix duplicate labels in YOLO label files."""

from pathlib import Path

labels_dir = Path("D:/elix/chart_analysis_ai_v3/data/training/labels")
fixed = 0

for split in ["train", "val", "test"]:
    split_dir = labels_dir / split
    if not split_dir.exists():
        continue
    for txt_file in split_dir.glob("*.txt"):
        lines = txt_file.read_text().strip().split("\n")
        if len(lines) > 1:
            txt_file.write_text(lines[0] + "\n")
            fixed += 1

print(f"Fixed {fixed} files with duplicate labels")
