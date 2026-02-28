#!/usr/bin/env python3
"""Quick data audit script."""
from pathlib import Path
import json

base = Path("data/academic_dataset")

print("TOP LEVEL DIRS:")
for d in sorted(base.iterdir()):
    if d.is_dir():
        print(f"  {d.name}/")

print("\nIMAGE COUNTS:")
for subdir in ["classified_charts", "detected_charts", "images"]:
    p = base / subdir
    if p.exists():
        subdirs = [d.name for d in p.iterdir() if d.is_dir()]
        imgs = sum(1 for f in p.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
        print(f"  {subdir}: {imgs} images, subdirs={subdirs[:12]}")
    else:
        print(f"  {subdir}: MISSING")

print("\nQA JSON COUNTS (generated/):")
qa_base = base / "chart_qa_v2" / "generated"
for subdir in sorted(qa_base.iterdir()):
    if subdir.is_dir():
        n = sum(1 for f in subdir.iterdir() if f.suffix == ".json")
        print(f"  {subdir.name}: {n} json files")

print("\nSTAGE3 FEATURES:")
s3 = base / "stage3_features"
total_s3 = 0
for subdir in sorted(s3.iterdir()):
    if subdir.is_dir():
        n = sum(1 for f in subdir.iterdir() if f.suffix == ".json")
        total_s3 += n
        print(f"  {subdir.name}: {n} json files")
print(f"  TOTAL: {total_s3}")

print("\nSEARCH CACHE:")
for fp in sorted(Path("data/search_cache").glob("*.json")):
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    # Print concise info
    interesting = {k: d[k] for k in list(d.keys())[:6] if k in [
        "total_downloaded", "downloaded", "total_papers", "processed",
        "total_processed", "total_detected", "total_extracted",
        "completed", "last_updated", "status"
    ]}
    print(f"  {fp.name}: {interesting if interesting else list(d.keys())[:6]}")

print("\nMANIFESTS:")
for fp in sorted((base / "manifests").glob("*.json")):
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        print(f"  {fp.name}: {len(d)} entries")
    else:
        print(f"  {fp.name}: {d}")
